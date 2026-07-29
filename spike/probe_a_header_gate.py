"""候选 A：用 docling 自己的 `column_header` 标记排除表头，量黄牌精度。

═══ 为什么要这一步 ═══

`tall_rows` 现在排除表头用的是 `r0 < 1` —— **只排第 0 行**。
阈值验证集（53 篇）实测 22 处检测里 **10 处在行 1、3 处在行 2**，
逐条看全是**多级表头的第二/第三行**（`Undetectable/Low/High` + `(n=3)/(n=60)/(n=16)`、
`Geographic/origin` + `Number/of HCCs`、`Tumors,/eye` + `m=/1`）。
多级表头 docling 会拼平成一格（AGENTS.md 事实 #16），行高天然偏高，于是全部误报。

`docling_core` 的 `TableCell` 有 **`column_header: bool`** 字段 ——
用库自己的表头判定，比我另造一套几何启发式可靠得多，也不用再定一个阈值。
我们的缓存当初没存这个字段，所以这个探针自建缓存。

═══ 判据变化 ═══

原：`r0 < 1` 跳过
新：跳过任何 `column_header=True` 的 cell 所在的行（表头可能占多行）

只测**检测**（黄牌），不测拆分 —— 用户批的第 4 条是「只检测不拆分」那一版。

用法：
    conda run -n gemini python spike/probe_a_header_gate.py            # 关键 7 篇
    conda run -n gemini python spike/probe_a_header_gate.py --all      # 三个语料全跑
"""

from __future__ import annotations

import argparse
import pickle
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import geom  # noqa: E402
from pdf_table_extract import engine_docling  # noqa: E402

CACHE = Path("/tmp/pte_a_header_cache")
CACHE.mkdir(exist_ok=True)

DEV = ROOT / "0.pdf_input"
HOLDOUT = Path("/Users/funanhe/Downloads/Zotero_Precancer_PDFs")
CORPUS3 = Path("/Users/funanhe/Downloads/pte_threshold_corpus")

# 阈值验证集上报过检测的那几篇 + 开发集里唯一的真阳性所在篇。
# 先用它们快速判断 `column_header` 能不能把两类分开，再决定要不要全跑 90 篇。
KEY = [
    (CORPUS3, "Hiyama_1995"), (CORPUS3, "Lucchesi_1968"), (CORPUS3, "Ng et al"),
    (CORPUS3, "Nowell"), (CORPUS3, "Knudson_1971"), (CORPUS3, "Dobzhansky_1946"),
    (CORPUS3, "Rovillain"), (DEV, "pbc_21296"),
]

CELL_TALL = 1.5
HEIGHT_EVEN_MAX = 0.30
MIN_EVEN_SAMPLES = 4


def dump(pdf: Path) -> list[dict]:
    """跑 docling，**多存 `column_header`**（原缓存没有这个字段）。"""
    ck = CACHE / f"{pdf.stem[:60]}.pkl"
    if ck.exists():
        return pickle.loads(ck.read_bytes())
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    from pdf_table_extract import quiet

    quiet.hush_loggers()
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = True
    conv = DocumentConverter(format_options={"pdf": PdfFormatOption(pipeline_options=opts)})
    out = []
    with geom.normalized_pdf(pdf) as (norm, _info):
        doc = conv.convert(str(norm)).document
        for ti in doc.tables:
            if not ti.prov:
                continue
            prov = ti.prov[0]
            ph = float(doc.pages[prov.page_no].size.height)
            cells = []
            for c in (ti.data.table_cells or []):
                bb = None
                if c.bbox:
                    r = engine_docling._to_topleft(c.bbox, ph)
                    bb = (r.x0, r.y0, r.x1, r.y1)
                cells.append({"text": c.text or "", "r0": c.start_row_offset_idx,
                              "c0": c.start_col_offset_idx, "rspan": c.row_span,
                              "cspan": c.col_span, "bbox": bb,
                              "hdr": bool(getattr(c, "column_header", False))})
            out.append({"page": prov.page_no, "cells": cells})
    ck.write_bytes(pickle.dumps(out))
    return out


def column_evenness(items: list[tuple[int, float]], skip_row: int) -> float | None:
    hs = [h for r, h in items if r != skip_row and r >= 1]
    if len(hs) < MIN_EVEN_SAMPLES:
        return None
    med = statistics.median(hs)
    if med <= 0:
        return None
    return (max(hs) - min(hs)) / med


def tall_rows(cells: list[dict], header_mode: str) -> dict[int, list[int]]:
    """`header_mode`: `row0` = 现状（只排第 0 行）；`flag` = 用 docling 的 column_header。"""
    hdr_rows = {c["r0"] for c in cells if c["hdr"]} if header_mode == "flag" else set()
    by_col: dict[int, list[tuple[int, float]]] = {}
    for c in cells:
        if not c["bbox"] or c["rspan"] != 1:
            continue
        h = abs(c["bbox"][3] - c["bbox"][1])
        if h > 0:
            by_col.setdefault(c["c0"], []).append((c["r0"], h))
    sus: dict[int, list[int]] = {}
    for col, items in by_col.items():
        if len(items) < 6:
            continue
        med = statistics.median(h for _, h in items)
        if med <= 0:
            continue
        for r0, h in items:
            if r0 < 1 or r0 in hdr_rows or h <= CELL_TALL * med:
                continue
            ev = column_evenness(items, r0)
            if ev is None or ev > HEIGHT_EVEN_MAX:
                continue
            sus.setdefault(r0, []).append(col)
    return {r: cols for r, cols in sus.items() if len(cols) >= 2}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="三个语料全跑（90 篇，约 15 分钟）")
    args = ap.parse_args()

    targets: list[Path] = []
    if args.all:
        for d in (DEV, HOLDOUT, CORPUS3):
            targets += sorted(d.glob("*.pdf"))
    else:
        for d, pat in KEY:
            hit = [p for p in sorted(d.glob("*.pdf")) if pat in p.stem]
            targets += hit[:1]

    print(f"{len(targets)} 篇\n")
    print("%-40s %5s %6s %6s  %s" % ("论文", "页", "现状", "用flag", "该行的格子内容（人核用）"))
    print("-" * 118)
    tot_old = tot_new = 0
    for pdf in targets:
        try:
            tables = dump(pdf)
        except Exception as exc:  # noqa: BLE001
            print(f"  {pdf.stem[:40]}: docling 失败 {type(exc).__name__}")
            continue
        for t in tables:
            old = tall_rows(t["cells"], "row0")
            new = tall_rows(t["cells"], "flag")
            if not old and not new:
                continue
            tot_old += len(old)
            tot_new += len(new)
            hdr_rows = sorted({c["r0"] for c in t["cells"] if c["hdr"]})
            for r in sorted(set(old) | set(new)):
                mark = ("现状+flag" if r in old and r in new
                        else "只现状(flag 已排除)" if r in old else "只flag")
                texts = [c["text"][:16] for c in t["cells"]
                         if c["r0"] == r and c["text"].strip()][:5]
                print("%-40s %5s  行%-4d %-22s %s" %
                      (pdf.stem[:40], t["page"], r, mark, texts))
                if r in old and r not in new:
                    print("%53s（docling 判为表头的行: %s）" % ("", hdr_rows[:6])
                          if hdr_rows else "%53s（docling 没标任何表头行）" % "")
    print("-" * 118)
    print(f"检测处数：现状(只排第0行) **{tot_old}** → 用 column_header **{tot_new}**")
    return 0


if __name__ == "__main__":
    sys.exit(main())
