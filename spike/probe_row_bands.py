"""探针：用 PDF 词坐标验证 docling 的行切分对不对。

    conda run -n gemini python spike/probe_row_bands.py                # 全部
    conda run -n gemini python spike/probe_row_bands.py --only pbc_21296
    conda run -n gemini python spike/probe_row_bands.py --dump-cells pbc_21296:4

**不改包代码。** 范式照抄 `spike/probe_candidate_fixes.py`：并排跑、输出两栏
（修复效果 / 回归影响），回归必须为 0。

═══ 要回答的问题 ═══

现有零标注判据（`eval/checks.py`）只问「格子里的字串在文字层里存在吗」，对**位置错误
天然全盲** —— F-001 把两个数据行压进一格，字符一个没丢，三道防线全过，标了 `high`。
开发集 12 张文字表全是 100% 命中、零告警，判据已饱和。

本探针补上缺的维度：**词在纸上的坐标**。核心对比是一对整数 ——

    坐标行带数   vs   docling 报的数据行数

差 0 = 一致；**带 > docling = docling 少了行（压行，F-001）**；带 < docling = 多出行。

═══ 两个检测器，成本差一个数量级 ═══

**D-cell（便宜）**：`ti.data.table_cells` 每个 cell 自带 bbox。压行的那个 cell 会**明显比
同列其他 cell 高**。实测 `pbc_21296` p4 首列：中位高 8.52，`ALL-8 ALL-16` 那格 **15.78
= 1.85x**，且是全列**唯一**超过 1.5x 的。三行代码就能检出，而这份数据 docling 一直在算、
代码从来没读过（`engine_docling._df_to_rows` 只用 `export_to_dataframe()`，只返回字符串）。

**D-band（通用）**：把词按行轴聚类成带，跟 docling 行数比。比 D-cell 贵，但不依赖
docling 的 bbox 诚不诚实，且能顺带查断列。

两个都跑，互为佐证。
"""

from __future__ import annotations

import argparse
import pickle
import statistics
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import geom  # noqa: E402
from pdf_table_extract import engine_docling, quiet  # noqa: E402
from pdf_table_extract.models import Rect  # noqa: E402

CACHE = Path("/tmp/pte_rowbands_cache")

# —— 阈值，全部待普查，不要当结论 ——
GAP_M1 = 3.0      # M1 中心聚类的带间隙（用户在 pbc_21296 上实测 4.0 可行，这里收紧看余量）
GAP_M2 = 1.0      # M2 区间合并的间隙
CELL_TALL = 1.5   # D-cell：高于同列中位数这么多倍就算可疑
MIN_COL_GAP = 2.0 # 列投影剖面的最小空白宽度


# ————————————————————————— docling 侧（带缓存） —————————————————————————


def docling_dump(pdf: Path) -> list[dict]:
    """跑一次 docling，把每张表的 rect / rows / table_cells 存成纯数据。

    缓存到 /tmp —— docling 每篇 8-14 秒，反复迭代判据时不该反复付这个钱。
    """
    CACHE.mkdir(exist_ok=True)
    ck = CACHE / f"{pdf.stem}.pkl"
    if ck.exists():
        return pickle.loads(ck.read_bytes())

    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    quiet.hush_loggers()
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = True
    conv = DocumentConverter(format_options={"pdf": PdfFormatOption(pipeline_options=opts)})

    with geom.normalized_pdf(pdf) as (norm, _info):
        doc = conv.convert(str(norm)).document
        out = []
        for ti in doc.tables:
            if not ti.prov:
                continue
            prov = ti.prov[0]
            ph = float(doc.pages[prov.page_no].size.height)
            try:
                rows = engine_docling._df_to_rows(ti.export_to_dataframe(doc))
            except Exception:
                rows = []
            cells = []
            for c in (ti.data.table_cells or []):
                bb = None
                if c.bbox:
                    r = engine_docling._to_topleft(c.bbox, ph)
                    bb = (r.x0, r.y0, r.x1, r.y1)
                cells.append({
                    "text": c.text or "",
                    "r0": c.start_row_offset_idx, "r1": c.end_row_offset_idx,
                    "c0": c.start_col_offset_idx, "c1": c.end_col_offset_idx,
                    "rspan": c.row_span, "bbox": bb,
                })
            out.append({
                "page": prov.page_no,
                "rect": engine_docling._to_topleft(prov.bbox, ph).as_tuple(),
                "rows": rows,
                "cells": cells,
            })
    ck.write_bytes(pickle.dumps(out))
    return out


# ————————————————————————— D-cell：cell 高度离群 —————————————————————————


def detect_tall_cells(cells: list[dict]) -> list[tuple[int, int, str, float, float]]:
    """按列比 cell 高度，返回 (行, 列, 文本, 高, 该列中位高)。

    为什么按列比而不是全表比：不同列的字号/折行情况不同，全表比会被长文本列拖偏。
    """
    by_col: dict[int, list[tuple[int, dict, float]]] = {}
    for c in cells:
        if not c["bbox"] or c["rspan"] != 1:
            continue
        h = abs(c["bbox"][3] - c["bbox"][1])
        if h <= 0:
            continue
        by_col.setdefault(c["c0"], []).append((c["r0"], c, h))

    out = []
    for col, items in sorted(by_col.items()):
        if len(items) < 6:
            continue
        med = statistics.median(h for _, _, h in items)
        if med <= 0:
            continue
        for r0, c, h in items:
            if h > CELL_TALL * med:
                out.append((r0, col, c["text"], h, med))
    return out


# ————————————————————————— D-band：坐标行带 —————————————————————————


_WORDS: dict[tuple[str, int], list] = {}


def words_of(pdf: Path, page: int) -> list:
    """取归一化后某页的词，带缓存 —— `read_doc` 要扫全篇，每张表重跑一次太贵。"""
    key = (str(pdf), page)
    if key not in _WORDS:
        with geom.normalized_pdf(pdf) as (norm, _info):
            for pg in {t["page"] for t in docling_dump(pdf)}:
                _WORDS[(str(pdf), pg)] = geom.page_words(norm, pg)
    return _WORDS.get(key, [])


def analyse_table(pdf: Path, t: dict, method: str) -> dict | None:
    """对一张 docling 表做坐标行带分析。返回 None 表示这张表判据不适用。"""
    rect = Rect(*t["rect"])
    words = words_of(pdf, t["page"])
    # **先按 bbox 过滤再判方向。** 拿整页判会被页上的其它内容带偏 —— 实测续表页
    # `pbc_28772` p4（76%）与 `pbc_21078` p5（71%）都因此被误判成「方向混杂」跳过，
    # 而表体本身方向是一致的。
    in_rect = geom.words_in_rect(words, rect)
    axis, share = geom.row_axis_of(in_rect)
    if axis is None:
        return {"skip": f"框内方向混杂(占比{share:.0%})"}

    inside = geom.keep_axis_dir(in_rect, axis)
    if len(inside) < 10:
        return {"skip": f"框内词太少({len(inside)})"}

    bands = (geom.bands_by_center(inside, axis, GAP_M1) if method == "M1"
             else geom.bands_by_overlap(inside, axis, GAP_M2))
    stats = geom.gap_stats(inside, axis, bands)

    # 表头区：**用第 1 行（第一个数据行）cell 的顶边划界**，它上面的带全排除。
    #
    # 两个坑，都是实测踩出来的：
    # ① **不能只跳过第一个带** —— 多行表头会切成 2-3 个带（`pbc_28772` p3 的
    #    `KM estimate of median time to event` 折两行），只跳一个就把剩下的表头行
    #    当成数据行，`pbc_28772`/`blood` 那几张「带多 1~2」全是这么来的。
    # ② **不能用第 0 行 cell 的底边划界** —— 多级表头时 docling 把各层用 `.` 拼平
    #    （PDF 事实 #16），row 0 的 bbox 只盖住最下面那层，上面那层漏成一个假数据带。
    #    实测 `566455` p61 的 `Reported Ethnicity.African` 就是这么多报一行的。
    # ③ **要用中位数而不是最小值。** 纵跨多行的标签格 bbox 顶边会远高于本行，
    #    而 docling 有时把这种格**逐行重复**（`row_span` 仍是 1，按 span 排不掉）。
    #    实测 `566455` p61 的 `Inferred Ethnicity` 纵排标签，bbox 顶边 778.28 比
    #    第二行表头（782.52）还高 —— 用 min 就会把那行表头算成数据行。
    #    第 1 行绝大多数格子都起于同一条线，中位数天然把这种离群格排除。
    tops = [c["bbox"][1] for c in t["cells"] if c["r0"] == 1 and c["bbox"]]
    row1_top = statistics.median(tops) if tops else None
    head_bot = row1_top if row1_top is not None else max(
        (c["bbox"][3] for c in t["cells"] if c["r0"] == 0 and c["bbox"]), default=rect.y0)
    n_head = sum(1 for b in bands
                 if statistics.median((w.y0 + w.y1) / 2 for w in b) <= head_bot)
    bands = bands[n_head:] if n_head < len(bands) else bands[1:]

    body = [w for b in bands for w in b]
    cols = geom.column_spans(body, axis, MIN_COL_GAP) if body else []

    occ = [geom.occupancy(b, axis, cols) for b in bands]
    used = [sum(1 for n in row if n) for row in occ]
    med_used = statistics.median([u for u in used if u]) if any(used) else 0

    return {
        "axis": axis, "n_words": len(inside), "n_bands": len(bands),
        "n_cols": len(cols), "used": used, "med_used": med_used,
        "within_max": stats["within_max"], "between_min": stats["between_min"],
        "bands": bands,
    }


def full_band_counts(res: dict, num_cols: set[int]) -> dict[str, int]:
    """三条「满带」规则各数出多少个数据行（不含表头带）。"""
    used, med = res["used"][1:], res["med_used"]
    occ_cols = res.get("occ_cols", [])
    r1 = sum(1 for u in used if u >= 3)
    r2 = sum(1 for u in used if med and u >= 0.5 * med)
    r3 = sum(1 for i, u in enumerate(used)
             if u >= 2 and (not num_cols or (occ_cols and num_cols & occ_cols[i])))
    return {"R1": r1, "R2": r2, "R3": r3 if num_cols else r1}


# ————————————————————————————— 主流程 —————————————————————————————


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-dir", type=Path, default=ROOT / "0.pdf_input")
    ap.add_argument("--only", help="只跑文件名含该串的 PDF")
    ap.add_argument("--dump-cells", help="dump 某张表的 table_cells，格式 stem:page")
    ap.add_argument("--method", choices=("M1", "M2", "both"), default="both")
    args = ap.parse_args()

    if args.dump_cells:
        stem, _, pg = args.dump_cells.partition(":")
        pdf = next(p for p in args.pdf_dir.glob("*.pdf") if stem in p.stem)
        for t in docling_dump(pdf):
            if t["page"] != int(pg):
                continue
            print(f"{pdf.stem} p{pg}  docling 行数={len(t['rows']) - 1}  cells={len(t['cells'])}")
            for r, c, txt, h, med in detect_tall_cells(t["cells"]):
                print(f"  ⚠ 行{r:3d} 列{c:2d}  高={h:6.2f} (列中位 {med:.2f}, {h / med:.2f}x)  {txt[:40]!r}")
        return 0

    pdfs = sorted(p for p in args.pdf_dir.glob("*.pdf")
                  if not args.only or args.only in p.stem)

    print("=" * 108)
    print("① D-cell：table_cells 的 bbox 高度离群（便宜，只看 docling 自己的数据）")
    print("=" * 108)
    print("判据：排除表头行(row 0)后，同一数据行有 >=2 列的 cell 高于该列中位数 %.1fx" % CELL_TALL)
    print("（表头行必须排除 —— 实测 pbc_21296 p4 的表头合法折行，6 个列全部超标 2.07-2.24x）")
    print()
    print("%-24s %4s %8s %6s  %s" % ("PDF", "页", "docling行", "命中", "明细"))
    print("-" * 108)
    tall_hits = []
    for pdf in pdfs:
        try:
            dump = docling_dump(pdf)
        except Exception as e:
            print("%-24s  docling 失败: %s" % (pdf.stem[:24], str(e)[:50]))
            continue
        for t in dump:
            by_row: dict[int, list] = {}
            for r, c, txt, h, med in detect_tall_cells(t["cells"]):
                if r == 0:
                    continue  # 表头合法折行
                by_row.setdefault(r, []).append((c, txt, h / med))
            sus = {r: v for r, v in by_row.items() if len(v) >= 2}
            nrow = len(t["rows"]) - 1 if t["rows"] else 0
            if sus:
                tall_hits.append((pdf.stem, t["page"], sus))
                d = "; ".join(
                    f"行{r}({len(v)}列,{max(x for _, _, x in v):.2f}x) {v[0][1][:14]!r}"
                    for r, v in sorted(sus.items())[:2]
                )
            else:
                d = "-"
            print("%-24s %4d %8d %6d  %s" % (pdf.stem[:24], t["page"], nrow, len(sus), d[:56]))
    print("-" * 108)
    print(f"共 {len(tall_hits)} 张表命中")

    print()
    print("=" * 108)
    print("② D-band：坐标行带数 vs docling 行数")
    print("=" * 108)
    print("原始带数含折行续段与表内脚注，必须用「满带」过滤：")
    print("  R1 = 占用列数 >= 3 ；R2 = 占用列数 >= 0.5 x 全带中位数")
    print()
    print("%-24s %4s %7s %6s %7s %7s %8s %8s  %s"
          % ("PDF", "页", "docling", "列数", "M1满带", "M2满带", "带内max", "带间min", "M1 判定"))
    print("-" * 108)
    tally = {"M1": [0, 0, 0], "M2": [0, 0, 0]}  # [一致, 带多, 带少]
    margins: list[tuple[float, float]] = []
    for pdf in pdfs:
        try:
            dump = docling_dump(pdf)
        except Exception:
            continue
        for t in dump:
            if not t["rows"]:
                continue
            nrow = len(t["rows"]) - 1
            got = {}
            for m in ("M1", "M2"):
                res = analyse_table(pdf, t, m)
                if res is None or "skip" in res:
                    got[m] = None
                    continue
                used, med = res["used"], res["med_used"]  # 表头带已剔除
                got[m] = (sum(1 for u in used if med and u >= 0.5 * med), res)
            if got["M1"] is None:
                print("%-24s %4d  %s" % (pdf.stem[:24], t["page"],
                                         (analyse_table(pdf, t, "M1") or {}).get("skip", "?")))
                continue
            for m in ("M1", "M2"):
                if got[m] is None:
                    continue
                n = got[m][0]
                tally[m][0 if n == nrow else (1 if n > nrow else 2)] += 1
            n1, res = got["M1"]
            if res["between_min"] < float("inf"):
                margins.append((res["within_max"], res["between_min"]))
            n2 = got["M2"][0] if got["M2"] else -1
            verdict = "✓" if n1 == nrow else (f"带多 {n1 - nrow} 疑压行" if n1 > nrow
                                              else f"带少 {nrow - n1}")
            print("%-24s %4d %7d %6d %7d %7d %8.2f %8.2f  %s"
                  % (pdf.stem[:24], t["page"], nrow, res["n_cols"], n1, n2,
                     res["within_max"], res["between_min"], verdict))
    print("-" * 108)
    for m in ("M1", "M2"):
        a, more, less = tally[m]
        print(f"  {m}: 一致 {a} 张 / 带多(疑压行) {more} 张 / 带少 {less} 张")
    print("\n**带多方向才是 F-001 那类错（docling 丢了行）。带少是反方向，多半是聚类过度合并。**")

    if margins:
        wi = max(w for w, _ in margins)
        be = min(b for _, b in margins)
        print()
        print("③ 阈值余量普查（M1 的 GAP_M1）")
        print("-" * 108)
        print(f"  {len(margins)} 个页-表：带内最大 gap = {wi:.2f}pt，带间最小 gap = {be:.2f}pt")
        print(f"  余量：{wi:.2f} <- 阈值 {GAP_M1} -> {be:.2f}"
              f"   （下侧 {GAP_M1 / wi:.1f}x，上侧 {be / GAP_M1:.1f}x）")
        if wi >= be:
            print("  ⚠ 两侧交叠 —— 固定阈值不成立，须改自适应")
        tight = sorted(margins, key=lambda m: m[1])[:3]
        print(f"  带间最小的三张（最接近翻转）：{[round(b, 2) for _, b in tight]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
