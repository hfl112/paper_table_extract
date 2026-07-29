"""图片表低分辨率 OCR：检测器参数扫描。

**背景**：外部建议里排第一的是"先超分辨率"。但我们早先量到
`TableRecognitionPipelineV2` 内部的文字检测把长边压到 **960**
（`text_det_limit_side_len=960 / limit_type=max`），当时判断"再放大也白搭"。

那个判断少了一步 —— **960 是可配置的**，`predict()` 直接收
`text_det_limit_side_len` / `text_det_limit_type` / `text_det_thresh` /
`text_det_box_thresh` / `text_det_unclip_ratio`。
所以在装任何超分模型之前，先把这几个旋钮扫一遍：**零安装、零幻觉风险**。

另外扫 `text_det_unclip_ratio` —— 外部建议指出"多行粘连"的根因常在**检测**
（DBNet 的框向外膨胀把两行黏成一个框），调低膨胀系数应当能拆开。这正是
我们 F-005 那类"列错位/行合并"的候选解释。

**打分不用 gold**（gold 在库外且不许回显）。三个客观指标：
1. 形状（行×列）
2. 首列标签与**同篇文字表**首列的精确匹配数 —— 文字表来自文字层，可信
3. 响应码列的合法率（码表 {PD1,PD2,SD,PR,CR,MCR} 是原文用的，不是我们编的）

用法：
    conda run -n gemini python spike/probe_lowres_ocr.py
"""

from __future__ import annotations

import csv
import itertools
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdf_table_extract import engine_paddle, pdfio  # noqa: E402
from pdf_table_extract.models import Rect  # noqa: E402

WORK = Path("/tmp/pte_lowres_sweep")
WORK.mkdir(exist_ok=True)

CODES = {"PD1", "PD2", "SD", "PR", "CR", "MCR"}

# 两个目标：一个 120dpi（我们的最差档）、一个 300dpi（最好档）。
# pbc_21296 p3 的位图被 docling 拆成两个 PictureItem，表格在左半
# `[126,75,298,302]`（AGENTS.md PDF 事实 #9c）—— 必须带 region，否则拿到整图。
TARGETS = [
    {
        "pdf": "0.pdf_input/pbc_21296.pdf",
        "page": 3,
        "region": Rect(126, 75, 298, 302),
        "ref_csv": "1.output/pbc_21296/p04_table_i.csv",
        "tag": "21296_120dpi",
    },
    {
        "pdf": "0.pdf_input/pbc_24724.pdf",
        "page": 5,
        "region": None,
        "ref_csv": "1.output/pbc_24724/p04_table_ii.csv",
        "tag": "24724_300dpi",
    },
]

# 扫描网格。baseline 用 None 表示"不传参、走 pipeline 默认"（即现在产品的行为）。
SWEEP = [
    ("baseline(默认960/max)", {}),
    ("side_len=1536", {"text_det_limit_side_len": 1536}),
    ("side_len=2048", {"text_det_limit_side_len": 2048}),
    ("limit_type=min/960", {"text_det_limit_side_len": 960, "text_det_limit_type": "min"}),
    ("unclip=1.2", {"text_det_unclip_ratio": 1.2}),
    ("unclip=1.0", {"text_det_unclip_ratio": 1.0}),
    ("side_len=1536+unclip=1.2", {"text_det_limit_side_len": 1536, "text_det_unclip_ratio": 1.2}),
    ("side_len=1536+thresh=0.4", {"text_det_limit_side_len": 1536, "text_det_thresh": 0.4}),
]


def ref_labels(csv_path: Path) -> list[str]:
    """同篇文字表的首列（去表头、去空）。"""
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        rows = list(csv.reader(f))
    return [r[0].strip() for r in rows[1:] if r and r[0].strip()]


def prepare(t: dict) -> Path | None:
    """复刻产品图片路径的取图步骤：抠原生图 → 放大 → 闸门 → 裁剪。"""
    pdf = ROOT / t["pdf"]
    info = pdfio.read_doc(pdf)
    page = next((p for p in info.pages if p.page == t["page"]), None)
    if page is None or not page.images:
        print(f"  [{t['tag']}] 该页无嵌入位图，跳过")
        return None
    img = max(page.images, key=lambda i: i.px_width * i.px_height)
    raw = WORK / f"{t['tag']}_raw.png"
    w, h = pdfio.extract_native_image(pdf, img, raw, region=t["region"])
    eff = img.effective_dpi
    factor = engine_paddle.upscale_factor(eff)
    print(f"  [{t['tag']}] 原生 {img.px_width}x{img.px_height} 有效 {eff:.0f}dpi "
          f"→ 抠出 {w}x{h} → 放大 {factor}x")
    cur = raw
    if factor > 1:
        up = WORK / f"{t['tag']}_x{factor}.png"
        engine_paddle.upscale(raw, up, factor)
        cur = up
    grid = engine_paddle.cells_summary(cur)
    print(f"  [{t['tag']}] 闸门: {grid.n_cells} 格, 行带 {grid.n_row_bands} 列带 {grid.n_col_bands}")
    if grid.bbox is None:
        return cur
    crop = WORK / f"{t['tag']}_crop.png"
    engine_paddle.crop(cur, grid.bbox, crop)
    from PIL import Image

    cw, ch = Image.open(crop).size
    print(f"  [{t['tag']}] 裁剪后 {cw}x{ch}  (长边 {max(cw, ch)}；"
          f"默认 960 上限{'会' if max(cw, ch) > 960 else '不会'}压缩)")
    return crop


def score(rows: list[list[str]], refs: list[str]) -> dict:
    if not rows:
        return {"shape": "0x0", "ref_hit": 0, "code_col": -1, "code_rate": 0.0, "n_code": 0}
    flat = {c.strip() for r in rows for c in r}
    ref_hit = sum(1 for r in refs if r in flat)
    ncol = max(len(r) for r in rows)
    best_col, best_n, best_rate = -1, 0, 0.0
    for ci in range(ncol):
        vals = [r[ci].strip() for r in rows[1:] if len(r) > ci and r[ci].strip()]
        if not vals:
            continue
        n = sum(1 for v in vals if v.upper() in CODES)
        if n > best_n:
            best_col, best_n, best_rate = ci, n, n / len(vals)
    return {
        "shape": f"{len(rows)}x{ncol}",
        "ref_hit": ref_hit,
        "code_col": best_col,
        "code_rate": best_rate,
        "n_code": best_n,
    }


def main() -> None:
    pipe = engine_paddle._get_table_pipe()  # 复用产品的英文识别模型配置
    for t in TARGETS:
        print(f"\n{'=' * 78}\n{t['tag']}")
        img = prepare(t)
        if img is None:
            continue
        refs = ref_labels(ROOT / t["ref_csv"])
        print(f"  参照（同篇文字表首列）{len(refs)} 个标签，例: {refs[:4]}")
        print(f"\n  {'配置':<26} {'形状':>8} {'秒':>6} {'参照命中':>8} {'响应码列':>8} {'合法率':>8}")
        for name, kw in SWEEP:
            t0 = time.time()
            try:
                best: list[list[str]] = []
                for r in pipe.predict(str(img), **kw):
                    for tb in r.get("table_res_list", []) or []:
                        rr = engine_paddle._html_to_rows(tb.get("pred_html", "") or "")
                        if len(rr) > len(best):
                            best = rr
                rows = best
            except Exception as exc:  # noqa: BLE001
                print(f"  {name:<26} 报错 {type(exc).__name__}: {exc}")
                continue
            dt = time.time() - t0
            s = score(rows, refs)
            print(f"  {name:<26} {s['shape']:>8} {dt:6.1f} "
                  f"{s['ref_hit']:>4}/{len(refs):<3} "
                  f"{('#' + str(s['code_col'])) if s['code_col'] >= 0 else '—':>8} "
                  f"{s['n_code']:>3}/{s['code_rate'] * 100:.0f}%")
            if rows:
                print(f"       表头: {rows[0][:8]}")
                if s["code_col"] >= 0:
                    col = [r[s["code_col"]].strip() for r in rows[1:]
                           if len(r) > s["code_col"]][:12]
                    print(f"       码列: {col}")


if __name__ == "__main__":
    main()
