"""图片表低分辨率 OCR：**图像预处理**扫描 + 像素预算测量。

配套 `probe_lowres_ocr.py`（那个扫检测器参数，结论是没用）。本探针测两件事：

**一、外部建议里 ★★★★☆ 的那一档预处理是否有效** —— CLAHE / 自适应二值化 /
unsharp / 去噪 / 不同插值。全部只用 cv2（`opencv 5.0.0` 已随 paddleocr 装好，
**零新增依赖、零幻觉风险**），所以在考虑装 Real-ESRGAN 之前先把它们量掉。

**二、像素预算** —— 每个文字行摊到多少像素。这是判断"信息还在不在"的物理量，
比 dpi 更直接：dpi 要配合显示尺寸才有意义，而"一行 8 px"是 OCR 识别器
本身就达不到的输入尺度，此时任何后端、任何超分都只是在猜。

用法：
    conda run -n gemini python spike/probe_lowres_prep.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdf_table_extract import engine_paddle, pdfio  # noqa: E402
from pdf_table_extract.models import Rect  # noqa: E402

WORK = Path("/tmp/pte_lowres_prep")
WORK.mkdir(exist_ok=True)
CODES = {"PD1", "PD2", "SD", "PR", "CR", "MCR"}

TARGETS = [
    {"pdf": "0.pdf_input/pbc_21296.pdf", "page": 3, "region": Rect(126, 75, 298, 302),
     "ref_csv": "1.output/pbc_21296/p04_table_i.csv", "tag": "21296_120dpi"},
    {"pdf": "0.pdf_input/pbc_24724.pdf", "page": 5, "region": None,
     "ref_csv": "1.output/pbc_24724/p04_table_ii.csv", "tag": "24724_300dpi"},
]


# ---- 预处理变体。输入输出都是 BGR ndarray，便于串联。 ----
def p_none(im):
    return im


def p_unsharp(im):
    blur = cv2.GaussianBlur(im, (0, 0), 1.0)
    return cv2.addWeighted(im, 1.8, blur, -0.8, 0)


def p_clahe(im):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def p_clahe_unsharp(im):
    return p_unsharp(p_clahe(im))


def p_adaptive(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    b = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, 25, 10)
    return cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)


def p_otsu(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)


def p_denoise(im):
    return cv2.fastNlMeansDenoisingColored(im, None, 5, 5, 7, 21)


PREPS = [
    ("baseline(Lanczos放大)", p_none),
    ("+unsharp", p_unsharp),
    ("+CLAHE", p_clahe),
    ("+CLAHE+unsharp", p_clahe_unsharp),
    ("+adaptiveThreshold", p_adaptive),
    ("+Otsu二值化", p_otsu),
    ("+fastNlMeansDenoising", p_denoise),
]

# 插值方式对照（外部建议称 bicubic 帮助有限，这里验一下我们用的 Lanczos 是否真的更好）
INTERPS = [("LANCZOS4", cv2.INTER_LANCZOS4), ("CUBIC", cv2.INTER_CUBIC),
           ("NEAREST", cv2.INTER_NEAREST), ("AREA", cv2.INTER_AREA)]


def ref_labels(p: Path) -> list[str]:
    if not p.exists():
        return []
    with p.open(newline="") as f:
        rows = list(csv.reader(f))
    return [r[0].strip() for r in rows[1:] if r and r[0].strip()]


def recognize(pipe, path: Path) -> list[list[str]]:
    best: list[list[str]] = []
    for r in pipe.predict(str(path)):
        for tb in r.get("table_res_list", []) or []:
            rr = engine_paddle._html_to_rows(tb.get("pred_html", "") or "")
            if len(rr) > len(best):
                best = rr
    return best


def score(rows, refs) -> dict:
    if not rows:
        return {"shape": "0x0", "ref_hit": 0, "n_code": 0, "rate": 0.0, "col": -1}
    flat = {c.strip() for r in rows for c in r}
    ncol = max(len(r) for r in rows)
    col, n, rate = -1, 0, 0.0
    for ci in range(ncol):
        vals = [r[ci].strip() for r in rows[1:] if len(r) > ci and r[ci].strip()]
        if not vals:
            continue
        k = sum(1 for v in vals if v.upper() in CODES)
        if k > n:
            col, n, rate = ci, k, k / len(vals)
    return {"shape": f"{len(rows)}x{ncol}", "ref_hit": sum(1 for r in refs if r in flat),
            "n_code": n, "rate": rate, "col": col}


def base_crop(t: dict) -> tuple[Path, dict]:
    """复刻产品取图，并同时量出像素预算。"""
    pdf = ROOT / t["pdf"]
    info = pdfio.read_doc(pdf)
    page = next(p for p in info.pages if p.page == t["page"])
    img = max(page.images, key=lambda i: i.px_width * i.px_height)
    raw = WORK / f"{t['tag']}_raw.png"
    pdfio.extract_native_image(pdf, img, raw, region=t["region"])
    eff = img.effective_dpi
    factor = engine_paddle.upscale_factor(eff)
    cur = raw
    if factor > 1:
        up = WORK / f"{t['tag']}_x{factor}.png"
        engine_paddle.upscale(raw, up, factor)
        cur = up
    grid = engine_paddle.cells_summary(cur)
    crop = WORK / f"{t['tag']}_crop.png"
    if grid.bbox is not None:
        engine_paddle.crop(cur, grid.bbox, crop)
    else:
        crop.write_bytes(cur.read_bytes())
    a = cv2.imread(str(raw))
    c = cv2.imread(str(crop))
    # 像素预算：**按放大前的原生像素算**，放大不增加信息
    px_row_native = a.shape[0] / grid.n_row_bands if grid.n_row_bands else 0
    return crop, {
        "native": f"{a.shape[1]}x{a.shape[0]}", "eff_dpi": eff, "factor": factor,
        "crop": f"{c.shape[1]}x{c.shape[0]}", "rows": grid.n_row_bands,
        "px_row_native": px_row_native, "px_row_fed": c.shape[0] / (grid.n_row_bands or 1),
    }


def main() -> None:
    pipe = engine_paddle._get_table_pipe()
    for t in TARGETS:
        crop, m = base_crop(t)
        refs = ref_labels(ROOT / t["ref_csv"])
        print(f"\n{'=' * 84}\n{t['tag']}")
        print(f"  原生 {m['native']} @ {m['eff_dpi']:.0f}dpi  →放大 {m['factor']}x→ 裁剪 {m['crop']}")
        print(f"  ★ 像素预算：{m['rows']} 行 → **原生每行 {m['px_row_native']:.1f} px**"
              f"（喂给模型的是 {m['px_row_fed']:.1f} px，但那是插值出来的）")
        base = cv2.imread(str(crop))

        print(f"\n  预处理（{len(PREPS)} 种，全部 cv2 零新增依赖）")
        print(f"  {'变体':<26} {'形状':>8} {'秒':>6} {'参照命中':>9} {'合法码':>10}")
        for name, fn in PREPS:
            out = WORK / f"{t['tag']}_{name.replace('+', 'p').replace('(', '_').replace(')', '')}.png"
            t0 = time.time()
            cv2.imwrite(str(out), fn(base.copy()))
            rows = recognize(pipe, out)
            dt = time.time() - t0
            s = score(rows, refs)
            print(f"  {name:<26} {s['shape']:>8} {dt:6.1f} {s['ref_hit']:>4}/{len(refs):<4} "
                  f"{s['n_code']:>3}/{s['rate'] * 100:3.0f}%")
            if rows and s["col"] >= 0:
                print(f"       码列: {[r[s['col']].strip() for r in rows[1:] if len(r) > s['col']][:10]}")

        if m["factor"] > 1:
            print(f"\n  插值方式对照（放大 {m['factor']}x，同样裁剪框）")
            a = cv2.imread(str(WORK / f"{t['tag']}_raw.png"))
            for name, flag in INTERPS:
                up = cv2.resize(a, (a.shape[1] * m["factor"], a.shape[0] * m["factor"]),
                                interpolation=flag)
                up_p = WORK / f"{t['tag']}_interp_{name}.png"
                cv2.imwrite(str(up_p), up)
                g = engine_paddle.cells_summary(up_p)
                cp = WORK / f"{t['tag']}_interp_{name}_crop.png"
                if g.bbox is not None:
                    engine_paddle.crop(up_p, g.bbox, cp)
                else:
                    cp.write_bytes(up_p.read_bytes())
                rows = recognize(pipe, cp)
                s = score(rows, refs)
                print(f"  {name:<26} {s['shape']:>8} {'':>6} {s['ref_hit']:>4}/{len(refs):<4} "
                      f"{s['n_code']:>3}/{s['rate'] * 100:3.0f}%   闸门 {g.n_cells} 格")


if __name__ == "__main__":
    main()
