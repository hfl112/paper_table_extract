"""把逐行 OCR 的产物**直接拿去打 gold 分** —— 这一轮该做而没做的那个测试。

前面几个探针量的都是**代理指标**：参照命中（首列与同篇文字表对得上几个）、
合法码率（响应码在不在码表里）。它们便宜、能定位错因，但**回答不了那个真问题**：
「这个改动对目标(b)（PDX × 处理 × 响应 三元组）到底有没有用」。

只有拿 `eval/gold.py --score` 去打才回答得了 —— 那是人核过的 gold，
按锚点对齐行，逐格比对。当前基线很难看：
    pbc_24724__p05_fig_1   计分 0 格 / 漏行 41 / 多出行 29   ← 完全对不齐
    pbc_21296__p03_fig_1   计分 5 格 / 漏行 39
对不齐的原因就是 F-018（首列词首字符被吃掉），而锚点正是瘤系名 ——
锚点全错 ⇒ 一行都对不上 ⇒ 打分器给不出分。
逐行 OCR 在 24724 上把首列参照命中从 0 拉到 32/41，**所以它应当让这张表重新可打分**。
这个探针就是去验这一句。

做法：把逐行 OCR 的结果写成与产品同名的 CSV，放进 `1.output` 的**副本**，
再对副本打分 —— 不动 `1.output` 本身，方便与基线对照。

用法：
    conda run -n gemini python spike/probe_rowocr_goldscore.py
"""

from __future__ import annotations

import csv
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdf_table_extract import engine_paddle, pdfio  # noqa: E402
from pdf_table_extract.models import Rect  # noqa: E402

SCRATCH = Path("/tmp/pte_rowocr_out")
SCRATCH_C = Path("/tmp/pte_rowocr_clahe_out")
WORK = Path("/tmp/pte_rowocr_work")
WORK.mkdir(exist_ok=True)

TARGETS = [
    {"pdf": "0.pdf_input/pbc_21296.pdf", "page": 3, "region": Rect(126, 75, 298, 302),
     "out": "pbc_21296/p03_fig_1.csv"},
    {"pdf": "0.pdf_input/pbc_24724.pdf", "page": 5, "region": None,
     "out": "pbc_24724/p05_fig_1.csv"},
]


def row_by_row(img: Path, eng, det) -> list[list[str]]:
    import numpy as np
    from PIL import Image

    boxes = [b["coordinate"] for r in det.predict(str(img)) for b in r["boxes"]]
    if not boxes:
        return []
    hs = [b[3] - b[1] for b in boxes]
    ws = [b[2] - b[0] for b in boxes]
    rows_c = [sum(bb) / len(bb) for bb in engine_paddle._bands(
        [(b[1] + b[3]) / 2 for b in boxes], statistics.median(hs) * 0.5)]
    cols_c = [sum(bb) / len(bb) for bb in engine_paddle._bands(
        [(b[0] + b[2]) / 2 for b in boxes], statistics.median(ws) * 0.5)]
    x0, x1 = min(b[0] for b in boxes), max(b[2] for b in boxes)
    rowh = statistics.median(hs)
    im = Image.open(img).convert("RGB")
    grid = []
    for cy in rows_c:
        crop = im.crop((max(0, x0 - 4), max(0, cy - rowh * 0.6),
                        min(im.width, x1 + 4), min(im.height, cy + rowh * 0.6)))
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
        row = ["" for _ in cols_c]
        try:
            res = eng(np.array(crop))
        except Exception:
            res = None
        if res and res.txts:
            for txt, bx in zip(res.txts, res.boxes):
                cx = x0 - 4 + ((min(q[0] for q in bx) + max(q[0] for q in bx)) / 2) / 2
                ci = min(range(len(cols_c)), key=lambda i: abs(cols_c[i] - cx))
                row[ci] = (row[ci] + " " + txt).strip() if row[ci] else txt
        grid.append(row)
    return grid


def clahe_png(src: Path, dst: Path) -> Path:
    """§34 量到 CLAHE 在 120dpi 上把首列参照命中从 5 拉到 18。
    但那是**代理指标**（标签出现在网格里任何位置），gold 分要求标签在**正确的行**上。
    两者会不会打架，只有真打一次分才知道 —— 这个变体就是为此加的。"""
    import cv2

    im = cv2.imread(str(src))
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    cv2.imwrite(str(dst), cv2.cvtColor(lab, cv2.COLOR_LAB2BGR))
    return dst


def main() -> int:
    from rapidocr import RapidOCR

    if not (ROOT / "1.output").exists():
        print("1.output 不存在，先跑 bash eval/matrix.sh")
        return 1
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    shutil.copytree(ROOT / "1.output", SCRATCH)
    if SCRATCH_C.exists():
        shutil.rmtree(SCRATCH_C)
    shutil.copytree(ROOT / "1.output", SCRATCH_C)
    print(f"已把 1.output 复制到 {SCRATCH} 与 {SCRATCH_C}（基线不动）\n")

    eng = RapidOCR()
    det = engine_paddle._get_cell_det()
    for t in TARGETS:
        pdf = ROOT / t["pdf"]
        info = pdfio.read_doc(pdf)
        page = next(p for p in info.pages if p.page == t["page"])
        img = max(page.images, key=lambda i: i.px_width * i.px_height)
        raw = WORK / f"{pdf.stem}_raw.png"
        pdfio.extract_native_image(pdf, img, raw, region=t["region"])
        factor = engine_paddle.upscale_factor(img.effective_dpi)
        cur = raw
        if factor > 1:
            up = WORK / f"{pdf.stem}_x{factor}.png"
            engine_paddle.upscale(raw, up, factor)
            cur = up
        for tag, root, src in (("逐行", SCRATCH, cur),
                               ("逐行+CLAHE", SCRATCH_C,
                                clahe_png(cur, WORK / f"{pdf.stem}_clahe.png"))):
            grid = row_by_row(src, eng, det)
            dst = root / t["out"]
            if not grid:
                print(f"  [{tag}] {t['out']}: 无产出，保留基线")
                continue
            width = max(len(r) for r in grid)
            with dst.open("w", newline="") as f:
                csv.writer(f).writerows([r + [""] * (width - len(r)) for r in grid])
            print(f"  [{tag}] {t['out']}: 写入 {len(grid)}x{width}"
                  f"（原生 {img.effective_dpi:.0f}dpi，放大 {factor}x）")

    print("\n" + "=" * 96)
    print("【基线】1.output —— 产品当前的整表管线")
    print("=" * 96)
    subprocess.run([sys.executable, "eval/gold.py", "--score", "--outdir", "1.output",
                    "--source", "human,agreed"], cwd=ROOT)
    print("\n" + "=" * 96)
    print("【逐行 OCR】两张图片表替换后")
    print("=" * 96)
    subprocess.run([sys.executable, "eval/gold.py", "--score", "--outdir", str(SCRATCH),
                    "--source", "human,agreed"], cwd=ROOT)
    print("\n【逐行 OCR + CLAHE】")
    subprocess.run([sys.executable, "eval/gold.py", "--score", "--outdir", str(SCRATCH_C),
                    "--source", "human,agreed"], cwd=ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
