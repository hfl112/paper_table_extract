"""逐行 OCR 能不能把留出集那 108 条「结构还原失败」救回来？

这是 `log.md` §35 留下的那个待测项，也是本轮**唯一能改变优先级**的实验。

背景：留出集 24 篇的图片路径，199 条候选里 108 条是
「检出网格但 `TableRecognitionPipelineV2` 结构还原返回空」，
LLM 分层抽样判读说其中 **27% 是真表**（约 29 张）。
§34 提的 CLAHE / 逐行 OCR 本来是为了提高**识别质量**，
但逐行 OCR 有一个副作用值得单独验：**它绕开结构还原**，
用单元格检测的行带/列带自己搭网格 —— 所以它可能在还原失败的地方照样出表。

若成立，逐行 OCR 的价值就不是"把 83% 提到 98%"，而是"把 0 变成有"，
优先级完全不同。

测的对象是 LLM **确认是真表**的 9 条：
  4 条来自 FAIL（结构还原失败）+ 5 条来自 DEGEN（输出了 1 行 0 列）
判据用 LLM 给的首列（`/tmp/pte_img_holdout_gold.json`），**只比首列、不比逐格**
—— LLM 读图自己也是 OCR，逐格数值不能当真值（见 findings F-021 的效力边界）。

用法：
    conda run -n gemini python spike/probe_img_rescue.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdf_table_extract import engine_docling, engine_paddle, pdfio  # noqa: E402

GOLD = Path("/tmp/pte_img_holdout_gold.json")
PDF_DIR = Path("/Users/funanhe/Downloads/Zotero_Precancer_PDFs")
WORK = Path("/tmp/pte_img_rescue")
WORK.mkdir(exist_ok=True)


def norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def pdf_for(paper: str) -> Path | None:
    from pdf_table_extract.emit import sanitize_name

    for p in PDF_DIR.glob("*.pdf"):
        if p.stem == paper or sanitize_name(p.stem) == paper:
            return p
    return None


def row_by_row(img: Path, eng, det) -> tuple[list[list[str]], int, int]:
    """绕开 TableRecognitionPipelineV2：用单元格检测的行带/列带自己搭网格。"""
    import numpy as np
    from PIL import Image

    boxes = [b["coordinate"] for r in det.predict(str(img)) for b in r["boxes"]]
    if not boxes:
        return [], 0, 0
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
        crop = crop.resize((crop.width * 2, crop.height * 2), 1)
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
    return grid, len(boxes), len(cols_c)


def main() -> int:
    from rapidocr import RapidOCR

    gold = json.loads(GOLD.read_text())
    targets = [g for g in gold if g["is_table"] and g["kind"] in ("FAIL", "DEGEN")]
    print(f"LLM 确认是真表、且当前抽不出来的：{len(targets)} 条\n")

    eng = RapidOCR()
    det = engine_paddle._get_cell_det()
    by_pdf: dict[Path, list[dict]] = {}
    for g in targets:
        p = pdf_for(g["paper"])
        if p:
            by_pdf.setdefault(p, []).append(g)

    tot = rescued = 0
    for pdf, gs in by_pdf.items():
        pages = {int(g["page"]) for g in gs}
        info = pdfio.read_doc(pdf)
        norm_pdf = WORK / f"{pdf.stem[:40]}_norm.pdf"
        pdfio.write_normalized(pdf, norm_pdf, info)
        res = engine_docling.convert(norm_pdf)
        info2 = pdfio.read_doc(norm_pdf)
        for g in gs:
            pg = int(g["page"])
            pics = [p for p in res.pictures if p.page == pg]
            page = next((x for x in info2.pages if x.page == pg), None)
            print(f"══ {g['paper'][:38]} p{pg}  ({g['kind']}, 工具 {g['shape']})")
            print(f"   LLM: {g['n_rows']}x{g['n_cols']}  首列 {g['first_col'][:6]}")
            if not pics:
                print("   该页无 PictureItem，跳过\n")
                continue
            best = None
            for k, pic in enumerate(pics, 1):
                png = WORK / f"{pdf.stem[:20]}_p{pg}_{k}.png"
                hit_img = None
                if page:
                    for im in page.images:
                        if (im.rect.x0 <= pic.rect.x1 and im.rect.x1 >= pic.rect.x0
                                and im.rect.y0 <= pic.rect.y1 and im.rect.y1 >= pic.rect.y0):
                            hit_img = im
                            break
                try:
                    if hit_img is not None:
                        pdfio.extract_native_image(norm_pdf, hit_img, png, region=pic.rect)
                    else:
                        pdfio.render_region(norm_pdf, pg, pic.rect, png, dpi=300)
                except Exception as exc:
                    print(f"   pic{k}: 抠图失败 {type(exc).__name__}")
                    continue
                grid, ncell, ncol = row_by_row(png, eng, det)
                if not grid:
                    continue
                flat = {norm(c) for r in grid for c in r if c.strip()}
                hit = sum(1 for v in g["first_col"] if norm(v) and norm(v) in flat)
                filled = sum(1 for r in grid for c in r if c.strip())
                cand = (hit, len(grid), ncol, ncell, filled, grid)
                if best is None or hit > best[0]:
                    best = cand
            tot += 1
            if best is None:
                print("   逐行 OCR：无产出\n")
                continue
            hit, nr, nc, ncell, filled, grid = best
            ok = hit >= max(1, len(g["first_col"]) // 2) and nc >= 2
            rescued += ok
            print(f"   逐行 OCR：{nr}x{nc}（检出 {ncell} 格，非空 {filled}）"
                  f"  首列命中 {hit}/{len(g['first_col'])}  {'✅ 救回' if ok else '❌ 没救回'}")
            for r in grid[:4]:
                print(f"       {[c[:18] for c in r[:6]]}")
            print()
    print("=" * 78)
    print(f"逐行 OCR 救回 {rescued}/{tot} 条（判据：首列命中 ≥ 一半 且 列数 ≥2）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
