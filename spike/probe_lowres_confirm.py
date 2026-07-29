"""确认 `probe_lowres_prep.py` 里那两个"看起来有效"的预处理是不是真的。

**为什么必须单独确认**：prep 探针的插值对照暴露了一件事 —— 裁剪框差几个像素
（闸门 280 vs 282 格），参照命中就从 5 掉到 0。也就是说**这条管线本身的抖动
和我要比较的差异同一个量级**。所以 CLAHE 的 5→18 必须换裁剪边距重测，
在两个边距下都成立才算真效果，否则就是拟合了某一次裁剪。

同时把首列原样打出来 —— 24724（300dpi）参照命中 0/41 反常，得看清是
"读错字"还是"根本没读到那一列"。

用法：
    conda run -n gemini python spike/probe_lowres_confirm.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdf_table_extract import engine_paddle, pdfio  # noqa: E402
from pdf_table_extract.models import Rect  # noqa: E402

WORK = Path("/tmp/pte_lowres_confirm")
WORK.mkdir(exist_ok=True)
CODES = {"PD1", "PD2", "SD", "PR", "CR", "MCR"}

TARGETS = [
    {"pdf": "0.pdf_input/pbc_21296.pdf", "page": 3, "region": Rect(126, 75, 298, 302),
     "ref_csv": "1.output/pbc_21296/p04_table_i.csv", "tag": "21296_120dpi"},
    {"pdf": "0.pdf_input/pbc_24724.pdf", "page": 5, "region": None,
     "ref_csv": "1.output/pbc_24724/p04_table_ii.csv", "tag": "24724_300dpi"},
]


def clahe(im):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def denoise(im):
    return cv2.fastNlMeansDenoisingColored(im, None, 5, 5, 7, 21)


VARIANTS = [
    ("baseline", lambda im: im),
    ("CLAHE", clahe),
    ("denoise", denoise),
    ("CLAHE+denoise", lambda im: denoise(clahe(im))),
]
PADS = [0.02, 0.04]  # 0.02 是产品当前的 CROP_PAD_RATIO


def ref_labels(p: Path) -> list[str]:
    with p.open(newline="") as f:
        rows = list(csv.reader(f))
    return [r[0].strip() for r in rows[1:] if r and r[0].strip()]


def recognize(pipe, path):
    best = []
    for r in pipe.predict(str(path)):
        for tb in r.get("table_res_list", []) or []:
            rr = engine_paddle._html_to_rows(tb.get("pred_html", "") or "")
            if len(rr) > len(best):
                best = rr
    return best


def code_stats(rows):
    if not rows:
        return -1, 0, 0
    ncol = max(len(r) for r in rows)
    col, n, tot = -1, 0, 0
    for ci in range(ncol):
        vals = [r[ci].strip() for r in rows[1:] if len(r) > ci and r[ci].strip()]
        k = sum(1 for v in vals if v.upper() in CODES)
        if k > n:
            col, n, tot = ci, k, len(vals)
    return col, n, tot


def phase2() -> None:
    """检测框**放大**扫描。

    第一阶段的首列原样暴露了真正的错法：`BT-29`→`8T-29`/`Y-29`、`EW5`→`W5`、
    `TC-71`→`-71`、`CHLA258`→`A258`、`SK-NEP-1`→`NEP-1` —— **词首字符被吃掉**。
    这不是"认错字"，是**检测框左边界切进了文字里**。

    所以方向和外部建议正好相反：那份建议调低 `unclip_ratio` 是为了治"两行粘连"，
    而我们看到的是"框太小"，该**调高**。同理 `box_thresh` 调低会保留更大的框。
    CLAHE 之所以能把参照命中从 5 拉到 18，八成也是同一个机制（局部对比度上去了，
    检测框才敢往左伸），这一阶段就是要把这个假设和参数直接对上。
    """
    pipe = engine_paddle._get_table_pipe()
    grid_kw = [
        ("默认", {}),
        ("unclip=1.5", {"text_det_unclip_ratio": 1.5}),
        ("unclip=2.0", {"text_det_unclip_ratio": 2.0}),
        ("unclip=2.5", {"text_det_unclip_ratio": 2.5}),
        ("unclip=3.0", {"text_det_unclip_ratio": 3.0}),
        ("box_thresh=0.4", {"text_det_box_thresh": 0.4}),
        ("box_thresh=0.3", {"text_det_box_thresh": 0.3}),
        ("unclip=2.5+box=0.4", {"text_det_unclip_ratio": 2.5, "text_det_box_thresh": 0.4}),
    ]
    for t in TARGETS:
        pdf = ROOT / t["pdf"]
        info = pdfio.read_doc(pdf)
        page = next(p for p in info.pages if p.page == t["page"])
        img = max(page.images, key=lambda i: i.px_width * i.px_height)
        raw = WORK / f"{t['tag']}_raw.png"
        pdfio.extract_native_image(pdf, img, raw, region=t["region"])
        factor = engine_paddle.upscale_factor(img.effective_dpi)
        cur = raw
        if factor > 1:
            up = WORK / f"{t['tag']}_x{factor}.png"
            engine_paddle.upscale(raw, up, factor)
            cur = up
        grid = engine_paddle.cells_summary(cur)
        crop = WORK / f"{t['tag']}_p2crop.png"
        if grid.bbox is not None:
            engine_paddle.crop(cur, grid.bbox, crop)
        else:
            crop.write_bytes(cur.read_bytes())
        base = cv2.imread(str(crop))
        refs = ref_labels(ROOT / t["ref_csv"])
        print(f"\n{'=' * 80}\n{t['tag']}  参照 {len(refs)} 个标签  （两种前处理 × 8 组检测参数）")
        for pname, pfn in (("raw", lambda im: im), ("CLAHE", clahe)):
            out = WORK / f"{t['tag']}_p2_{pname}.png"
            cv2.imwrite(str(out), pfn(base.copy()))
            print(f"  --- 前处理 {pname} ---")
            for name, kw in grid_kw:
                best = []
                for r in pipe.predict(str(out), **kw):
                    for tb in r.get("table_res_list", []) or []:
                        rr = engine_paddle._html_to_rows(tb.get("pred_html", "") or "")
                        if len(rr) > len(best):
                            best = rr
                flat = {c.strip() for r_ in best for c in r_}
                hit = sum(1 for r_ in refs if r_ in flat)
                col, n, tot = code_stats(best)
                shape = f"{len(best)}x{max((len(r_) for r_ in best), default=0)}"
                print(f"    {name:<20} {shape:>8} 参照 {hit:>3}/{len(refs):<4} "
                      f"码 {n:>3}/{tot:<4}={(n / tot * 100 if tot else 0):3.0f}%"
                      f"   首列 {[r_[0].strip() for r_ in best[1:]][:5]}")


def phase3() -> None:
    """**降采样对照实验** —— 给"要不要装超分模型"一个有上界的答案。

    问题：120dpi 那张表读不出来，是**信息没了**，还是**管线不行**？
    这两者对策完全相反 —— 前者只有超分（而超分必然是在编像素），后者是调管线（免费）。
    光看 120dpi 那张图分不开，因为没有对照。

    做法（TextZoom 那套配对数据的思路，但用我们自己的图，所以有真值）：
    拿 300dpi 那张**信息充足**的图，人工降到 120dpi 附近，再走一遍标准管线。
      - 若降采样后的成绩 ≈ 真 120dpi 那张的成绩 ⇒ 差距**就是分辨率**，
        且"完美超分"能拿回的上限 = 300dpi 原图的成绩 − 降采样后的成绩
      - 若降采样后成绩仍然好 ⇒ 真 120dpi 那张的问题不在分辨率，超分白装

    这是本轮唯一能给超分定量上界的实验，而且零安装。
    """
    pipe = engine_paddle._get_table_pipe()
    t = TARGETS[1]  # 300dpi 那张
    pdf = ROOT / t["pdf"]
    info = pdfio.read_doc(pdf)
    page = next(p for p in info.pages if p.page == t["page"])
    img = max(page.images, key=lambda i: i.px_width * i.px_height)
    raw = WORK / "p3_raw.png"
    pdfio.extract_native_image(pdf, img, raw, region=t["region"])
    refs = ref_labels(ROOT / t["ref_csv"])
    a = cv2.imread(str(raw))
    print(f"\n{'=' * 80}\n降采样对照：{t['tag']} 原生 {a.shape[1]}x{a.shape[0]} "
          f"@ {img.effective_dpi:.0f}dpi，参照 {len(refs)} 标签")

    for label, div in (("300dpi 原图(真值上界)", 1.0), ("→200dpi", 1.5), ("→120dpi", 2.5),
                       ("→100dpi", 3.0)):
        if div == 1.0:
            small = a
        else:
            small = cv2.resize(a, (int(a.shape[1] / div), int(a.shape[0] / div)),
                               interpolation=cv2.INTER_AREA)
        sp = WORK / f"p3_div{div}.png"
        cv2.imwrite(str(sp), small)
        eff = img.effective_dpi / div
        factor = engine_paddle.upscale_factor(eff)
        cur = sp
        if factor > 1:
            up = WORK / f"p3_div{div}_x{factor}.png"
            engine_paddle.upscale(sp, up, factor)
            cur = up
        grid = engine_paddle.cells_summary(cur)
        crop = WORK / f"p3_div{div}_crop.png"
        if grid.bbox is not None:
            engine_paddle.crop(cur, grid.bbox, crop)
        else:
            crop.write_bytes(cur.read_bytes())
        base = cv2.imread(str(crop))
        px_row = small.shape[0] / (grid.n_row_bands or 1)
        print(f"\n  {label}: {small.shape[1]}x{small.shape[0]}  有效 {eff:.0f}dpi  "
              f"放大 {factor}x  闸门 {grid.n_cells} 格/{grid.n_row_bands} 行带  "
              f"**原生每行 {px_row:.1f} px**")
        for pname, pfn in (("raw", lambda im: im), ("CLAHE", clahe)):
            out = WORK / f"p3_div{div}_{pname}.png"
            cv2.imwrite(str(out), pfn(base.copy()))
            best = []
            for r in pipe.predict(str(out)):
                for tb in r.get("table_res_list", []) or []:
                    rr = engine_paddle._html_to_rows(tb.get("pred_html", "") or "")
                    if len(rr) > len(best):
                        best = rr
            flat = {c.strip() for r_ in best for c in r_}
            hit = sum(1 for r_ in refs if r_ in flat)
            col, n, tot = code_stats(best)
            shape = f"{len(best)}x{max((len(r_) for r_ in best), default=0)}"
            print(f"    {pname:<6} {shape:>8} 参照 {hit:>3}/{len(refs):<4} "
                  f"码 {n:>3}/{tot:<4}={(n / tot * 100 if tot else 0):3.0f}%   "
                  f"首列 {[r_[0].strip() for r_ in best[1:]][:5]}")


def main() -> None:
    pipe = engine_paddle._get_table_pipe()
    for t in TARGETS:
        pdf = ROOT / t["pdf"]
        info = pdfio.read_doc(pdf)
        page = next(p for p in info.pages if p.page == t["page"])
        img = max(page.images, key=lambda i: i.px_width * i.px_height)
        raw = WORK / f"{t['tag']}_raw.png"
        pdfio.extract_native_image(pdf, img, raw, region=t["region"])
        factor = engine_paddle.upscale_factor(img.effective_dpi)
        cur = raw
        if factor > 1:
            up = WORK / f"{t['tag']}_x{factor}.png"
            engine_paddle.upscale(raw, up, factor)
            cur = up
        grid = engine_paddle.cells_summary(cur)
        refs = ref_labels(ROOT / t["ref_csv"])
        print(f"\n{'=' * 80}\n{t['tag']}  闸门 {grid.n_cells} 格 / {grid.n_row_bands} 行带 / "
              f"参照 {len(refs)} 个标签")
        print(f"  {'变体':<16} {'pad':>5} {'形状':>8} {'参照命中':>9} {'合法码 n/总':>14}")
        for pad in PADS:
            engine_paddle.CROP_PAD_RATIO = pad  # 临时改，只在本进程内
            crop = WORK / f"{t['tag']}_pad{int(pad * 100)}.png"
            if grid.bbox is not None:
                engine_paddle.crop(cur, grid.bbox, crop)
            else:
                crop.write_bytes(cur.read_bytes())
            base = cv2.imread(str(crop))
            for name, fn in VARIANTS:
                out = WORK / f"{t['tag']}_pad{int(pad * 100)}_{name.replace('+', '_')}.png"
                cv2.imwrite(str(out), fn(base.copy()))
                rows = recognize(pipe, out)
                flat = {c.strip() for r in rows for c in r}
                hit = sum(1 for r in refs if r in flat)
                col, n, tot = code_stats(rows)
                shape = f"{len(rows)}x{max((len(r) for r in rows), default=0)}"
                print(f"  {name:<16} {pad:5.2f} {shape:>8} {hit:>4}/{len(refs):<4} "
                      f"{n:>5}/{tot:<4} = {(n / tot * 100 if tot else 0):3.0f}%")
                if name == "baseline" and pad == PADS[0] and rows:
                    first = [r[0].strip() for r in rows[1:]][:14]
                    print(f"       ↳ 首列原样: {first}")
                    print(f"       ↳ 参照首列: {refs[:14]}")




def phase4() -> None:
    """**逐行 OCR × CLAHE** 的 2×2 —— 本轮矩阵里唯一还空着的一格。

    为什么值得单独跑：前三阶段各证了一半。
      - phase1/2：CLAHE 在 120dpi 上把首列参照命中从 5 拉到 18（换裁剪边距不变），
        但整表管线的**列几何**该错还错
      - phase3：300dpi 原图首列 0/41，比它自己降到 100dpi（4/41）还差
        ⇒ 24724 的首列错**根本不是分辨率问题**，是整表管线把列切错了
      - 早先量过逐行 OCR：24724 从 83% 到 98%（列错位一并修好），
        21296 反而从 82% 掉到 41%

    逐行 OCR 绕开整表列几何，CLAHE 补局部对比度 —— 两者治的是不同的病，
    所以要交叉验一遍它们能不能叠加。识别器用 RapidOCR（**和整表管线不同的引擎**），
    顺带就是图片路径当前唯一缺的第二意见。
    """
    import statistics

    import numpy as np
    from PIL import Image
    from rapidocr import RapidOCR

    eng = RapidOCR()
    det = engine_paddle._get_cell_det()
    for t in TARGETS:
        pdf = ROOT / t["pdf"]
        info = pdfio.read_doc(pdf)
        page = next(p for p in info.pages if p.page == t["page"])
        img = max(page.images, key=lambda i: i.px_width * i.px_height)
        raw = WORK / f"p4_{t['tag']}_raw.png"
        pdfio.extract_native_image(pdf, img, raw, region=t["region"])
        factor = engine_paddle.upscale_factor(img.effective_dpi)
        cur = raw
        if factor > 1:
            up = WORK / f"p4_{t['tag']}_x{factor}.png"
            engine_paddle.upscale(raw, up, factor)
            cur = up
        refs = ref_labels(ROOT / t["ref_csv"])
        print(f"\n{'=' * 80}\n{t['tag']}  逐行 OCR（RapidOCR）  参照 {len(refs)} 标签")
        for pname, pfn in (("raw", lambda im: im), ("CLAHE", clahe)):
            src = WORK / f"p4_{t['tag']}_{pname}.png"
            cv2.imwrite(str(src), pfn(cv2.imread(str(cur))))
            boxes = []
            for r in det.predict(str(src)):
                for b in r["boxes"]:
                    boxes.append(b["coordinate"])
            if not boxes:
                print(f"  {pname}: 单元格检测 0 个")
                continue
            hs = [b[3] - b[1] for b in boxes]
            ws = [b[2] - b[0] for b in boxes]
            rows_c = [sum(bb) / len(bb) for bb in engine_paddle._bands(
                [(b[1] + b[3]) / 2 for b in boxes], statistics.median(hs) * 0.5)]
            cols_c = [sum(bb) / len(bb) for bb in engine_paddle._bands(
                [(b[0] + b[2]) / 2 for b in boxes], statistics.median(ws) * 0.5)]
            x0, x1 = min(b[0] for b in boxes), max(b[2] for b in boxes)
            rowh = statistics.median(hs)
            im = Image.open(src).convert("RGB")
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
                        cx = x0 - 4 + ((min(p[0] for p in bx) + max(p[0] for p in bx)) / 2) / 2
                        ci = min(range(len(cols_c)), key=lambda i: abs(cols_c[i] - cx))
                        row[ci] = (row[ci] + " " + txt).strip() if row[ci] else txt
                grid.append(row)
            flat = {c.strip() for r_ in grid for c in r_}
            hit = sum(1 for r_ in refs if r_ in flat)
            col, n, tot = code_stats(grid)
            print(f"  {pname:<6} {len(grid)}x{len(cols_c)}  参照 {hit:>3}/{len(refs):<4} "
                  f"码 {n:>3}/{tot:<4}={(n / tot * 100 if tot else 0):3.0f}%")
            print(f"       首列 {[r_[0].strip() for r_ in grid[1:]][:8]}")
            if col >= 0:
                print(f"       码列#{col} {[r_[col].strip() for r_ in grid[1:] if len(r_) > col][:10]}")


def phase5() -> None:
    """逐行 OCR 的**分辨率下界**在哪 —— 用 phase3 那套配对降采样去量，不靠猜。

    phase4 给了两端：27.8 px/行 时逐行 OCR 大胜（参照 0→32、码 69%→98%），
    8.0 px/行 时它把响应列一半格子读成空（44 → 20~29 个非空）。
    中间没有样本，所以"什么时候该走逐行"这个阈值悬着。

    这里拿**同一张 300dpi 图**降到 200/120/100dpi，逐行 OCR 各跑一遍。
    同一张图 ⇒ 版式、字体、列数全都一样，唯一变量是分辨率，
    所以拐点是分辨率本身的拐点，不是某篇论文的特性。
    """
    import statistics

    import numpy as np
    from PIL import Image
    from rapidocr import RapidOCR

    eng = RapidOCR()
    det = engine_paddle._get_cell_det()
    t = TARGETS[1]
    pdf = ROOT / t["pdf"]
    info = pdfio.read_doc(pdf)
    page = next(p for p in info.pages if p.page == t["page"])
    img = max(page.images, key=lambda i: i.px_width * i.px_height)
    raw = WORK / "p5_raw.png"
    pdfio.extract_native_image(pdf, img, raw, region=t["region"])
    refs = ref_labels(ROOT / t["ref_csv"])
    a = cv2.imread(str(raw))
    print(f"\n{'=' * 80}\n逐行 OCR 的分辨率下界（同一张图降采样，参照 {len(refs)} 标签）")
    for label, div in (("300dpi", 1.0), ("200dpi", 1.5), ("150dpi", 2.0),
                       ("120dpi", 2.5), ("100dpi", 3.0)):
        small = a if div == 1.0 else cv2.resize(
            a, (int(a.shape[1] / div), int(a.shape[0] / div)), interpolation=cv2.INTER_AREA)
        sp = WORK / f"p5_{label}.png"
        cv2.imwrite(str(sp), small)
        factor = engine_paddle.upscale_factor(img.effective_dpi / div)
        cur = sp
        if factor > 1:
            up = WORK / f"p5_{label}_x{factor}.png"
            engine_paddle.upscale(sp, up, factor)
            cur = up
        boxes = [b["coordinate"] for r in det.predict(str(cur)) for b in r["boxes"]]
        if not boxes:
            print(f"  {label}: 单元格检测 0 个")
            continue
        hs = [b[3] - b[1] for b in boxes]
        ws = [b[2] - b[0] for b in boxes]
        rows_c = [sum(bb) / len(bb) for bb in engine_paddle._bands(
            [(b[1] + b[3]) / 2 for b in boxes], statistics.median(hs) * 0.5)]
        cols_c = [sum(bb) / len(bb) for bb in engine_paddle._bands(
            [(b[0] + b[2]) / 2 for b in boxes], statistics.median(ws) * 0.5)]
        x0, x1 = min(b[0] for b in boxes), max(b[2] for b in boxes)
        rowh = statistics.median(hs)
        im = Image.open(cur).convert("RGB")
        grid, empty = [], 0
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
            else:
                empty += 1
            grid.append(row)
        flat = {c.strip() for r_ in grid for c in r_}
        hit = sum(1 for r_ in refs if r_ in flat)
        col, n, tot = code_stats(grid)
        px_row = small.shape[0] / (len(rows_c) or 1)
        filled = sum(1 for r_ in grid for c in r_ if c.strip())
        print(f"  {label:<7} 原生每行 {px_row:5.1f} px  {len(grid)}x{len(cols_c)}  "
              f"参照 {hit:>3}/{len(refs):<4} 码 {n:>3}/{tot:<4}={(n / tot * 100 if tot else 0):3.0f}%  "
              f"整行读空 {empty:>2}/{len(rows_c):<3} 非空格子 {filled}")


if __name__ == "__main__":
    if "--phase2" in sys.argv:
        phase2()
    elif "--phase3" in sys.argv:
        phase3()
    elif "--phase4" in sys.argv:
        phase4()
    elif "--phase5" in sys.argv:
        phase5()
    else:
        main()
