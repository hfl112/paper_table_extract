"""paddleocr 引擎。**paddleocr 的唯一入口** —— 它的 API 不许漏到本文件外。

只在 OCR 路径上用（区域内文字层字符 ≈0，内容是像素）。三件事：

1. `cells_bbox()` —— OCR 闸门。用 TableCellsDetection 直接检测单元格，**绕过版面分类**。
   必须绕过的原因（AGENTS.md PDF 事实 #9）：期刊常把"表格 + 图"拼成一张图，
   PPStructureV3 会把整图判成 `chart`(0.79) 并跳过表格识别，
   TableRecognitionPipelineV2 直接喂全图同样返回 0 个表。
   实测成本 0.4-0.7 秒/图（vs 全套 OCR 10.6-28.4 秒），判别力 300 vs 0-6 个单元格。
   这条闸门替代了原设计里的领域词表，符合泛用性原则。

2. `plain_text()` —— 粗筛用的纯文字 OCR（只识字不还原结构）。

3. `recognize_table()` —— 结构还原。**必须指定英文识别模型**（PDF 事实 #10）：
   `lang='en'` 参数不被接受（ValueError: Unknown argument: lang），
   不指定则中英混合模型会把 `-3` 识别成中文 `心`、`Histology` 读成 `Hiatology`。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import quiet
from .models import Rect

# PDF 事实 #10：必须显式指定英文识别模型。
EN_REC_MODEL = "en_PP-OCRv5_mobile_rec"
CELL_DET_MODEL = "RT-DETR-L_wired_table_cell_det"

_cell_det = None
_table_pipe = None
_plain_ocr = None


def _get_cell_det():
    global _cell_det
    if _cell_det is None:
        from paddleocr import TableCellsDetection

        quiet.hush_loggers()  # paddlex/RapidOCR 在 import 时把级别设回 INFO

        _cell_det = TableCellsDetection(model_name=CELL_DET_MODEL)
    return _cell_det


def _get_table_pipe():
    global _table_pipe
    if _table_pipe is None:
        from paddleocr import TableRecognitionPipelineV2

        quiet.hush_loggers()  # paddlex/RapidOCR 在 import 时把级别设回 INFO

        _table_pipe = TableRecognitionPipelineV2(text_recognition_model_name=EN_REC_MODEL)
    return _table_pipe


def _get_plain_ocr():
    global _plain_ocr
    if _plain_ocr is None:
        from paddleocr import PaddleOCR

        quiet.hush_loggers()  # paddlex/RapidOCR 在 import 时把级别设回 INFO

        _plain_ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _plain_ocr


@dataclass
class CellGrid:
    """单元格检测的结果。既当 OCR 闸门，也当图片路径唯一的第二意见。"""

    n_cells: int
    bbox: Rect | None
    n_col_bands: int  # 由格子 x 中心聚类估出的列数
    n_row_bands: int  # 由格子 y 中心聚类估出的行数


def _bands(values: list[float], tol: float) -> list[list[float]]:
    """把一维坐标聚成带。tol 是同一带内允许的最大间距。"""
    if not values:
        return []
    vs = sorted(values)
    out: list[list[float]] = [[vs[0]]]
    for v in vs[1:]:
        if v - out[-1][-1] > tol:
            out.append([v])
        else:
            out[-1].append(v)
    return out


def _cluster_1d(values: list[float], tol: float) -> int:
    return len(_bands(values, tol))


# 稀疏带丢弃阈值：单元格数低于"中位带计数"这个比例的带，判为伪检测。
#
# 必须做的原因（实测，与 PDF 事实 #9 同一个坑）：检测器会在**条形图区域也检出伪单元格**
# （柱子和网格线形成方框），把全部单元格的外接框撑到整图 → 裁剪等于没裁 →
# 版面模型又把整图判成 chart → 返回 0 个表。
# 实测每 5% 宽度的单元格数分布：
#   pbc_21296 Fig.1  [47,0,0,48,0,47,47,0,47,47 | 0,0,0,1,0,0,0,1,0,0]
#                     ←──── 表格区 ────→         ←─ 图区只有 1 个 ─→   → 裁到 50% 宽才能成功
#   pbc_24724 Fig.1  [46,0,46,0,47,47,46,47,21 | 0×10]  → 天然只占 45%，本来就成功
SPARSE_BAND_RATIO = 0.25
SPARSE_BAND_MIN = 2


def _dense_span(values: list[float], tol: float) -> tuple[float, float] | None:
    """只保留密集带，返回它们覆盖的坐标范围。全是稀疏带时返回 None。"""
    bands = _bands(values, tol)
    if not bands:
        return None
    counts = sorted(len(b) for b in bands)
    median = counts[len(counts) // 2]
    floor = max(SPARSE_BAND_MIN, median * SPARSE_BAND_RATIO)
    kept = [b for b in bands if len(b) >= floor]
    if not kept:
        return None
    return min(b[0] for b in kept), max(b[-1] for b in kept)


def cells_summary(image_path: str | Path) -> CellGrid:
    """OCR 闸门 + 图片路径的第二意见。一次检测同时给出三个用途的数据。

    外接框用于裁剪 —— 实测 pbc_24724 Fig.1 是"左半表格 + 右半条形图"，
    检测到 300 个单元格、外接框占宽 48%，必须裁剪后才能正确还原。

    列带/行带数用于对账 —— 图片路径没有文字层，coverage 和 pdfplumber 都用不上，
    而这一步**已经免费数过每个格子的坐标**，聚类即得期望的行列数。
    实测教训：还原结果曾是 46x6，而正确形状是 45x7；只比总格子数分不开
    （276 vs 315 相对 300 分别差 8% 与 5%），比列数就一目了然。
    """
    res = _get_cell_det().predict(str(image_path))
    xs: list[float] = []
    ys: list[float] = []
    widths: list[float] = []
    heights: list[float] = []
    x0 = y0 = float("inf")
    x1 = y1 = float("-inf")
    for r in res:
        for b in r["boxes"]:
            c = b["coordinate"]
            xs.append((c[0] + c[2]) / 2)
            ys.append((c[1] + c[3]) / 2)
            widths.append(c[2] - c[0])
            heights.append(c[3] - c[1])
            x0, y0 = min(x0, c[0]), min(y0, c[1])
            x1, y1 = max(x1, c[2]), max(y1, c[3])
    n = len(xs)
    if n == 0:
        return CellGrid(0, None, 0, 0)
    med_w = sorted(widths)[len(widths) // 2]
    med_h = sorted(heights)[len(heights) // 2]

    # **裁剪框只用密集带算** —— 全部单元格的外接框会被图表区的伪检测撑到整图，
    # 裁剪等于没裁，版面模型又把整图判成 chart（见 SPARSE_BAND_RATIO 处的实测数据）。
    span_x = _dense_span(xs, med_w * 0.5)
    span_y = _dense_span(ys, med_h * 0.5)
    if span_x and span_y:
        bbox = Rect(
            max(float(x0), span_x[0] - med_w),
            max(float(y0), span_y[0] - med_h),
            min(float(x1), span_x[1] + med_w),
            min(float(y1), span_y[1] + med_h),
        )
    else:
        bbox = Rect(float(x0), float(y0), float(x1), float(y1))

    return CellGrid(
        n_cells=n,
        bbox=bbox,
        n_col_bands=_cluster_1d(xs, med_w * 0.5),
        n_row_bands=_cluster_1d(ys, med_h * 0.5),
    )


def plain_text(image_path: str | Path) -> str:
    """纯文字 OCR，只识字不还原结构。用于"关键词是否出现在图内"的粗筛。"""
    out: list[str] = []
    for r in _get_plain_ocr().predict(str(image_path)):
        out.extend(r.get("rec_texts", []) or [])
    return " ".join(out)


def recognize_table(image_path: str | Path) -> list[list[str]]:
    """把（已裁剪到表格区域的）图片还原成二维表。取行数最多的那张表。"""
    best: list[list[str]] = []
    for r in _get_table_pipe().predict(str(image_path)):
        for t in r.get("table_res_list", []) or []:
            rows = _html_to_rows(t.get("pred_html", "") or "")
            if len(rows) > len(best):
                best = rows
    return best


_TR = re.compile(r"<tr[^>]*>(.*?)</tr>", re.S | re.I)
_TD = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")


def _html_to_rows(html: str) -> list[list[str]]:
    rows = []
    for tr in _TR.findall(html):
        cells = [_TAG.sub("", c).strip() for c in _TD.findall(tr)]
        if cells:
            rows.append(cells)
    return rows


# 裁剪边距按比例给，不用固定像素。
# 实测教训：固定 8px 时 pbc_24724 Fig.1 被切掉了最右一列（Heat Map 色块列，
# 那列没有文字所以单元格检测器给的外接框到 x=911 就停了，而该列实际延伸到约 x=940），
# 结果还原出 46x6 而不是正确的 45x7。
# 但也不能放太宽：这张图右半是条形图，从约 x=960 开始；把它包进来会让版面模型
# 重新把整图判成 chart、返回 0 个表（PDF 事实 #9）。2% 在两者之间有余量。
CROP_PAD_RATIO = 0.02
CROP_PAD_MIN = 8


def crop(image_path: str | Path, rect: Rect, dst: str | Path) -> None:
    """按 rect 裁剪图片，边距按图幅比例。PIL 也只在本文件出现。"""
    from PIL import Image

    im = Image.open(str(image_path))
    w, h = im.size
    pad_x = max(CROP_PAD_MIN, int(w * CROP_PAD_RATIO))
    pad_y = max(CROP_PAD_MIN, int(h * CROP_PAD_RATIO))
    box = (
        max(0, int(rect.x0) - pad_x),
        max(0, int(rect.y0) - pad_y),
        min(w, int(rect.x1) + pad_x),
        min(h, int(rect.y1) + pad_y),
    )
    im.crop(box).save(str(dst))


# 低分辨率图在还原前放大。目标 ~300 有效 dpi，最多 4 倍。
#
# 实测依据（pbc_21296 Fig.1，560x374、有效 120dpi）：
#   只裁剪不放大 → 44x5，表头全空、`Rhabdoid` 读成 `Ahabdad`、`-2.0` 读成 `-210`
#   裁剪 + 4x    → 45x6，表头 `inte|Histology|Score|Difference|Group Respons|Map`、首行 `-2.0` 正确
# 注意**缺一不可**：单靠放大无效（未裁剪时 1x/2x/3x/4x 全部还原失败，因为整图被判成 chart）；
# 单靠裁剪能出结果但字全错。放大不增加信息，但让 OCR 模型达到它期望的输入尺度。
UPSCALE_TARGET_DPI = 300
UPSCALE_MAX = 4


def upscale_factor(effective_dpi: float) -> int:
    """按有效 dpi 算放大倍数。300dpi 及以上返回 1（不放大）。"""
    if effective_dpi <= 0:
        return 1
    import math

    return max(1, min(UPSCALE_MAX, math.ceil(UPSCALE_TARGET_DPI / effective_dpi)))


def upscale(image_path: str | Path, dst: str | Path, factor: int) -> None:
    from PIL import Image

    im = Image.open(str(image_path))
    w, h = im.size
    im.resize((w * factor, h * factor), Image.LANCZOS).save(str(dst))
