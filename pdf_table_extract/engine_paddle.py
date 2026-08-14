"""paddleocr engine. The ONLY entry point to paddleocr (and PIL); their APIs never leak past this file.

Figure-path primitives only: cell detection (the OCR gate and the grid geometry
source), per-cell OCR, plain-text OCR for coarse keyword screening, swatch color
sampling. Structure reconstruction (the old ppstructure
TableRecognitionPipelineV2) is gone: measured unusable on dual-panel journal
figures (17/101 intact row names); the dual-read merge replaced it.
"""

from __future__ import annotations

from pathlib import Path

from . import common
from .common import Box, Rect

common.silence()

# The English recognition model must be explicit: the zh/en default reads -3 as a CJK glyph.
EN_REC_MODEL = "en_PP-OCRv5_mobile_rec"
CELL_DET_MODEL = "RT-DETR-L_wired_table_cell_det"

_cell_det = None
_plain_ocr = None


def _get_cell_det():
    global _cell_det
    if _cell_det is None:
        from paddleocr import TableCellsDetection

        common.hush_loggers()  # paddlex resets its logger to INFO at import
        # enable_mkldnn=False: paddle 3.3 oneDNN+PIR crashes on these CPUs
        _cell_det = TableCellsDetection(model_name=CELL_DET_MODEL, enable_mkldnn=False)
    return _cell_det


def _get_plain_ocr():
    global _plain_ocr
    if _plain_ocr is None:
        from paddleocr import PaddleOCR

        common.hush_loggers()
        _plain_ocr = PaddleOCR(
            lang="en",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )
    return _plain_ocr


# ---------------------------------------------------------------------------
# cell detection (gate + geometry source)
# ---------------------------------------------------------------------------


def detect_cells(image_path: str | Path, threshold: float = 0.3) -> list[tuple[Box, float]]:
    """Detected cell boxes with scores. Tables give ~300 boxes, pure charts 0-6 (the gate)."""
    out: list[tuple[Box, float]] = []
    for r in _get_cell_det().predict(str(image_path), threshold=threshold):
        for b in r["boxes"]:
            c = b["coordinate"]
            out.append(((float(c[0]), float(c[1]), float(c[2]), float(c[3])), float(b["score"])))
    return out


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


def plain_text(image_path: str | Path) -> str:
    """Text-only OCR (no structure), for coarse in-image keyword screening."""
    out: list[str] = []
    for r in _get_plain_ocr().predict(str(image_path)):
        out.extend(r.get("rec_texts", []) or [])
    return " ".join(out)


def open_image(image_path: str | Path):
    """Opaque image handle for the per-cell functions below (PIL stays inside this file)."""
    from PIL import Image

    return Image.open(str(image_path)).convert("RGB")


def ocr_cell(image, box: Box, scratch: str | Path, *, pad_l: int, pad_r: int, pad_y: int, upscale: bool) -> str:
    """OCR one detected cell box cropped out of the page image.

    Vertical padding must stay tiny (cells sit one row pitch apart); 2x LANCZOS
    upscaling is what recovers thin minus signs and decimal dots (measured).
    """
    from PIL import Image

    x1, y1, x2, y2 = box
    crop = image.crop((max(0, x1 - pad_l), max(0, y1 - pad_y), x2 + pad_r, y2 + pad_y))
    if upscale:
        crop = crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)
    crop.save(str(scratch))
    res = list(_get_plain_ocr().predict(str(scratch)))[0]
    return " ".join(res["rec_texts"]) if res["rec_texts"] else ""


# ---------------------------------------------------------------------------
# swatch color sampling (heat-map cells carry data as color, not text)
# ---------------------------------------------------------------------------

SWATCH_ANCHORS = {
    "dark green": (0, 100, 0),
    "green": (60, 170, 70),
    "light green": (150, 220, 150),
    "gray": (128, 128, 128),
    "yellow": (235, 215, 60),
    "orange": (245, 160, 40),
    "red": (215, 45, 45),
    "blue": (60, 90, 215),
}


def swatch_color(image, box: Box) -> str | None:
    """Nearest anchor color word when the cell is a solid swatch, else None.

    A swatch is uniform AND chromatic (channel spread >= 25) or clearly darker than
    paper; white cells with sparse text are achromatic and near-white (measured:
    looser thresholds misread text cells as 'light green').
    """
    x1, y1, x2, y2 = box
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    im = image.crop((x1 + 2, y1 + 2, x2 - 2, y2 - 2)).resize((16, 16))
    px = list(im.getdata())
    means = [sum(c[i] for c in px) / len(px) for i in range(3)]
    stds = [(sum((c[i] - means[i]) ** 2 for c in px) / len(px)) ** 0.5 for i in range(3)]
    chroma = max(means) - min(means)
    if max(stds) > 30 or (chroma < 25 and min(means) > 170):
        return None
    return min(SWATCH_ANCHORS, key=lambda k: sum((SWATCH_ANCHORS[k][i] - means[i]) ** 2 for i in range(3)))
