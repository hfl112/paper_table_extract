"""pdfplumber engine. The ONLY entry point to pdfplumber.

One job: report the column count inside a given bbox, as a second opinion.
Measured: as a table FINDER it fails on three-line (borderless) journal tables;
its row counts measure text lines, not logical rows (ratios 1.07-2.18 vs docling);
its column counts agree within 0-2, which is exactly what a second opinion needs.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from .common import Rect

warnings.filterwarnings("ignore", module="pdfplumber")

# three-line tables have no vertical rules, so columns must come from text alignment
_SETTINGS = (
    {"horizontal_strategy": "text", "vertical_strategy": "text"},
    {"horizontal_strategy": "lines", "vertical_strategy": "text"},
)


def extract_in_bbox(pdf_path: str | Path, page_no: int, rect: Rect) -> list[list[str]]:
    """Extract a 2-d table inside the bbox; [] when nothing is found."""
    import pdfplumber

    try:
        with pdfplumber.open(str(pdf_path)) as doc:
            if page_no < 1 or page_no > len(doc.pages):
                return []
            page = doc.pages[page_no - 1]
            x0 = max(0.0, rect.x0)
            y0 = max(0.0, rect.y0)
            x1 = min(float(page.width), rect.x1)
            y1 = min(float(page.height), rect.y1)
            if x1 - x0 < 1 or y1 - y0 < 1:
                return []
            region = page.crop((x0, y0, x1, y1))
            for st in _SETTINGS:
                try:
                    tbl = region.extract_table(st)
                except Exception:
                    continue
                if tbl:
                    return [[("" if c is None else str(c)) for c in row] for row in tbl]
    except Exception:
        return []
    return []


def columns_in_bbox(pdf_path: str | Path, page_no: int, rect: Rect) -> int | None:
    """Column count inside the bbox (the second opinion); None on failure."""
    rows = extract_in_bbox(pdf_path, page_no, rect)
    if not rows:
        return None
    return max(len(r) for r in rows)
