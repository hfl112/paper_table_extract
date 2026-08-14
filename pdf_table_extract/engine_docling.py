"""docling engine. The ONLY entry point to docling; its API never leaks past this file.

docling is the backbone of the text path: measured 10/10 papers found their body
tables where the deterministic finders (pdfplumber lattice, PyMuPDF find_tables)
failed. Feed it the NORMALIZED pdf (sideways pages rotated): one whole-document
run measured identical to per-page rotation and >10x faster.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import common, evaluation
from .common import Rect, TableCellBox

common.silence()


@dataclass
class DoclingTable:
    page: int  # 1-based
    rect: Rect
    rows: list[list[str]]  # header row included, from export_to_dataframe
    caption: str  # docling's own caption link, often empty
    # table_cells geometry: a SECOND view of the same table whose row numbering
    # disagrees with rows for 184/427 measured tables; merged-row detection only
    cells: list[TableCellBox] = field(default_factory=list)


@dataclass
class DoclingResult:
    tables: list[DoclingTable]


def _to_topleft(bbox, page_height: float) -> Rect:
    """docling BoundingBox may use a bottom-left origin; unify to top-left."""
    origin = str(getattr(bbox, "coord_origin", "")).upper()
    if origin.endswith("BOTTOMLEFT"):
        top, bottom = page_height - bbox.t, page_height - bbox.b
    else:
        top, bottom = bbox.t, bbox.b
    top, bottom = min(top, bottom), max(top, bottom)
    return Rect(float(bbox.l), float(top), float(bbox.r), float(bottom))


def _df_to_rows(df) -> list[list[str]]:
    """DataFrame -> 2-d strings, row 0 = header; strips leaked glyph names (/emspaceMean -> Mean)."""
    header = [("" if c is None else str(c)) for c in df.columns]
    body = [[("" if v is None else str(v)) for v in row] for row in df.itertuples(index=False)]
    return evaluation.strip_glyph_names([header, *body])


def convert(pdf_path: str | Path) -> DoclingResult:
    """Run docling once; translate TableItem/PictureItem into plain data.

    docling's own OCR stays OFF: the dispatch criterion already routes pixel
    regions to the figure path, publisher-OCR text layers are readable as-is,
    and RapidOCR would only add startup noise.
    """
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    common.hush_loggers()

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = True
    converter = DocumentConverter(format_options={"pdf": PdfFormatOption(pipeline_options=opts)})
    doc = converter.convert(str(pdf_path)).document

    tables: list[DoclingTable] = []
    for ti in doc.tables:
        if not ti.prov:
            continue
        prov = ti.prov[0]
        ph = float(doc.pages[prov.page_no].size.height)
        try:
            rows = _df_to_rows(ti.export_to_dataframe(doc))
        except Exception:
            rows = []
        cells = [
            TableCellBox(
                text=c.text or "",
                r0=c.start_row_offset_idx,
                c0=c.start_col_offset_idx,
                row_span=c.row_span,
                col_span=c.col_span,
                # docling's own header verdict: measured headers span 1-3 rows, never guess
                column_header=bool(getattr(c, "column_header", False)),
                bbox=_to_topleft(c.bbox, ph) if c.bbox else None,
            )
            for c in (ti.data.table_cells or [])
        ]
        tables.append(
            DoclingTable(
                page=prov.page_no,
                rect=_to_topleft(prov.bbox, ph),
                rows=rows,
                caption=re.sub(r"\s+", " ", ti.caption_text(doc) or "").strip(),
                cells=cells,
            )
        )

    tables.sort(key=lambda t: (t.page, t.rect.y0))
    return DoclingResult(tables=tables)
