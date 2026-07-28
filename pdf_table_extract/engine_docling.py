"""docling 引擎。**docling 的唯一入口** —— 它的 API 不许漏到本文件外。

docling 是主干：实测 10/10 篇找到正文表，pbc_28772 抽出 38x11 + 6x11 = 44 行，
与前身项目的人工金标准一致。两个确定性表格发现器（pdfplumber lattice / PyMuPDF
find_tables）在同一批料上失效（见 AGENTS.md PDF 事实 #5），故不用它们做发现。

必须喂**归一化后**的 PDF（横向表页已转正）。实测整篇归一化跑一次的结果与逐页转正
逐字一致，所以不必逐页调用 —— 那样 19 页的文章要跑 19 次，慢十几倍。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from . import quiet, rules
from .models import Rect


@dataclass
class DoclingTable:
    page: int  # 1-based
    rect: Rect  # top-left 原点，与 pdfio.Rect 一致
    rows: list[list[str]]  # 含表头行
    caption: str  # docling 自己关联的 caption，**经常为空**（PDF 事实 #13）


@dataclass
class DoclingPicture:
    page: int
    rect: Rect
    caption: str


@dataclass
class DoclingResult:
    tables: list[DoclingTable]
    pictures: list[DoclingPicture]


def _to_topleft(bbox, page_height: float) -> Rect:
    """docling 的 BoundingBox 可能用 BOTTOMLEFT 原点，统一转成 top-left。"""
    origin = str(getattr(bbox, "coord_origin", "")).upper()
    if origin.endswith("BOTTOMLEFT"):
        top, bottom = page_height - bbox.t, page_height - bbox.b
    else:
        top, bottom = bbox.t, bbox.b
    top, bottom = min(top, bottom), max(top, bottom)
    return Rect(float(bbox.l), float(top), float(bbox.r), float(bottom))


def _df_to_rows(df) -> list[list[str]]:
    """DataFrame → 二维字符串表，第 0 行是列名（表头）。

    顺带清掉 docling 漏出来的**字形名**（`/emspaceMean` → `Mean`）——
    见 rules.strip_glyph_names。这是 docling 的产出问题，所以在翻译它的输出时就地清掉。
    """
    header = [("" if c is None else str(c)) for c in df.columns]
    body = [[("" if v is None else str(v)) for v in row] for row in df.itertuples(index=False)]
    return rules.strip_glyph_names([header, *body])


def convert(pdf_path: str | Path) -> DoclingResult:
    """跑一次 docling，把 TableItem / PictureItem 翻译成纯数据。

    **关掉 docling 自己的 OCR（do_ocr=False）**，三个理由：
      1. 架构上它就不该做 OCR —— 分流判据已经把"区域内无文字层"的像素内容
         送去 paddle 那条路了（含单元格闸门 + 裁剪 + 英文识别模型），
         docling 在本项目里只负责文字层内容。
      2. 扫描件也不受影响：那些 PDF 的文字层是**出版社 OCR 生成的、确实存在**，
         docling 读文字层坐标即可（实测能从中恢复出正确行列结构）。
      3. docling 默认拉 RapidOCR，每次运行往 stderr 刷 9 行 INFO；关掉即静音，且更快。
    """
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    quiet.hush_loggers()

    opts = PdfPipelineOptions()
    opts.do_ocr = False
    opts.do_table_structure = True
    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=opts)}
    )
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
        tables.append(
            DoclingTable(
                page=prov.page_no,
                rect=_to_topleft(prov.bbox, ph),
                rows=rows,
                caption=re.sub(r"\s+", " ", ti.caption_text(doc) or "").strip(),
            )
        )

    pictures: list[DoclingPicture] = []
    for pi in doc.pictures:
        if not pi.prov:
            continue
        prov = pi.prov[0]
        ph = float(doc.pages[prov.page_no].size.height)
        pictures.append(
            DoclingPicture(
                page=prov.page_no,
                rect=_to_topleft(prov.bbox, ph),
                caption=re.sub(r"\s+", " ", pi.caption_text(doc) or "").strip(),
            )
        )

    tables.sort(key=lambda t: (t.page, t.rect.y0))
    pictures.sort(key=lambda p: (p.page, p.rect.y0))
    return DoclingResult(tables=tables, pictures=pictures)
