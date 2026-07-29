"""纯数据结构。零第三方依赖 —— 阶段之间只传这些，不传库对象、不传文件路径。

这样 rules.py 能被纯单测，engine_* 能被整体替换（三个引擎都可能被换，
本项目开发过程中已经换掉一个：pdfplumber 从"发现器"降级成"只提供列数的第二意见"）。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rect:
    """页面坐标矩形，top-left 原点，单位 pt。刻意不用 fitz.Rect，避免 PyMuPDF 泄漏出 pdfio.py。"""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass(frozen=True)
class Word:
    """文字层里的一个词 + 它的矩形。给「压行检测」用（见 rules.detect_merged_rows）。

    只需要坐标和文本 —— 方向不用带，因为 `pdfio.page_words` 已经把横向表页
    的坐标转到旋转后的帧里了，竖排文字在那个帧里就是正常水平文字。
    """

    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass(frozen=True)
class TableCellBox:
    """docling `TableCell` 的几何 + 索引，翻译成纯数据。

    为什么必须单独带出来（`log.md` §40）：产品写进 CSV 的网格来自
    `export_to_dataframe()`，而压行检测需要**每个格子的高度**，只有 `table_cells` 有。
    实测这两套视图在 **184/427 张表上行数不一致**（多级表头被拼平成一行），
    所以不能用行号在两者之间换算 —— 见 `rules.match_cell_row_to_output_row`。
    """

    text: str
    r0: int          # start_row_offset_idx
    c0: int          # start_col_offset_idx
    row_span: int
    col_span: int
    column_header: bool   # docling 自己的表头判定，**别自己猜表头有几行**
    bbox: Rect | None


@dataclass(frozen=True)
class ImageInfo:
    """页面上的一张嵌入位图。"""

    xref: int
    rect: Rect
    px_width: int
    px_height: int
    cover_w: float  # 占页面宽度的比例 0-1
    cover_h: float  # 占页面高度的比例 0-1

    @property
    def effective_dpi(self) -> float:
        """有效 dpi = 原生像素宽 / 显示宽度(inch)。见 AGENTS.md PDF 事实 #4。"""
        if self.rect.width <= 0:
            return 0.0
        return 72.0 * self.px_width / self.rect.width


@dataclass
class Label:
    """一个图表标号的一次出现。"""

    raw: str  # 原文，如 "TABLE I"
    key: str  # 归一化，如 "table i"
    kind: str  # "table" | "figure"
    page: int  # 1-based
    is_caption: bool  # True = 真 caption（行首判据）；False = 正文引用
    text: str  # is_caption 时是 caption+legend；否则是所在 block 全文（用于同句共现）
    rect: Rect | None = None  # caption 所在行的矩形

    @property
    def is_supplementary(self) -> bool:
        return self.key.startswith("supp ")


@dataclass
class PageInfo:
    """一页的结构特征。全部来自文字层与图形对象，不做任何 ML。"""

    page: int  # 1-based
    width: float
    height: float
    n_chars: int  # 文字层字符数（strip 后）
    vert_lines: int  # 竖排文字行数 —— 见 AGENTS.md PDF 事实 #3
    horiz_lines: int
    n_vector_ops: int  # 矢量绘图指令数（矢量表的线条）
    images: list[ImageInfo] = field(default_factory=list)
    ocr_text_layer: bool = False  # 被单张位图覆盖 ≥95% ⇒ 文字层是 OCR 派生，见事实 #8b
    needs_rotation: int = 0  # 0 / 90 / 270


@dataclass
class DocInfo:
    """整篇的特征。"""

    path: str
    n_pages: int
    pages: list[PageInfo]
    metadata: dict[str, str]
    preprint_markers: dict[str, int] = field(default_factory=dict)
    preprint_pages: int = 0  # 含水印词的页数 —— 判 preprint 看的是它占总页数的比例
    p1_watermark: str = ""   # 第 1 页左缘的竖排预印本水印原文（arXiv 型只盖首页，见 pdfio）

    @property
    def total_chars(self) -> int:
        return sum(p.n_chars for p in self.pages)


@dataclass
class TableCandidate:
    """一个待抽取的候选：知道在哪一页的哪个矩形，也知道该走哪条路。"""

    page: int
    rect: Rect
    source_type: str  # "text"（区域内有文字层） | "image"（区域内≈0字，内容是像素）
    label: Label | None = None
    chars_in_rect: int = 0
    origin: str = ""  # "docling_table" | "docling_picture" | "raster_image"
    image: ImageInfo | None = None


@dataclass
class ExtractedTable:
    """抽取结果。rows 含表头行。"""

    candidate: TableCandidate
    rows: list[list[str]]
    extractor: str  # "docling" | "ppstructure" | ...
    notes: list[str] = field(default_factory=list)
    coverage: float | None = None  # 单元格总字符 ÷ 区域文字层字符
    plumber_cols: int | None = None  # pdfplumber-in-bbox 的列数（第二意见）
    confidence: str = "medium"
    grid_status: str = "ok"
    matched_keywords: list[str] = field(default_factory=list)
    matched_on: str = ""  # "caption" | "legend" | "image_text"
    continued_from: str | None = None  # 拼接进了哪个 table_id

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    @property
    def cell_chars(self) -> int:
        return sum(len(c.strip()) for r in self.rows for c in r if c)

    # 关键词匹配逻辑在 rules.match_table() —— 放这里会与 rules 成环（rules 依赖 models）。
    @property
    def caption_text(self) -> str:
        return self.candidate.label.text if self.candidate.label else ""

    @property
    def content_text(self) -> str:
        """整张表的单元格文本拼起来（含表头行）。"""
        return " ".join(c for r in self.rows for c in r if c)


@dataclass
class ManifestRow:
    """落盘的一行。失败的表也必须有一行（铁律 #2：不静默丢表）。

    manifest 是**累加**的：同一篇 PDF 始终一个文件夹，反复用不同关键词搜，
    每次追加行。`query` 列记录那一次搜的是什么，所以"用 a 搜到 table1、
    用 b 也搜到 table1"会是两行，各自带自己的 query。
    这样你不必为了不覆盖记录而去手工起 --prefix。
    """

    table_id: str
    label: str
    page: int
    caption: str
    csv_path: str
    extractor: str
    source_type: str
    matched_on: str
    matched_keywords: str
    n_rows: int
    n_cols: int
    coverage: str
    plumber_cols: str
    confidence: str
    grid_status: str
    notes: str
    query: str = ""  # 那一次运行搜的关键词（含 AND/OR 模式），也是去重键
    run_at: str = ""  # 本地时间，便于排序

    COLUMNS = (
        "query",
        "run_at",
        "table_id",
        "label",
        "page",
        "caption",
        "csv_path",
        "extractor",
        "source_type",
        "matched_on",
        "matched_keywords",
        "n_rows",
        "n_cols",
        "coverage",
        "plumber_cols",
        "confidence",
        "grid_status",
        "notes",
    )
