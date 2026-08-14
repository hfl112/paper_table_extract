"""Base layer shared by all modes. Imports nothing from this package.

Contents: third-party log silencing, the data structures every stage exchanges,
keyword loading/matching semantics, pre-extraction judgement (labels, captions,
rotation, dispatch, continuation), and figure-panel grid geometry.
Everything here is pure stdlib and unit-testable without a PDF.
"""

from __future__ import annotations

import logging
import os
import re
import statistics
import warnings
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# third-party log silencing (must run before any engine import)
# ---------------------------------------------------------------------------

_NOISY = (
    "paddle", "paddleocr", "paddlex", "docling", "docling_core", "docling_ibm_models",
    "RapidOCR", "rapidocr", "pypdfium2", "torch", "PIL", "matplotlib",
)
_env_done = False


def silence() -> None:
    """Set env vars and hush loggers. Call once before any engine import."""
    global _env_done
    if not _env_done:
        _env_done = True
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TQDM_DISABLE", "1")
        # paddlex probes model sources at startup: seconds of delay plus log spam
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("GLOG_minloglevel", "2")
        # torch.compile is a net loss for one-shot CLI runs and crashes on old g++ (c++20);
        # setdefault, so exporting the variable yourself still overrides
        os.environ.setdefault("DOCLING_INFERENCE_COMPILE_TORCH_MODELS", "0")
        warnings.filterwarnings("ignore")
    hush_loggers()


def hush_loggers() -> None:
    """Drop noisy loggers to WARNING. Call again AFTER each engine import: paddlex resets its own logger to INFO at import time."""
    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
        prefix = name + "."
        for existing in list(logging.Logger.manager.loggerDict):
            if existing.startswith(prefix):
                logging.getLogger(existing).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# data structures (stages exchange only these, never library objects)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rect:
    """Page rectangle, top-left origin, pt units. Deliberately not fitz.Rect so PyMuPDF stays inside pdfio."""

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
    """One text-layer word with its rectangle (already in the rotated frame for rotated pages)."""

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
    """Geometry + indices of one docling TableCell.

    Needed separately from the exported rows: the two views disagree on row
    numbering for 184/427 measured tables (multi-line headers get flattened),
    so merged-row detection must match by content, never by row index.
    """

    text: str
    r0: int
    c0: int
    row_span: int
    col_span: int
    column_header: bool  # docling's own header verdict; never guess header depth yourself
    bbox: Rect | None


@dataclass(frozen=True)
class ImageInfo:
    """One embedded bitmap on a page."""

    xref: int
    rect: Rect
    px_width: int
    px_height: int
    cover_w: float  # fraction of page width, 0-1
    cover_h: float  # fraction of page height, 0-1


@dataclass
class Label:
    """One occurrence of a table/figure label."""

    raw: str  # as printed, e.g. "TABLE I"
    key: str  # normalized, e.g. "table i"
    kind: str  # "table" | "figure"
    page: int  # 1-based
    is_caption: bool  # True = real caption (line-start rule); False = body reference
    text: str  # caption+legend when is_caption, else the whole containing block
    rect: Rect | None = None

    @property
    def is_supplementary(self) -> bool:
        return self.key.startswith("supp ")


@dataclass
class PageInfo:
    """Structural features of one page, from text layer and drawing objects only."""

    page: int  # 1-based
    width: float
    height: float
    n_chars: int
    vert_lines: int
    horiz_lines: int
    images: list[ImageInfo] = field(default_factory=list)
    ocr_text_layer: bool = False  # page covered by one bitmap in a scanned doc
    needs_rotation: int = 0  # 0 / 90 / 270


@dataclass
class DocInfo:
    """Whole-document features."""

    path: str
    n_pages: int
    pages: list[PageInfo]
    metadata: dict[str, str]
    preprint_markers: dict[str, int] = field(default_factory=dict)
    preprint_pages: int = 0  # pages carrying a watermark word; the ratio is the criterion
    p1_watermark: str = ""  # vertical preprint watermark on page 1 (arXiv style)

    @property
    def total_chars(self) -> int:
        return sum(p.n_chars for p in self.pages)


@dataclass
class TableCandidate:
    """One extraction candidate: where it is and which path it takes."""

    page: int
    rect: Rect
    source_type: str  # "text" (text layer inside) | "image" (pixels)
    label: Label | None = None
    chars_in_rect: int = 0
    origin: str = ""  # "docling_table" | "docling_picture" | "raster_image"
    image: ImageInfo | None = None


@dataclass
class ExtractedTable:
    """One extraction result. rows includes the header row."""

    candidate: TableCandidate
    rows: list[list[str]]
    extractor: str  # "docling" | "dual_ocr"
    notes: list[str] = field(default_factory=list)
    coverage: float | None = None
    plumber_cols: int | None = None
    confidence: str = "medium"
    grid_status: str = "ok"
    matched_keywords: list[str] = field(default_factory=list)
    matched_on: str = ""  # "caption" | "cell" | "caption+cell" | "image_text"

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        return max((len(r) for r in self.rows), default=0)

    @property
    def cell_chars(self) -> int:
        return sum(len(c.strip()) for r in self.rows for c in r if c)

    @property
    def caption_text(self) -> str:
        return self.candidate.label.text if self.candidate.label else ""

    @property
    def content_text(self) -> str:
        return " ".join(c for r in self.rows for c in r if c)


@dataclass
class ManifestRow:
    """One manifest line. Failed/skipped tables get a line too (never drop silently).

    The manifest accumulates: rerunning the same paper with other keywords appends
    rows; the query column records what each run searched and doubles as the
    replace key when the exact same query is rerun.
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
    query: str = ""
    run_at: str = ""

    COLUMNS = (
        "query", "run_at", "table_id", "label", "page", "caption", "csv_path",
        "extractor", "source_type", "matched_on", "matched_keywords", "n_rows",
        "n_cols", "coverage", "plumber_cols", "confidence", "grid_status", "notes",
    )


# ---------------------------------------------------------------------------
# keyword loading and matching semantics
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")


def collapse_ws(text: str) -> str:
    """Collapse all whitespace to single spaces. Mandatory before label matching: the same label wraps differently across PDFs."""
    return _WS.sub(" ", text).strip()


def read_words(path: str | Path) -> list[str]:
    """Keyword/word-list file: one word per line, # comments allowed."""
    return [
        ln.strip()
        for ln in Path(path).read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]


_VOWELS = "aeiou"
_NO_DOUBLE = "wxyhjq"


def _morph_variants(word: str) -> list[str]:
    """Common English inflections of one word; pure morphology, no domain knowledge."""
    w = word.lower()
    out = [w]
    if any(ch.isdigit() for ch in w):
        return out  # codes like PD1/S2 do not inflect
    out += [w + "s", w + "es", w + "d", w + "ed", w + "ing"]
    if w.endswith("y") and len(w) > 2 and w[-2] not in _VOWELS:
        out.append(w[:-1] + "ies")  # toxicity -> toxicities
    if w.endswith("e"):
        out += [w[:-1] + "ing", w[:-1] + "ed"]  # dose -> dosing
    if (
        len(w) >= 3
        and w[-1] not in _VOWELS
        and w[-1] not in _NO_DOUBLE
        and w[-2] in _VOWELS
        and w[-3] not in _VOWELS
    ):
        out += [w + w[-1] + "ed", w + w[-1] + "ing"]  # control -> controlled
    return sorted(set(out), key=len, reverse=True)


def keyword_regex(keyword: str) -> re.Pattern[str]:
    """Whole-word + inflections, case-insensitive; hyphen counts as a word character.

    Measured: substring matching makes 2-3 letter keywords pure noise (PR hits 130
    times inside words); plain whole-word misses inflections (xenograft 11 vs 38
    hits with variants). Hyphen-as-word-char keeps PR-104/NB-SD/PD-L1 from matching
    PR/SD/PD: every hyphenated token in the corpus is an identifier.
    """
    kw = collapse_ws(keyword)
    parts = kw.split(" ")
    head = r"\s+".join(re.escape(p) for p in parts[:-1])
    tail_alts = "|".join(re.escape(v) for v in _morph_variants(parts[-1]))
    body = (head + r"\s+" if head else "") + f"(?:{tail_alts})"
    return re.compile(rf"(?<![\w-]){body}(?![\w-])", re.I)


def keyword_hits(text: str, keywords: list[str]) -> list[str]:
    """Keywords that match in text, in the given order."""
    return [k for k in keywords if keyword_regex(k).search(text)]


def match_table(table: ExtractedTable, keywords: list[str], *, include_content: bool = True) -> tuple[list[str], str]:
    """Match a table against keywords; returns (hits, where).

    Text tables must match cell content too, and it is free (already in memory).
    Measured: pbc_28772's gold table has no keyword in the caption but an
    'Obj. Response' column header. include_content=False is for image tables,
    where content costs OCR and the caller decides the gating.
    """
    cap_hits = keyword_hits(table.caption_text, keywords)
    body_hits = keyword_hits(table.content_text if include_content else "", keywords)
    union = [k for k in keywords if k in cap_hits or k in body_hits]
    if not union:
        return [], ""
    if cap_hits and body_hits and set(cap_hits) != set(body_hits):
        return union, "caption+cell"
    return union, ("caption" if cap_hits else "cell")


# sentence splitting protects common abbreviations, otherwise "Fig. 1" splits a sentence
_ABBREVS = (
    "fig", "figs", "tab", "tabs", "al", "e.g", "i.e", "cf", "vs", "no", "nos",
    "ref", "refs", "approx", "ca", "dr", "mr", "ms", "st", "eq", "ch", "pp",
)
_SPLIT_CANDIDATE = re.compile(r"\.(?=\s+[A-Z(])")
_WORD_BEFORE = re.compile(r"([A-Za-z.]+)$")


def split_sentences(text: str) -> list[str]:
    """Conservative splitting: period + whitespace + capital, protecting abbreviations and decimals."""
    flat = collapse_ws(text)
    flat = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", flat)
    cuts: list[int] = []
    for m in _SPLIT_CANDIDATE.finditer(flat):
        before = _WORD_BEFORE.search(flat[: m.start()])
        if before and before.group(1).lower().rstrip(".") in _ABBREVS:
            continue
        cuts.append(m.start())
    parts, prev = [], 0
    for c in cuts + [len(flat)]:
        seg = flat[prev:c].strip()
        if seg:
            parts.append(seg.replace("<DOT>", "."))
        prev = c + 1
    return parts


def cooccurs_in_sentence(text: str, label_raw: str, keywords: list[str]) -> list[str]:
    """Keywords co-occurring with the label in the SAME sentence (paragraph scope creates false links).

    Whole-word semantics like everywhere else: substring matching makes short keywords noise.
    """
    lab = collapse_ws(label_raw).lower()
    hits: set[str] = set()
    for sent in split_sentences(text):
        if lab in sent.lower():
            hits.update(keyword_hits(sent, keywords))
    return sorted(hits)


# ---------------------------------------------------------------------------
# label conventions (typesetting knowledge, journal-agnostic)
# ---------------------------------------------------------------------------

# Label core regex. Prefix variants are journal conventions measured in the corpora;
# eTable (JAMA) and Additional file (BMC) are deliberately absent (0 instances / handled via Table SN).
DEFAULT_LABEL_CORE = (
    r"(?:Supplementary\s+Data\s+|Supplementary\s+|Supplemental\s+|Extended\s+Data\s+|Online\s+)?"
    r"(?:Tables?|TABLES?|Tabs?\.|Figures?|FIGURES?|Figs?\.?)\s+"
    r"(?:S?\d+|[IVXivx]{1,4})"
)
# caption match may end at end-of-line ("TABLE 1" alone on its line is a real caption)
_CAPTION_TAIL = r"(?:[.:\s]|$)"
# reference search ends at a word boundary ("Table IV," / "(Table IV)" must match)
_LOOSE_TAIL = r"\b"
CAPTION_MAX_OFFSET = 0  # a real caption starts its line


def normalize_label(raw: str) -> str:
    """Fold label spellings to one key: TABLE I / Table I / Tab. I -> table i; Supplementary/Supplemental -> supp."""
    s = collapse_ws(raw).lower()
    s = s.replace("supplementary data ", "supp ").replace("supplementary ", "supp ")
    s = s.replace("supplemental ", "supp ")
    s = re.sub(r"\btabs?\.", "table", s)
    s = re.sub(r"\btables\b", "table", s)
    s = re.sub(r"\bfigures?\b", "fig", s)
    s = re.sub(r"\bfigs?\.", "fig", s)
    s = re.sub(r"\bfigs\b", "fig", s)
    s = re.sub(r"[.:,;]+$", "", s)
    s = collapse_ws(s)
    # "Supplementary Table S3" and "Table S3" are the same table when the number carries the S
    s = re.sub(r"^supp (table|fig) (s\d)", r"\1 \2", s)
    return s


def label_kind(key: str) -> str:
    return "table" if "table" in key else "figure"


def compile_label_pattern(core: str | None = None) -> re.Pattern[str]:
    """Caption criterion: anchored at line start, may end at end-of-line."""
    return re.compile(rf"^(?P<label>{core or DEFAULT_LABEL_CORE}){_CAPTION_TAIL}")


def compile_loose_pattern(core: str | None = None) -> re.Pattern[str]:
    """Reference search: unanchored, word-boundary tail."""
    return re.compile(rf"(?P<label>{core or DEFAULT_LABEL_CORE}){_LOOSE_TAIL}")


def match_label_at_line_start(line: str, pat: re.Pattern[str]) -> str | None:
    """Line-start label = caption; anywhere else = body reference. Measured 6/6 papers correct.

    The pattern deliberately does NOT require all-caps (that is one journal's habit);
    it enumerates measured case variants instead of using re.IGNORECASE, which would
    also accept lowercase 'table 1' sentence starts as captions.
    """
    m = pat.match(collapse_ws(line))
    if m and m.start() <= CAPTION_MAX_OFFSET:
        return m.group("label") if "label" in (m.groupdict() or {}) else m.group(0)
    return None


_SENTENCE_END = re.compile(r"[.!?:][\"')\]]*$")


def caption_line_is_plausible(prev_line: str | None) -> bool:
    """Second check for line-start hits: a reference wrapped to line start has an unfinished sentence above it."""
    if prev_line is None:
        return True
    prev = collapse_ws(prev_line)
    if not prev:
        return True
    return bool(_SENTENCE_END.search(prev))


def find_labels_anywhere(text: str, loose_pat: re.Pattern[str]) -> list[str]:
    """All labels in a text block (for the referenced-only inventory). Pass compile_loose_pattern output."""
    return [m.group("label") for m in loose_pat.finditer(collapse_ws(text))]


# legend continuation: gap and prose-line length must BOTH hold, measured on four cases
LEGEND_MAX_BLOCK_GAP = 15.0  # pt
LEGEND_MIN_PROSE_LINE = 40  # chars


def is_legend_continuation(gap: float, max_line_len: int) -> bool:
    """Is the block after a caption a legend continuation (not table body or footer)?"""
    return gap < LEGEND_MAX_BLOCK_GAP and max_line_len >= LEGEND_MIN_PROSE_LINE


def pair_label_to_candidate(label: Label, candidates: list[TableCandidate]) -> TableCandidate | None:
    """Pair a caption with a same-page candidate. Convention: table captions sit above the table, figure captions below."""
    same_page = [c for c in candidates if c.page == label.page]
    if not same_page:
        return None
    if label.rect is None:
        return same_page[0]
    want_below = label.kind == "table"
    y = label.rect.y1 if want_below else label.rect.y0

    def sort_key(c: TableCandidate) -> tuple[int, float]:
        if want_below:
            return (0 if c.rect.y0 >= y - 4 else 1, abs(c.rect.y0 - y))
        return (0 if c.rect.y1 <= y + 4 else 1, abs(y - c.rect.y1))

    return sorted(same_page, key=sort_key)[0]


# ---------------------------------------------------------------------------
# continuation criteria (both are needed; each alone misses a measured case)
# ---------------------------------------------------------------------------

_CONTINUED = re.compile(r"\(\s*continued\s*\)", re.I)


def caption_says_continued(caption: str) -> bool:
    """Criterion 1: caption literally says (Continued), incl. '( Continued )' spacing."""
    return bool(_CONTINUED.search(collapse_ws(caption)))


def same_column_names(a: list[str], b: list[str]) -> bool:
    """Criterion 2: identical column names. Needed because continuation pages may have an empty caption."""
    if not a or not b or len(a) != len(b):
        return False
    return [collapse_ws(x).lower() for x in a] == [collapse_ws(x).lower() for x in b]


def is_continuation(prev: ExtractedTable, cur: ExtractedTable, *, max_page_gap: int = 1, prev_page: int | None = None) -> str | None:
    """Is cur a continuation of prev? Returns the criterion name or None.

    prev_page lets the caller pass the LAST stitched page of prev; without it a
    table spanning 3+ pages fails the gap check after the second page.
    """
    base = prev.candidate.page if prev_page is None else prev_page
    if cur.candidate.page - base > max_page_gap:
        return None
    cur_caption = cur.candidate.label.text if cur.candidate.label else ""
    if caption_says_continued(cur_caption):
        return "caption_continued"
    prev_cols = prev.rows[0] if prev.rows else []
    cur_cols = cur.rows[0] if cur.rows else []
    if same_column_names(prev_cols, cur_cols):
        return "same_columns"
    return None


# ---------------------------------------------------------------------------
# page-structure judgement (rotation, scanned docs, dispatch)
# ---------------------------------------------------------------------------

# rotated-table pages: measured 134-446 vertical lines vs 1 for side watermarks
VERTICAL_LINES_THRESHOLD = 30
# and >=50% of all lines vertical (multi-panel axis labels reach 41-79 lines but low ratio)
VERTICAL_RATIO_THRESHOLD = 0.5


def rotation_for_page(
    vert_lines: int,
    dominant_dir: tuple[int, int],
    horiz_lines: int = 0,
    *,
    has_table_label: bool = False,
    prev_page_rotated: bool = False,
) -> int:
    """Degrees to rotate a sideways-table page. Three criteria, all measured necessary.

    Count + ratio alone still misclassify per-character watermark lines and dense
    axis labels; the third criterion is a Table label inside the vertical text
    (8/8 true positives, 0/6 false ones). prev_page_rotated covers continuation
    pages whose caption is empty.
    """
    if vert_lines < VERTICAL_LINES_THRESHOLD:
        return 0
    total = vert_lines + horiz_lines
    if total and vert_lines / total < VERTICAL_RATIO_THRESHOLD:
        return 0
    if not (has_table_label or prev_page_rotated):
        return 0
    return 270 if dominant_dir == (0, 1) else 90


FULL_PAGE_IMAGE_COVER = 0.95  # one bitmap covering >=95% of both page dimensions
OCR_LAYER_MIN_CHARS = 1000  # true scans average >=~2163 chars/page; image-only pages ~185
SCANNED_DOC_PAGE_RATIO = 0.8  # true scans: 97-100% full-bitmap pages; modern papers: 6-21%


def has_full_page_image(images: list[ImageInfo]) -> bool:
    return any(
        im.cover_w >= FULL_PAGE_IMAGE_COVER and im.cover_h >= FULL_PAGE_IMAGE_COVER
        for im in images
    )


def is_scanned_document(n_full_page_images: int, n_pages: int, total_chars: int) -> bool:
    """Scanned-with-publisher-OCR is a whole-document property, not per page (measured on 90 papers)."""
    if not n_pages:
        return False
    if n_full_page_images / n_pages < SCANNED_DOC_PAGE_RATIO:
        return False
    return total_chars / n_pages >= OCR_LAYER_MIN_CHARS


def is_ocr_text_layer(images: list[ImageInfo], *, doc_is_scanned: bool) -> bool:
    """This page's text layer is publisher OCR: document is scanned AND this page is bitmap-covered."""
    return doc_is_scanned and has_full_page_image(images)


OCR_PRODUCER_RE = re.compile(r"ABBYY|Recognition|Tesseract|FineReader|OmniPage|ocrmypdf|Scansoft", re.I)


def metadata_suggests_ocr(metadata: dict[str, str]) -> bool:
    """OCR producer words in metadata confirm but never refute (often empty on true scans)."""
    blob = " ".join(v or "" for v in metadata.values())
    return bool(OCR_PRODUCER_RE.search(blob))


MIN_IMAGE_COVER_H = 0.08  # journal logos are <4% page height; real content >=13%


def is_decorative_image(cover_h: float) -> bool:
    return cover_h < MIN_IMAGE_COVER_H


TEXT_IN_REGION_THRESHOLD = 30  # measured: bitmap table regions have 0 chars, text tables 1000+


def dispatch_source_type(chars_in_rect: int) -> str:
    """Text-layer chars inside the region decide the path; covers vector vs bitmap automatically."""
    return "text" if chars_in_rect >= TEXT_IN_REGION_THRESHOLD else "image"


def region_chars_of(rect: Rect, spans: list[tuple[Rect, str]]) -> int:
    """Count text-layer characters whose span center falls inside rect."""
    total = 0
    for r, text in spans:
        if rect.contains_point((r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2):
            total += len(text.strip())
    return total


# ---------------------------------------------------------------------------
# figure-panel grid geometry (dual-read figure mode)
# ---------------------------------------------------------------------------

Box = tuple[float, float, float, float]

CELL_GATE_MIN = 20  # cell-detection gate: tables give ~300 boxes, pure charts 0-6
# raster legibility gate: NATIVE pixels per cell decide OCR viability, not dpi
# (measured: two 120-dpi figures, 12.9px native cell height read at 100%, 8.8px was garbage)
NATIVE_CELL_MIN_PX = 10.5


def iou_dedupe(boxes: list[tuple[Box, float]]) -> list[Box]:
    """Overlapping detections (IoU > 0.5) keep only the higher-score box."""
    keep: list[tuple[Box, float]] = []
    for c, s in sorted(boxes, key=lambda b: -b[1]):
        x1, y1, x2, y2 = c
        dup = False
        for (kx1, ky1, kx2, ky2), _ in keep:
            ox = min(x2, kx2) - max(x1, kx1)
            oy = min(y2, ky2) - max(y1, ky1)
            if ox > 0 and oy > 0:
                inter = ox * oy
                union = (x2 - x1) * (y2 - y1) + (kx2 - kx1) * (ky2 - ky1) - inter
                if inter / union > 0.5:
                    dup = True
                    break
        if not dup:
            keep.append((c, s))
    return [c for c, _ in keep]


def split_panel(boxes: list[Box]) -> tuple[list[Box], int]:
    """Cut at the widest x gap when it exceeds 2x median cell width (drops a side bar-chart panel).

    Returns (kept boxes, dropped count); the caller must record the drop in the manifest.
    """
    med_w = statistics.median(x2 - x1 for x1, _, x2, _ in boxes)
    best, cut, cur_end = 0.0, None, None
    for x1, _, x2, _ in sorted(boxes, key=lambda b: b[0]):
        if cur_end is not None and x1 - cur_end > best:
            best, cut = x1 - cur_end, (cur_end + x1) / 2
        cur_end = x2 if cur_end is None else max(cur_end, x2)
    dropped = 0
    if best > 2 * med_w and cut is not None:
        dropped = sum(1 for b in boxes if (b[0] + b[2]) / 2 >= cut)
        boxes = [b for b in boxes if (b[0] + b[2]) / 2 < cut]
    return boxes, dropped


def drop_giant_boxes(boxes: list[Box]) -> list[Box]:
    """Boxes far larger than a table cell (>6x median area) are caption/body text, not cells."""
    med_a = statistics.median((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes)
    return [b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) < 6 * med_a]


def bands_1d(centers: list[float], tol: float) -> list[float]:
    """Single-link 1-d clustering; returns band centers."""
    out: list[list[float]] = []
    for c in sorted(centers):
        if out and c - out[-1][-1] <= tol:
            out[-1].append(c)
        else:
            out.append([c])
    return [statistics.mean(b) for b in out]


def panel_geometry(boxes: list[Box]) -> tuple[list[float], list[float], float]:
    """Row anchors from the first column (equal pitch, synthetic anchors fill gaps) + column bands.

    Measured: anchoring rows on first-column boxes eliminates the adjacent-row
    merging that transitive clustering produces near off-grid bar-chart labels.
    Falls back to plain row clustering when the first column yields no usable pitch.
    Returns (row_centers, col_centers, pitch).
    """
    med_w = statistics.median(x2 - x1 for x1, _, x2, _ in boxes)
    cols = bands_1d([(x1 + x2) / 2 for x1, _, x2, _ in boxes], med_w * 0.6)
    first = [b for b in boxes if abs((b[0] + b[2]) / 2 - cols[0]) <= med_w * 0.6]
    ys = sorted((y1 + y2) / 2 for _, y1, _, y2 in first)
    diffs = [b - a for a, b in zip(ys, ys[1:]) if b - a > 1]
    if len(diffs) < 2:
        med_h = statistics.median(y2 - y1 for _, y1, _, y2 in boxes)
        rows = bands_1d([(y1 + y2) / 2 for _, y1, _, y2 in boxes], med_h * 0.6)
        gaps = [b - a for a, b in zip(rows, rows[1:])]
        return rows, cols, (statistics.median(gaps) if gaps else med_h)
    pitch = statistics.median(diffs)
    rows: list[float] = []
    for y in ys:
        if rows and y - rows[-1] < 0.5 * pitch:
            continue
        while rows and y - rows[-1] > 1.5 * pitch:
            rows.append(rows[-1] + pitch)  # synthetic anchor: first-column box missing on that row
        rows.append(y)
    return rows, cols, pitch


def snap_to_grid(rows: list[float], cols: list[float], pitch: float, col_tol: float, x: float, y: float) -> tuple[int, int] | None:
    """Assign a box center to (row, col); strays beyond 0.6 tolerance on either axis are dropped."""
    i = min(range(len(rows)), key=lambda k: abs(rows[k] - y))
    if abs(rows[i] - y) > 0.6 * pitch:
        return None
    j = min(range(len(cols)), key=lambda k: abs(cols[k] - x))
    if abs(cols[j] - x) > col_tol:
        return None
    return i, j


def attach_labels(cands: list[TableCandidate], caps: list[Label]) -> None:
    """Pair captions to candidate regions, each caption at most once, in reading order."""
    used: set[int] = set()
    for lab in sorted(caps, key=lambda l: (l.page, l.rect.y0 if l.rect else 0)):
        pool = [c for i, c in enumerate(cands) if i not in used and c.page == lab.page]
        if not pool:
            continue
        pick = pair_label_to_candidate(lab, pool)
        if pick is None:
            continue
        pick.label = lab
        used.add(cands.index(pick))
