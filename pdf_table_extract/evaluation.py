"""Reliability evaluation of extraction results (the extracted CSV, not the PDF).

Contents: content-coverage audit, grid second opinions, merged-row detection,
hyphen-break rejoining, OCR artifact checks, caption-ownership verification,
confidence grading, and the dual-read merge with per-cell conflict flags.
Pure stdlib; unit-testable without a PDF.
"""

from __future__ import annotations

import re

from .common import ExtractedTable, Label, TableCellBox, Word, collapse_ws

# ---------------------------------------------------------------------------
# content coverage (extracted chars vs text-layer chars in the region)
# ---------------------------------------------------------------------------

COVERAGE_LOW = 0.35  # legitimate merged cells lose no chars; recognizer collapses do


def coverage_ratio(cell_chars: int, region_chars: int) -> float | None:
    """Extracted cell chars / text-layer chars in the region; ~2% means a structural collapse."""
    if region_chars <= 0:
        return None
    return cell_chars / region_chars


# ---------------------------------------------------------------------------
# grid second opinions
# ---------------------------------------------------------------------------

# column diff of 1 = a whole lost column and must flag; row diff of 1 is header-band noise
GRID_COL_TOLERANCE = 0
GRID_ROW_TOLERANCE = 1


def grid_shape_mismatch(det_rows: int, det_cols: int, n_rows: int, n_cols: int) -> str | None:
    """Compare detector row/col band counts against the reconstructed grid; free second opinion for image tables."""
    if det_rows <= 0 or det_cols <= 0 or n_rows <= 0 or n_cols <= 0:
        return None
    bad = []
    if abs(det_cols - n_cols) > GRID_COL_TOLERANCE:
        bad.append(f"cols detected={det_cols}/reconstructed={n_cols}")
    if abs(det_rows - n_rows) > GRID_ROW_TOLERANCE:
        bad.append(f"rows detected={det_rows}/reconstructed={n_rows}")
    if not bad:
        return None
    return "grid_shape_mismatch(" + ", ".join(bad) + ")"


# ---------------------------------------------------------------------------
# hyphen-break rejoining (docling keeps "cyclophos- phamide" split inside one cell)
# ---------------------------------------------------------------------------

# regex alone cannot distinguish a line-break split from a legal compound
# (lineage-defining, p-value); the joined string must exist as a real word
# in this paper's text layer at least MIN_WORD_COUNT times (measured: 2, tightest margin)
_HYPHEN_BREAK = re.compile(r"([a-z])-\s+([a-z])")
_WORD_RUN = re.compile(r"[a-z]+")
MIN_WORD_COUNT = 2


def has_hyphen_break(rows: list[list[str]]) -> bool:
    """Cheap prefilter so the whole-document word count is only computed when needed."""
    return any(_HYPHEN_BREAK.search(c) for row in rows for c in row)


def rejoin_hyphen_breaks(rows: list[list[str]], word_counts: dict[str, int]) -> tuple[list[list[str]], int]:
    """Rejoin line-break hyphens. Returns (NEW rows, join count); never mutates in place."""
    n = 0

    def fix(s: str) -> str:
        nonlocal n

        def repl(m: re.Match) -> str:
            nonlocal n
            left = (_WORD_RUN.findall(s[: m.start(1) + 1].lower()) or [""])[-1]
            right = (_WORD_RUN.findall(s[m.end(2) - 1 :].lower()) or [""])[0]
            joined = left + right
            if joined and word_counts.get(joined, 0) >= MIN_WORD_COUNT:
                n += 1
                return m.group(1) + m.group(2)
            return m.group(0)

        return _HYPHEN_BREAK.sub(repl, s)

    return [[fix(c) for c in row] for row in rows], n


# ---------------------------------------------------------------------------
# OCR artifact checks
# ---------------------------------------------------------------------------

_LEADING_LT = re.compile(r"^\s*<")
_LEADING_MINUS = re.compile(r"^\s*-\s*\d")


def suspect_lt_as_minus(rows: list[list[str]]) -> list[str]:
    """Catch OCR reading '<' as '-' (<0.001 -> -0.001): a column mostly starting with '<' plus one '-' cell.

    The most dangerous OCR error class: the result is a perfectly legal-looking number.
    Column character-set consistency only; no domain knowledge.
    """
    if len(rows) < 3:
        return []
    n_cols = max(len(r) for r in rows)
    out: list[str] = []
    for c in range(n_cols):
        col = [(i, r[c]) for i, r in enumerate(rows) if c < len(r) and r[c]]
        if len(col) < 3:
            continue
        n_lt = sum(1 for _, v in col if _LEADING_LT.match(v))
        if n_lt < max(2, len(col) // 3):
            continue
        for i, v in col:
            if _LEADING_MINUS.match(v):
                out.append(f"suspect_lt_as_minus(R{i}C{c}={v.strip()!r})")
    return out


# docling leaks unmapped glyph names into cells (/emspaceMean -> Mean); only known
# whitespace glyphs are stripped and the payload kept (measured: /emspaceClear cell is real data)
_SPACE_GLYPHS = "em|en|thin|hair|figure|punctuation|third|quarter|sixth|zerowidth|nb|no-break"
_GLYPH_NAME = re.compile(rf"^/(?:{_SPACE_GLYPHS})space\s*", re.I)


def strip_glyph_names(rows: list[list[str]]) -> list[list[str]]:
    """Strip leaked whitespace glyph names from cell starts, in place."""
    for row in rows:
        for i, cell in enumerate(row):
            if cell.startswith("/"):
                row[i] = _GLYPH_NAME.sub("", cell)
    return rows


# ---------------------------------------------------------------------------
# caption ownership (a grid must prove it belongs to the caption it claims)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "table", "figure", "fig", "supplementary", "supp", "continued", "of", "the",
    "in", "for", "and", "a", "an", "to", "on", "with", "from", "by", "as", "at",
    "all", "summary", "results", "data",
}
MIN_RIVAL_OVERLAP = 2  # a rival caption needs >=2 content words in the grid to count as positive evidence


def caption_words(caption: str) -> set[str]:
    """Content words of a caption; the stopword list is typographic/grammatical, not domain vocabulary."""
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", collapse_ws(caption).lower())
    return {w for w in words if w not in _STOPWORDS}


def _grid_haystack(rows: list[list[str]]) -> str:
    header = " ".join(rows[0]) if rows else ""
    first_col = " ".join(r[0] for r in rows if r)
    return collapse_ws(f"{header} {first_col}").lower()


def _overlap(caption: str, hay: str) -> int:
    return sum(1 for w in caption_words(caption) if w in hay)


def verify_caption_belongs(caption: str, rows: list[list[str]], *, other_captions: list[str] | None = None) -> bool:
    """Mismatch only on POSITIVE counter-evidence: this caption overlaps the grid zero
    times while another same-kind caption overlaps it >= MIN_RIVAL_OVERLAP words.

    Absence of overlap alone is normal (captions state topics, headers list variables).
    Pass only same-kind captions: figure legends naturally enumerate table row names.
    """
    if not rows:
        return True
    hay = _grid_haystack(rows)
    if _overlap(caption, hay) > 0:
        return True
    best_other = max((_overlap(c, hay) for c in (other_captions or [])), default=0)
    return best_other < MIN_RIVAL_OVERLAP


# ---------------------------------------------------------------------------
# merged-row detection (two data rows squeezed into one cell; detect, never split)
# ---------------------------------------------------------------------------

# auto-splitting measurably loses characters; detection only, with confidence downgrade.
# four layers measured at ~90% precision on 90 papers (each layer necessary)
MERGED_ROW_NOTE = "merged_row?="
CELL_TALL = 1.5
WRAP_MAX_COLS = 2
MIN_COL_SAMPLES = 6
MIN_EVEN_SAMPLES = 4
BAND_GAP = 3.0
HEIGHT_EVEN_MAX = 0.30  # the tall-cell signal only means something in an evenly-set column


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if not n:
        return 0.0
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _column_evenness(heights: list[tuple[int, float]], skip_row: int) -> float | None:
    hs = [h for r, h in heights if r != skip_row]
    if len(hs) < MIN_EVEN_SAMPLES:
        return None
    med = _median(hs)
    if med <= 0:
        return None
    return (max(hs) - min(hs)) / med


def suspicious_tall_rows(cells: list[TableCellBox]) -> dict[int, list[int]]:
    """Layers 1-3: {cell row index: [evidence columns]}, rows with >=2 evidence columns only."""
    header_rows = {c.r0 for c in cells if c.column_header}
    by_col: dict[int, list[tuple[int, float]]] = {}
    for c in cells:
        if c.bbox is None or c.row_span != 1 or c.r0 in header_rows:
            continue
        h = abs(c.bbox.y1 - c.bbox.y0)
        if h > 0:
            by_col.setdefault(c.c0, []).append((c.r0, h))
    sus: dict[int, list[int]] = {}
    for col, heights in by_col.items():
        if len(heights) < MIN_COL_SAMPLES:
            continue
        med = _median([h for _, h in heights])
        if med <= 0:
            continue
        for r0, h in heights:
            if h <= CELL_TALL * med:
                continue
            ev = _column_evenness(heights, r0)
            if ev is None or ev > HEIGHT_EVEN_MAX:
                continue
            sus.setdefault(r0, []).append(col)
    return {r: cols for r, cols in sus.items() if len(cols) >= 2}


def _bands_by_y(words: list[Word], gap: float = BAND_GAP) -> list[list[Word]]:
    if not words:
        return []
    ordered = sorted(words, key=lambda w: w.cy)
    out: list[list[Word]] = [[ordered[0]]]
    prev = ordered[0].cy
    for w in ordered[1:]:
        if w.cy - prev > gap:
            out.append([])
        out[-1].append(w)
        prev = w.cy
    return out


def confirm_merged_row(cells: list[TableCellBox], row_idx: int, words: list[Word]) -> int:
    """Layer 4: columns with multiple y bands in that row. Label wrapping affects 1-2 columns, true merges 3+."""
    n = 0
    for c in cells:
        if c.r0 != row_idx or c.bbox is None:
            continue
        inside = [
            w for w in words
            if c.bbox.x0 - 1 <= w.cx <= c.bbox.x1 + 1 and c.bbox.y0 - 1 <= w.cy <= c.bbox.y1 + 1
        ]
        if len(_bands_by_y(inside)) >= 2:
            n += 1
    return n


def match_cell_row_to_output_row(cells: list[TableCellBox], row_idx: int, rows: list[list[str]]) -> int | None:
    """Translate a cell row index to an output row index BY CONTENT: the two views disagree on numbering (184/427 tables)."""
    want = {"".join(c.text.split()).lower() for c in cells if c.r0 == row_idx and c.text.strip()}
    if not want:
        return None
    best_i, best_ov = None, 0
    for i, row in enumerate(rows):
        have = {"".join(v.split()).lower() for v in row if v.strip()}
        ov = len(want & have)
        if ov > best_ov:
            best_i, best_ov = i, ov
    return best_i if best_ov else None


def detect_merged_rows(cells: list[TableCellBox], rows: list[list[str]], words: list[Word]) -> list[tuple[int, list[str]]]:
    """All four layers chained; returns [(output row index, that row's cell texts)]. Reports only, never edits rows."""
    out: list[tuple[int, list[str]]] = []
    for row_idx in sorted(suspicious_tall_rows(cells)):
        if confirm_merged_row(cells, row_idx, words) <= WRAP_MAX_COLS:
            continue
        i = match_cell_row_to_output_row(cells, row_idx, rows)
        if i is None:
            continue
        texts = [c.text.strip() for c in sorted(cells, key=lambda c: c.c0) if c.r0 == row_idx and c.text.strip()]
        out.append((i, texts))
    return out


# ---------------------------------------------------------------------------
# confidence grading and the audit entry point
# ---------------------------------------------------------------------------


def audit(table: ExtractedTable, *, region_chars: int, ocr_page: bool) -> ExtractedTable:
    """Stamp coverage / notes / confidence onto one extraction result, in place."""
    notes = table.notes
    if table.candidate.source_type == "text":
        table.coverage = coverage_ratio(table.cell_chars, region_chars)
        if table.coverage is not None and table.coverage < COVERAGE_LOW:
            notes.append(f"coverage_low={table.coverage:.0%}")
    if table.plumber_cols is not None and table.n_cols:
        if abs(table.plumber_cols - table.n_cols) > 1:
            notes.append(f"col_mismatch(docling={table.n_cols},plumber={table.plumber_cols})")
    for hint in suspect_lt_as_minus(table.rows):
        notes.append(hint)
    table.confidence = decide_confidence(table, ocr_page=ocr_page)
    return table


def decide_confidence(table: ExtractedTable, *, ocr_page: bool) -> str:
    """Three grades, never a fourth.

    low = anything OCR-derived (structure can be right while characters are not);
    high = docling + normal coverage + plumber column agreement (+-1) + clean grid + no merged-row warning;
    medium = the rest.
    """
    if table.candidate.source_type == "image" or ocr_page:
        return "low"
    if table.grid_status != "ok":
        return "medium"
    if table.coverage is None or table.coverage < COVERAGE_LOW:
        return "medium"
    if table.plumber_cols is None:
        return "medium"
    if abs(table.plumber_cols - table.n_cols) > 1:
        return "medium"
    if any(n.startswith(MERGED_ROW_NOTE) for n in table.notes):
        return "medium"
    return "high"


# ---------------------------------------------------------------------------
# dual-read merge (figure panels): two fills on one geometry, per-cell flags
# ---------------------------------------------------------------------------


def _norm(c: str) -> str:
    return re.sub(r"\s+", "", str(c)).lower().replace("−", "-")


def format_signature(value: str) -> str:
    """Generic format class of a cell: digit runs -> 9, letter runs -> A, other chars kept."""
    s = re.sub(r"\d+", "9", value.strip())
    return re.sub(r"[A-Za-z]+", "A", s)


def _column_majority_signature(column_values: list[str]) -> str | None:
    """Majority format among non-conflicted values of a column; None without a clear majority."""
    sigs = [format_signature(v) for v in column_values if v.strip()]
    if len(sigs) < 3:
        return None
    best = max(set(sigs), key=sigs.count)
    return best if sigs.count(best) > len(sigs) / 2 else None


def merge_grids(conservative: list[list[str]], enhanced: list[list[str]]) -> tuple[list[list[str]], list[list[str]]]:
    """Merge the two fills of one geometry by shared row index; returns (rows, flags).

    Per cell: one side empty -> take the other; agreement -> take it; conflict ->
    the candidate matching the column's majority format wins (derived from the
    agreeing cells of that column, nothing hardcoded), otherwise keep the enhanced
    value flagged conflict_unresolved. Rows with <2 non-empty cells are pruned
    AFTER merging so a single-fill pruning can never lose rows.
    Flags: agree / b2_only / b3_only / conflict_format / conflict_unresolved.
    """
    n_rows = max(len(conservative), len(enhanced))
    width = max(
        max((len(r) for r in conservative), default=0),
        max((len(r) for r in enhanced), default=0),
    )

    def cell(g: list[list[str]], i: int, j: int) -> str:
        return g[i][j] if i < len(g) and j < len(g[i]) else ""

    agreed_by_col: dict[int, list[str]] = {}
    for i in range(n_rows):
        for j in range(width):
            a, b = cell(conservative, i, j), cell(enhanced, i, j)
            if a and b and _norm(a) == _norm(b):
                agreed_by_col.setdefault(j, []).append(b)

    merged: list[list[str]] = []
    flags: list[list[str]] = []
    for i in range(n_rows):
        row, frow = [], []
        for j in range(width):
            a, b = cell(conservative, i, j), cell(enhanced, i, j)
            if _norm(a) and _norm(b) and _norm(a) != _norm(b):
                maj = _column_majority_signature(agreed_by_col.get(j, []))
                fa, fb = format_signature(a) == maj, format_signature(b) == maj
                if maj and fa != fb:
                    row.append(a if fa else b)
                    frow.append("conflict_format")
                else:
                    row.append(b)
                    frow.append("conflict_unresolved")
            elif _norm(b):
                row.append(b)
                frow.append("agree" if _norm(a) else "b3_only")
            else:
                row.append(a)
                frow.append("b2_only" if _norm(a) else "")
        merged.append(row)
        flags.append(frow)

    keep = [i for i, r in enumerate(merged) if sum(1 for c in r if c.strip()) >= 2]
    return [merged[i] for i in keep], [flags[i] for i in keep]


def conflict_summary(flags: list[list[str]]) -> tuple[int, int]:
    """(total conflict cells, unresolved conflict cells) for the manifest."""
    total = sum(1 for fr in flags for x in fr if x.startswith("conflict"))
    unresolved = sum(1 for fr in flags for x in fr if x == "conflict_unresolved")
    return total, unresolved
