"""Physical PDF access. The ONLY entry point to PyMuPDF; fitz never leaks past this file.

Duties: text layer, label scanning, page-structure stats, normalized (rotated) PDF,
region character counts, page rendering, preprint detection.
Zero judgement: criteria live in common.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf as fitz  # the old "import fitz" alias prints a deprecation line to stdout, polluting CSV output

from . import common
from .common import DocInfo, ImageInfo, Label, PageInfo, Rect, Word

# Preprint detection, two independent criteria (both measured):
# 1) watermark words on >=40% of pages (bioRxiv/medRxiv stamp every page: 84% vs 6% for
#    a published paper that merely cites preprints)
PREPRINT_MARKERS = ("bioRxiv", "medRxiv", "preprint", "peer review", "peer-review")
PREPRINT_MIN_PAGE_RATIO = 0.4
# 2) vertical "platform + id" watermark on page 1's left edge (arXiv stamps only page 1;
#    structural criterion measured on 90 papers, 4/4 arXiv caught, zero false hits)
P1_WATERMARK_MAX_X_RATIO = 0.08
P1_WATERMARK_RE = re.compile(
    r"(arXiv|bioRxiv|medRxiv|Research\s*Square|SSRN|ChemRxiv)\s*:?\s*\d{4}\.\d{4,5}", re.I
)

_VERT_DIRS = {(0, 1), (0, -1)}


def _p1_watermark(page) -> str:
    """Vertical preprint watermark on page 1's left edge, or empty string."""
    limit = page.rect.width * P1_WATERMARK_MAX_X_RATIO
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            if tuple(round(x) for x in line["dir"]) not in _VERT_DIRS:
                continue
            if line["bbox"][2] > limit:
                continue
            text = "".join(s["text"] for s in line["spans"]).strip()
            if P1_WATERMARK_RE.search(text):
                return text
    return ""


def _rect(bbox) -> Rect:
    return Rect(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


# ---------------------------------------------------------------------------
# page structure
# ---------------------------------------------------------------------------


def _line_dirs(page: fitz.Page, pat: re.Pattern[str] | None = None) -> tuple[int, int, tuple[int, int], bool]:
    """(horizontal lines, vertical lines, dominant vertical direction, vertical text has a Table label)."""
    pat = pat or common.compile_label_pattern()
    horiz = vert = 0
    dir_count: dict[tuple[int, int], int] = {}
    has_label = False
    # directions must be judged in the DISPLAY frame: raw-horizontal text under a
    # publisher /Rotate 90 displays sideways and must count as vertical
    m = page.rotation_matrix if page.rotation else None
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            dx, dy = line["dir"]
            if m is not None:
                dx, dy = m.a * dx + m.c * dy, m.b * dx + m.d * dy
            d = (round(dx), round(dy))
            if d == (1, 0):
                horiz += 1
            elif d in _VERT_DIRS:
                vert += 1
                dir_count[d] = dir_count.get(d, 0) + 1
                if not has_label:
                    text = "".join(s["text"] for s in line["spans"])
                    raw = common.match_label_at_line_start(text, pat)
                    if raw and common.label_kind(common.normalize_label(raw)) == "table":
                        has_label = True
    dominant = max(dir_count, key=lambda k: dir_count[k]) if dir_count else (0, -1)
    return horiz, vert, dominant, has_label


def _images(page: fitz.Page) -> list[ImageInfo]:
    """Embedded bitmaps in the DISPLAY frame (rotation matrix applied), consistent with page.rect."""
    out = []
    pw, ph = page.rect.width, page.rect.height
    mat = page.rotation_matrix if page.rotation else None
    for info in page.get_image_info(xrefs=True):
        fr = fitz.Rect(info["bbox"])
        if mat is not None:
            fr = fr * mat
        r = Rect(fr.x0, fr.y0, fr.x1, fr.y1)
        out.append(
            ImageInfo(
                xref=int(info.get("xref") or 0),
                rect=r,
                px_width=int(info["width"]),
                px_height=int(info["height"]),
                cover_w=(r.width / pw) if pw else 0.0,
                cover_h=(r.height / ph) if ph else 0.0,
            )
        )
    return out


def read_doc(path: str | Path) -> DocInfo:
    """Read the whole document's structural features in one pass. No ML."""
    path = str(path)
    doc = fitz.open(path)
    pages: list[PageInfo] = []
    marker_counts: dict[str, int] = {}
    preprint_pages = 0
    p1_watermark = ""
    label_pat = common.compile_label_pattern()
    prev_rotated = False
    try:
        for pno, page in enumerate(doc, 1):
            text = page.get_text()
            page_hit = False
            for m in PREPRINT_MARKERS:
                n = len(re.findall(re.escape(m), text, re.I))
                if n:
                    marker_counts[m] = marker_counts.get(m, 0) + n
                    page_hit = True
            if page_hit:
                preprint_pages += 1
            if pno == 1:
                p1_watermark = _p1_watermark(page)
            horiz, vert, dominant, has_label = _line_dirs(page, label_pat)
            imgs = _images(page)
            pages.append(
                PageInfo(
                    page=pno,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    n_chars=len(text.strip()),
                    vert_lines=vert,
                    horiz_lines=horiz,
                    images=imgs,
                    ocr_text_layer=False,  # second pass below: scanned-doc is a whole-document property
                    needs_rotation=common.rotation_for_page(
                        vert, dominant, horiz,
                        has_table_label=has_label, prev_page_rotated=prev_rotated,
                    ),
                )
            )
            prev_rotated = bool(pages[-1].needs_rotation)

        scanned = common.is_scanned_document(
            n_full_page_images=sum(1 for p in pages if common.has_full_page_image(p.images)),
            n_pages=len(pages),
            total_chars=sum(p.n_chars for p in pages),
        )
        for p in pages:
            p.ocr_text_layer = common.is_ocr_text_layer(p.images, doc_is_scanned=scanned)

        meta = {k: (v or "") for k, v in (doc.metadata or {}).items()}
    finally:
        doc.close()
    return DocInfo(
        path=path,
        n_pages=len(pages),
        pages=pages,
        metadata=meta,
        preprint_markers=marker_counts,
        preprint_pages=preprint_pages,
        p1_watermark=p1_watermark,
    )


def is_preprint(info: DocInfo) -> bool:
    """Preprint / peer-review copy? Two independent criteria, see the constants above."""
    if not info.n_pages:
        return False
    if info.p1_watermark:
        return True
    return info.preprint_pages / info.n_pages >= PREPRINT_MIN_PAGE_RATIO


# ---------------------------------------------------------------------------
# label scanning (criteria come from common; this file only fetches text)
# ---------------------------------------------------------------------------

LEGEND_MAX_CHARS = 3000  # safety valve; longest real legend measured 1186 chars


def scan_labels(path: str | Path, pattern: str | None = None) -> list[Label]:
    """Scan labels line by line. Line-start hit = real caption; other occurrences = body references.

    Legend runs from the hit line to the next caption line in the same block (blocks may
    hold two captions), then absorbs following prose blocks per the continuation criterion.
    No fixed line/char cap: real legends reach 23 lines / 1186 chars.
    """
    pat = common.compile_label_pattern(pattern)
    loose = common.compile_loose_pattern(pattern)
    doc = fitz.open(str(path))
    labels: list[Label] = []
    try:
        for pno, page in enumerate(doc, 1):
            mat = page.rotation_matrix if page.rotation else None
            raw_blocks = [b for b in page.get_text("dict")["blocks"] if "lines" in b]
            ordered = sorted(raw_blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
            for block in raw_blocks:
                lines = block["lines"]
                texts = [common.collapse_ws("".join(s["text"] for s in ln["spans"])) for ln in lines]
                cap_at: dict[int, str] = {}
                for i, line_text in enumerate(texts):
                    raw = common.match_label_at_line_start(line_text, pat)
                    if raw is None:
                        continue
                    # second check: a reference wrapped to line start is not a caption
                    if not common.caption_line_is_plausible(texts[i - 1] if i else None):
                        continue
                    cap_at[i] = raw
                caption_idx = set(cap_at)
                starts = sorted(cap_at)
                for n, i in enumerate(starts):
                    raw = cap_at[i]
                    end = starts[n + 1] if n + 1 < len(starts) else len(texts)
                    key = common.normalize_label(raw)
                    body = common.collapse_ws(" ".join(texts[i:end]))
                    if end >= len(texts):
                        body = _absorb_legend_tail(block, ordered, body, pat, mat)
                    body = body[:LEGEND_MAX_CHARS]
                    cap_r = fitz.Rect(lines[i]["bbox"])
                    if mat is not None:
                        cap_r = cap_r * mat  # caption rect must live in the rotated frame too
                    labels.append(
                        Label(
                            raw=common.collapse_ws(raw),
                            key=key,
                            kind=common.label_kind(key),
                            page=pno,
                            is_caption=True,
                            text=body,
                            rect=Rect(cap_r.x0, cap_r.y0, cap_r.x1, cap_r.y1),
                        )
                    )
                block_text = common.collapse_ws(" ".join(texts))
                cap_keys = {
                    common.normalize_label(r)
                    for i in caption_idx
                    for r in [common.match_label_at_line_start(texts[i], pat)]
                    if r
                }
                for raw in common.find_labels_anywhere(block_text, loose):
                    key = common.normalize_label(raw)
                    if key in cap_keys:
                        continue
                    if any(l.page == pno and l.key == key and l.is_caption for l in labels):
                        continue
                    labels.append(
                        Label(
                            raw=common.collapse_ws(raw),
                            key=key,
                            kind=common.label_kind(key),
                            page=pno,
                            is_caption=False,
                            text=block_text,
                            rect=None,
                        )
                    )
    finally:
        doc.close()
    return labels


def _absorb_legend_tail(block, ordered, body: str, pat, mat=None) -> str:
    """Absorb legend continuation blocks below the caption block.

    Non-overlapping side blocks (e.g. a vertical 'Author Manuscript' watermark) are
    skipped, not treated as the end; a block containing a caption line ends the legend.
    'Below' is judged in the DISPLAY frame (mat = rotation matrix) or rotated pages
    would never absorb their legend tails.
    """
    def disp(bbox):
        if mat is None:
            return bbox
        r = fitz.Rect(bbox) * mat
        return (r.x0, r.y0, r.x1, r.y1)

    bb = disp(block["bbox"])
    bottom = bb[3]
    bx0, bx1 = bb[0], bb[2]
    try:
        idx = ordered.index(block)
    except ValueError:
        return body
    for nxt in ordered[idx + 1 :]:
        nx0, ny0, nx1, ny1 = disp(nxt["bbox"])
        if ny0 < bottom - 1:
            continue
        if min(bx1, nx1) - max(bx0, nx0) <= 0:
            continue
        nlines = [common.collapse_ws("".join(sp["text"] for sp in ln["spans"])) for ln in nxt["lines"]]
        if any(common.match_label_at_line_start(t, pat) for t in nlines):
            break
        if not common.is_legend_continuation(ny0 - bottom, max((len(t) for t in nlines), default=0)):
            break
        body = common.collapse_ws(body + " " + " ".join(nlines))
        bottom = ny1
    return body


def caption_labels(labels: list[Label]) -> list[Label]:
    return [l for l in labels if l.is_caption]


def referenced_only(labels: list[Label]) -> list[Label]:
    """Labels that are referenced but have no caption: the entity is not in this PDF (separate supplementary file)."""
    cap_keys = {l.key for l in labels if l.is_caption}
    seen: set[str] = set()
    out = []
    for l in labels:
        if l.is_caption or l.key in cap_keys or l.key in seen:
            continue
        seen.add(l.key)
        out.append(l)
    return out


# ---------------------------------------------------------------------------
# text-layer access for criteria
# ---------------------------------------------------------------------------


def page_spans(path: str | Path, page_no: int) -> list[tuple[Rect, str]]:
    """(rect, text) of every text span on a page, for common.region_chars_of.

    The rotation matrix is mandatory on rotated pages: get_text stays in the
    unrotated frame while docling rects use the rotated frame (measured, same
    trap as page_words); without it region character counts are frame-mixed.
    """
    doc = fitz.open(str(path))
    try:
        page = doc[page_no - 1]
        mat = page.rotation_matrix if page.rotation else None
        out = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    r = fitz.Rect(span["bbox"])
                    if mat is not None:
                        r = r * mat
                    out.append((Rect(r.x0, r.y0, r.x1, r.y1), span["text"]))
        return out
    finally:
        doc.close()


def page_words(path: str | Path, page_no: int) -> list[Word]:
    """Every word with its rect, 1-based page. MUST apply the rotation matrix:
    get_text keeps unrotated coordinates on rotated pages while docling reports the
    rotated frame; without the transform, merged-row detection silently dies on
    every rotated table (measured)."""
    doc = fitz.open(str(path))
    try:
        page = doc[page_no - 1]
        mat = page.rotation_matrix if page.rotation else None
        out: list[Word] = []
        for x0, y0, x1, y1, text, *_ in page.get_text("words"):
            r = fitz.Rect(x0, y0, x1, y1)
            if mat is not None:
                r = r * mat
            out.append(Word(r.x0, r.y0, r.x1, r.y1, text))
        return out
    finally:
        doc.close()


_WORDISH = re.compile(r"[a-z]+")


def doc_word_counts(path: str | Path) -> dict[str, int]:
    """Whole-document counts of pure-letter words (lowercased), for hyphen-break rejoining.

    Document scope on purpose (split halves can sit on different pages); words are
    taken as [a-z]+ runs WITHOUT pre-joining hyphens (a pre-join once fabricated the
    very word it was supposed to verify); counts not sets (threshold is >=2).
    """
    doc = fitz.open(str(path))
    try:
        text = " ".join(doc[i].get_text() for i in range(len(doc)))
    finally:
        doc.close()
    counts: dict[str, int] = {}
    for w in _WORDISH.findall(text.lower()):
        counts[w] = counts.get(w, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# normalized PDF (rotate sideways-table pages upright)
# ---------------------------------------------------------------------------


def write_normalized(src: str | Path, dst: str | Path, info: DocInfo) -> list[int]:
    """Copy the document, rotating only pages with needs_rotation. Returns rotated page numbers.

    Measured: docling output on the normalized whole document is identical to
    per-page rotation, so one docling run suffices.
    """
    doc = fitz.open(str(src))
    rotated: list[int] = []
    try:
        for p in info.pages:
            if p.needs_rotation:
                page = doc[p.page - 1]
                # compose: needs_rotation is ADDITIONAL display rotation on top of any /Rotate
                page.set_rotation((page.rotation + p.needs_rotation) % 360)
                rotated.append(p.page)
        doc.save(str(dst))
    finally:
        doc.close()
    return rotated


# ---------------------------------------------------------------------------
# images and rendering
# ---------------------------------------------------------------------------


def render_page(path: str | Path, page_no: int, dst: str | Path, dpi: int = 300) -> tuple[int, int]:
    """Render a whole page to PNG (figure-mode input and --dump-pages output)."""
    doc = fitz.open(str(path))
    try:
        pix = doc[page_no - 1].get_pixmap(dpi=dpi)
        pix.save(str(dst))
        return pix.width, pix.height
    finally:
        doc.close()
