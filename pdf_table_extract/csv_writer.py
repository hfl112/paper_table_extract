"""All disk output. The ONLY entry point for CSV/manifest writing.

Invariant: no table is dropped silently — every failure or skip gets a manifest
row with the reason.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from .common import ExtractedTable, Label, ManifestRow

_SAFE = re.compile(r"[^a-z0-9]+")
_UNSAFE_NAME = re.compile(r"[^\w.\-]+", re.UNICODE)


def sanitize_name(name: str) -> str:
    """Make a PDF stem usable as a directory name (spaces, trailing dots)."""
    s = _UNSAFE_NAME.sub("_", name.strip())
    s = re.sub(r"_{2,}", "_", s).strip("._-")
    return s or "output"


def resolve_outdir(outdir: Path | None, prefix: str | None, pdf: Path) -> Path:
    """One folder per PDF: <outdir>/<prefix>/; outdir defaults to cwd, prefix to the PDF stem."""
    root = outdir if outdir is not None else Path(".")
    return root / sanitize_name(prefix if prefix else pdf.stem)


def table_id(label: Label | None, page: int, seq: int) -> str:
    """Identifier used in CSV names: the label when there is one, else page+sequence."""
    if label is not None:
        return _SAFE.sub("_", label.key).strip("_")
    return f"p{page:02d}_t{seq}"


def csv_name(table_id_: str, page: int) -> str:
    return f"p{page:02d}_{table_id_}.csv"


def write_table(outdir: Path, table: ExtractedTable, name: str) -> str:
    path = outdir / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(table.rows)
    return name


def write_flags(outdir: Path, name: str, flags: list[list[str]]) -> str:
    """Per-cell provenance flags of a dual-read table, next to its CSV."""
    path = outdir / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(flags)
    return name


def to_manifest_row(table: ExtractedTable, *, table_id_: str, csv_path: str, notes_extra: list[str] | None = None) -> ManifestRow:
    lab = table.candidate.label
    notes = list(table.notes) + list(notes_extra or [])
    return ManifestRow(
        table_id=table_id_,
        label=lab.raw if lab else "",
        page=table.candidate.page,
        caption=(lab.text[:200] if lab else ""),
        csv_path=csv_path,
        extractor=table.extractor,
        source_type=table.candidate.source_type,
        matched_on=table.matched_on,
        matched_keywords=";".join(table.matched_keywords),
        n_rows=table.n_rows,
        n_cols=table.n_cols,
        coverage=("" if table.coverage is None else f"{table.coverage:.2f}"),
        plumber_cols=("" if table.plumber_cols is None else str(table.plumber_cols)),
        confidence=table.confidence,
        grid_status=table.grid_status,
        notes="; ".join(notes),
    )


def failure_row(*, table_id_: str, label: Label | None, page: int, reason: str, source_type: str = "") -> ManifestRow:
    """Failed or skipped tables get a manifest row too (the no-silent-drop invariant)."""
    return ManifestRow(
        table_id=table_id_,
        label=label.raw if label else "",
        page=page,
        caption=(label.text[:200] if label else ""),
        csv_path="",
        extractor="",
        source_type=source_type,
        matched_on="",
        matched_keywords="",
        n_rows=0,
        n_cols=0,
        coverage="",
        plumber_cols="",
        confidence="",
        grid_status="",
        notes=reason,
    )


def describe_query(mode: str, keywords: list[str]) -> str:
    """Stable string for this run's search condition; manifest query column and dedup key."""
    joined = ";".join(sorted(keywords)) if keywords else "all"
    return f"{mode}:{joined}"


def write_manifest(outdir: Path, rows: list[ManifestRow], *, query: str) -> Path:
    """Accumulating manifest: keep rows from other queries, replace rows of this exact query.

    Rerunning the same paper with different keywords appends history instead of
    overwriting it, so nobody needs artificial --prefix values just to keep records.
    """
    path = outdir / "manifest.csv"
    kept: list[list[str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            for old in csv.DictReader(fh):
                if old.get("query", "") == query:
                    continue
                kept.append([old.get(c, "") for c in ManifestRow.COLUMNS])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(ManifestRow.COLUMNS)
        w.writerows(kept)
        for r in rows:
            w.writerow([getattr(r, c) for c in ManifestRow.COLUMNS])
    return path


def print_summary(rows: list[ManifestRow], outdir: Path, *, manifest: Path | None = None, query: str = "") -> None:
    """Human-readable summary; low confidence and red flags must be visible."""
    ok = [r for r in rows if r.csv_path]
    bad = [r for r in rows if not r.csv_path]
    print(f"\noutput dir: {outdir}")
    if query:
        print(f"query: {query}")
    print(f"exported {len(ok)} table(s)" + (f", {len(bad)} not exported (see manifest)" if bad else ""))
    if ok:
        print(f"\n{'table':22s} {'pg':>3s} {'shape':>9s} {'conf':7s} {'source':6s} {'matched':11s} notes")
        print("-" * 104)
        for r in sorted(ok, key=lambda x: x.page):
            shape = f"{r.n_rows}x{r.n_cols}"
            print(
                f"{r.table_id[:22]:22s} {r.page:3d} {shape:>9s} {r.confidence:7s} "
                f"{r.source_type:6s} {r.matched_on:11s} {r.notes[:34]}"
            )
    # chart-gated skips come in batches (multi-panel figures); aggregate unnamed ones
    gated = [r for r in bad if "no table grid" in r.notes]
    gated_named = [r for r in gated if r.label]
    gated_anon = [r for r in gated if not r.label]
    for r in bad:
        if r in gated:
            continue
        print(f"  [not exported] {r.label or r.table_id} p{r.page}: {r.notes}")
    for r in gated_named:
        print(f"  [not exported] {r.label} p{r.page}: no table grid in the image, judged a pure chart")
    if gated_anon:
        pages = sorted({r.page for r in gated_anon})
        print(
            f"  [not exported] {len(gated_anon)} unnamed bitmap(s) without a table grid "
            f"(p{','.join(map(str, pages))}), judged pure charts - details in the manifest"
        )
    low = [r for r in ok if r.confidence == "low"]
    if low:
        print(f"\nnote: {len(low)} table(s) are low confidence (OCR-derived); check them manually, --dump-pages exports the source pages")
    if manifest is not None:
        print(f"\nmanifest: {manifest}")
        print("  (accumulating: rerunning with other keywords appends rows; the query column records each run)")


def print_zero_hit_hint(labels: list[Label], query: str) -> None:
    """On zero hits, list what the paper actually has: usually the vocabulary differs, not the content missing."""
    tabs = [l for l in labels if l.kind == "table"]
    figs = [l for l in labels if l.kind == "figure"]
    if not tabs and not figs:
        return
    print(
        f"\nnote: this paper has {len(tabs)} table(s) and {len(figs)} figure(s); none matched [{query}] - "
        f"likely different wording, not missing content:"
    )
    for l in sorted(tabs + figs, key=lambda x: (x.kind != "table", x.page)):
        head = l.text[:74] if l.text.strip() != l.raw else f"{l.raw} (caption has no description)"
        print(f"    {l.raw:22s} p{l.page:<3} {head}")
    print("  next: run --mode list to see full captions, or put synonyms into a --keywords-file")


CAPTION_COLUMNS = ("label", "kind", "page", "present_in_pdf", "n_chars", "matched_keywords", "caption_legend")


def caption_rows(caps: list[Label], refs: list[Label], hits: dict[int, list[str]] | None = None):
    """Rows of the list-mode inventory (single source for captions.csv AND stdout).

    hits maps id(label) -> matched keywords: same-key captions ((Continued) pages)
    must not share one entry.
    """
    for l in sorted(caps, key=lambda x: (x.page, x.key)):
        yield [l.raw, l.kind, l.page, "yes", len(l.text), ";".join((hits or {}).get(id(l), [])), l.text]
    for l in sorted(refs, key=lambda x: x.key):
        yield [l.raw, l.kind, l.page, "no", 0, ";".join((hits or {}).get(id(l), [])), ""]


def write_captions(outdir: Path, caps: list[Label], refs: list[Label], hits: dict[int, list[str]] | None = None) -> Path:
    """Machine-readable output of list mode; one row per label with the full caption/legend."""
    path = outdir / "captions.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(CAPTION_COLUMNS)
        w.writerows(caption_rows(caps, refs, hits))
    return path
