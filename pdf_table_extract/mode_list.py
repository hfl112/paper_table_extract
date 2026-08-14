"""list mode: inventory every table/figure label with its caption/legend.

Machine-first output: stdout IS the CSV (same columns as captions.csv), so
`--mode list > labels.csv` and pipes just work; human-facing metadata (page
count, preprint warning, rotation notes) goes to stderr. Runs with or without
keywords; with keywords the matched_keywords column shows per-label hits.
Nearly free: no layout analysis, no ML.
"""

from __future__ import annotations

import csv
import sys

from . import common, csv_writer, pdfio


def run(args, keywords: list[str]) -> int:
    info = pdfio.read_doc(args.pdf)
    labels = pdfio.scan_labels(args.pdf)
    caps = pdfio.caption_labels(labels)
    refs = pdfio.referenced_only(labels)

    # keyed by id(label): same-key captions ((Continued) pages) must not share one entry
    hits: dict[int, list[str]] = {}
    if keywords:
        for l in caps + refs:
            h = common.keyword_hits(l.text, keywords)
            if h:
                hits[id(l)] = h

    err = sys.stderr
    print(f"file: {args.pdf}  pages: {info.n_pages}  text-layer chars: {info.total_chars}", file=err)
    if pdfio.is_preprint(info):
        detail = ", ".join(f"{k}x{v}" for k, v in sorted(info.preprint_markers.items()))
        print(f"warning: this looks like a preprint ({detail}) - consider replacing the PDF with the published version", file=err)
    ocr_pages = sorted(p.page for p in info.pages if p.ocr_text_layer)
    if ocr_pages:
        print(f"note: OCR text-layer pages (tables there are unconditionally low confidence): {ocr_pages}", file=err)
    rot = [(p.page, p.needs_rotation) for p in info.pages if p.needs_rotation]
    if rot:
        print(f"note: sideways-table pages (auto-rotated during extraction): {', '.join(f'p{p}->{r}deg' for p, r in rot)}", file=err)
    if refs:
        print(f"note: {len(refs)} label(s) referenced but not in this PDF (present_in_pdf=no rows below) - separate supplementary files", file=err)

    # single source with captions.csv: same columns, same rows (csv_writer.caption_rows)
    w = csv.writer(sys.stdout)
    w.writerow(csv_writer.CAPTION_COLUMNS)
    w.writerows(csv_writer.caption_rows(caps, refs, hits))

    if args.outdir or args.prefix:
        outdir = csv_writer.resolve_outdir(args.outdir, args.prefix, args.pdf)
        outdir.mkdir(parents=True, exist_ok=True)
        path = csv_writer.write_captions(outdir, caps, refs, hits)
        print(f"also written to: {path}", file=err)
    return 0
