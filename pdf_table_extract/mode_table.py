"""table mode: text-layer tables via docling, pdfplumber column second opinion.

Pipeline: normalize rotation -> scan labels -> docling once -> keep text-path
candidates -> hyphen rejoin, merged-row detection, caption-ownership check,
audit -> stitch continuations -> keyword filter (none = keep all) -> write.
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime
from pathlib import Path

from . import common, csv_writer, engine_docling, engine_plumber, evaluation, pdfio
from .common import ExtractedTable, ManifestRow, TableCandidate


def _spans_cache(pdf: Path):
    class _Lazy(dict):
        def __missing__(self, page_no: int):
            v = pdfio.page_spans(pdf, page_no)
            self[page_no] = v
            return v

    return _Lazy()


def echo_preprint(info) -> None:
    """Warn (do not refuse; batch-safe) and let the manifest carry the note."""
    if pdfio.is_preprint(info):
        detail = ", ".join(f"{k}x{v}" for k, v in sorted(info.preprint_markers.items()))
        print(
            f"warning: {Path(info.path).name} looks like a preprint ({detail}). "
            f"Its tables may differ from the published version - consider replacing the PDF.",
            file=sys.stderr,
        )


def _no_labels_exit(info) -> int:
    print(
        "error: no table/figure labels found in the whole document. Either the paper has no tables,\n"
        "       or its label convention is unusual. Run --mode list on similar papers to check.",
        file=sys.stderr,
    )
    return 4


def run(args, keywords: list[str]) -> int:
    info_orig = pdfio.read_doc(args.pdf)
    echo_preprint(info_orig)
    outdir = csv_writer.resolve_outdir(args.outdir, args.prefix, args.pdf)
    outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pte_") as tmp:
        norm_pdf = Path(tmp) / "normalized.pdf"
        rotated = pdfio.write_normalized(args.pdf, norm_pdf, info_orig)
        if rotated:
            print(f"rotated sideways-table pages upright: {rotated}")

        labels = pdfio.scan_labels(norm_pdf)
        caps = pdfio.caption_labels(labels)
        if not caps:
            return _no_labels_exit(info_orig)

        spans = _spans_cache(norm_pdf)
        # ocr_text_layer is rotation-invariant; info_orig already holds it, and re-reading
        # the normalized copy would recompute it in a mixed frame
        ocr_pages = {p.page for p in info_orig.pages if p.ocr_text_layer}

        print("running layout analysis and table reconstruction (docling)...")
        dl = engine_docling.convert(norm_pdf)

        rows_out: list[ManifestRow] = []
        cands: list[TableCandidate] = []
        # each candidate keeps a direct reference to its docling table: a (page, y0)
        # lookup can collide for side-by-side tables and silently swap their rows
        source_of: dict[int, object] = {}
        for seq, t in enumerate(dl.tables, 1):
            n = common.region_chars_of(t.rect, spans[t.page])
            if common.dispatch_source_type(n) != "text":
                rows_out.append(csv_writer.failure_row(
                    table_id_=f"p{t.page:02d}_t{seq}", label=None, page=t.page, source_type="image",
                    reason="pixel table region (no text layer inside): use --mode figure for this one",
                ))
                continue
            c = TableCandidate(page=t.page, rect=t.rect, source_type="text", chars_in_rect=n, origin="docling_table")
            source_of[id(c)] = t
            cands.append(c)

        # tables eat table captions first; leftovers may take figure captions (vector-drawn tables inside figures)
        common.attach_labels(cands, [l for l in caps if l.kind == "table"])
        common.attach_labels([c for c in cands if c.label is None], [l for l in caps if l.kind == "figure"])

        all_captions = [(l.kind, l.text) for l in caps]
        word_counts: dict[str, int] | None = None
        tables: list[ExtractedTable] = []
        for c in cands:
            dt = source_of[id(c)]
            if not dt.rows:
                rows_out.append(csv_writer.failure_row(
                    table_id_=csv_writer.table_id(c.label, c.page, 0), label=c.label, page=c.page,
                    reason="docling located the table but its row export came back empty; check the page",
                ))
                continue
            rows, n_join = dt.rows, 0
            if evaluation.has_hyphen_break(dt.rows):
                if word_counts is None:
                    word_counts = pdfio.doc_word_counts(norm_pdf)  # lazy: whole-document read
                rows, n_join = evaluation.rejoin_hyphen_breaks(dt.rows, word_counts)
            table = ExtractedTable(candidate=c, rows=rows, extractor="docling")
            if n_join:
                table.notes.append(f"hyphen_rejoined={n_join}")
            if dt.cells and evaluation.suspicious_tall_rows(dt.cells):
                merged = evaluation.detect_merged_rows(dt.cells, rows, pdfio.page_words(norm_pdf, c.page))
                for i, texts in merged:
                    table.notes.append(f"{evaluation.MERGED_ROW_NOTE}row{i}: " + " | ".join(t[:24] for t in texts[:4]))
            caption = c.label.text if c.label else dt.caption
            kind = c.label.kind if c.label else "table"
            others = [t for k, t in all_captions if k == kind and t != caption]
            if caption and not evaluation.verify_caption_belongs(caption, rows, other_captions=others):
                table.grid_status = "grid_mismatch"
                table.notes.append("caption words do not match header/first column; ownership doubtful")
            table.plumber_cols = engine_plumber.columns_in_bbox(norm_pdf, c.page, c.rect)
            evaluation.audit(table, region_chars=c.chars_in_rect, ocr_page=c.page in ocr_pages)
            if c.page in ocr_pages:
                table.notes.append("ocr_text_layer_page")
            tables.append(table)

        # stitch continuations ((Continued) captions or identical column names);
        # the gap check must use the LAST stitched page or 3+-page tables split after page 2
        tables.sort(key=lambda t: (t.candidate.page, t.candidate.rect.y0))
        stitched: list[ExtractedTable] = []
        tail_page: dict[int, int] = {}
        for t in tables:
            if stitched:
                prev = stitched[-1]
                why = common.is_continuation(prev, t, prev_page=tail_page[id(prev)])
                if why:
                    body = t.rows[1:] if common.same_column_names(prev.rows[0] if prev.rows else [], t.rows[0] if t.rows else []) else t.rows
                    prev.rows.extend(body)
                    prev.notes.append(f"spans_pages={prev.candidate.page}-{t.candidate.page}({why})")
                    tail_page[id(prev)] = t.candidate.page
                    continue
            stitched.append(t)
            tail_page[id(t)] = t.candidate.page

        kept: list[ExtractedTable] = []
        for t in stitched:
            if keywords:
                hits, where = common.match_table(t, keywords)
                if not hits:
                    continue
                t.matched_keywords = hits
                t.matched_on = where
            kept.append(t)

        for seq, t in enumerate(kept, 1):
            tid = csv_writer.table_id(t.candidate.label, t.candidate.page, seq)
            name = csv_writer.csv_name(tid, t.candidate.page)
            csv_writer.write_table(outdir, t, name)
            rows_out.append(csv_writer.to_manifest_row(t, table_id_=tid, csv_path=name))
            if args.dump_pages:
                pdfio.render_page(norm_pdf, t.candidate.page, outdir / f"page_{t.candidate.page:02d}.png", dpi=args.dpi)

        # a caption matched keywords but nothing was exported: leave a visible failure row
        exported_keys = {(t.candidate.label.key if t.candidate.label else None) for t in kept}
        if keywords:
            for lab in caps:
                if lab.key in exported_keys:
                    continue
                hits = common.keyword_hits(lab.text, keywords)
                if not hits:
                    continue
                rows_out.append(
                    csv_writer.failure_row(
                        table_id_=csv_writer.table_id(lab, lab.page, 0),
                        label=lab,
                        page=lab.page,
                        reason=(
                            f"caption/legend matched [{';'.join(hits)}] but no text-layer table was extracted there - "
                            f"it may be a figure panel (try --mode figure) or a pure chart; check the page"
                        ),
                    )
                )

        # labels referenced in the body but absent from this PDF (separate supplementary file)
        for lab in pdfio.referenced_only(labels):
            hits = common.cooccurs_in_sentence(lab.text, lab.raw, keywords) if keywords else []
            if hits or not keywords:
                rows_out.append(
                    csv_writer.failure_row(
                        table_id_=csv_writer.table_id(lab, lab.page, 0),
                        label=lab,
                        page=lab.page,
                        reason=(
                            "label_referenced_but_absent: the body cites it"
                            + (f" (same sentence as [{';'.join(hits)}])" if hits else "")
                            + " but it is not in this PDF; probably a separate supplementary file"
                        ),
                    )
                )

        if pdfio.is_preprint(info_orig):
            rows_out.append(csv_writer.failure_row(table_id_="preprint_check", label=None, page=1, reason="document looks like a preprint; tables may differ from the published version"))

        query = csv_writer.describe_query("table", keywords)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for r in rows_out:
            r.query = query
            r.run_at = stamp
        path = csv_writer.write_manifest(outdir, rows_out, query=query)
        csv_writer.print_summary(rows_out, outdir, manifest=path, query=query)
        if not kept and keywords:
            csv_writer.print_zero_hit_hint(caps, query)
    return 0
