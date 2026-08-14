"""figure mode: table-like figure panels via cell detection + dual OCR reads.

Per page with a figure caption: render -> detect cells -> gate (>=20 cells, else
pure chart) -> raster legibility gate (native px per cell) -> panel split /
dedupe / giants -> row anchors -> two fills (conservative and enhanced) ->
merge by row index with per-cell flags -> write CSV + flags.
Keyword tiers: caption/legend hit -> extract; else coarse OCR of the page decides.
Known limitation: one grid per page, attributed to the first figure caption;
pages with several figures get a manifest note.
"""

from __future__ import annotations

import statistics
import tempfile
from datetime import datetime
from pathlib import Path

from . import common, csv_writer, engine_paddle, evaluation, pdfio
from .common import ExtractedTable, Label, ManifestRow, TableCandidate, Rect


def _dominant_raster(page_info) -> object | None:
    """The largest content-sized embedded bitmap on the page, or None (vector figure)."""
    imgs = [im for im in page_info.images if not common.is_decorative_image(im.cover_h)]
    return max(imgs, key=lambda im: im.rect.width * im.rect.height, default=None)


def _dual_read(page_png: Path, boxes, scratch: Path):
    """Two fills on one geometry; returns (merged rows, flags, geometry stats)."""
    rows, cols, pitch = common.panel_geometry(boxes)
    med_w = statistics.median(x2 - x1 for x1, _, x2, _ in boxes)
    im = engine_paddle.open_image(page_png)
    placed = []
    for box in boxes:
        pos = common.snap_to_grid(rows, cols, pitch, med_w * 0.6, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        if pos is not None:
            placed.append((pos, box))
    swatch_probe = {pos: engine_paddle.swatch_color(im, box) for pos, box in placed}
    col_n = [0] * len(cols)
    col_sw = [0] * len(cols)
    for (i, j), _ in placed:
        col_n[j] += 1
        col_sw[j] += bool(swatch_probe[(i, j)])
    swatch_cols = {j for j in range(len(cols)) if col_n[j] and col_sw[j] / col_n[j] >= 0.5}

    def fill(enhanced: bool) -> list[list[str]]:
        grid = [["" for _ in cols] for _ in rows]
        for (i, j), box in placed:
            if enhanced and j in swatch_cols and swatch_probe[(i, j)]:
                text = swatch_probe[(i, j)]
            else:
                text = engine_paddle.ocr_cell(im, box, scratch, pad_l=6 if enhanced else 2, pad_r=2, pad_y=2, upscale=enhanced)
            if text and text not in grid[i][j]:
                grid[i][j] = (grid[i][j] + " " + text).strip()
        return grid

    merged, flags = evaluation.merge_grids(fill(False), fill(True))
    return merged, flags, (len(rows), len(cols))


def run(args, keywords: list[str]) -> int:
    from .mode_table import echo_preprint, _no_labels_exit

    info = pdfio.read_doc(args.pdf)
    echo_preprint(info)
    outdir = csv_writer.resolve_outdir(args.outdir, args.prefix, args.pdf)
    outdir.mkdir(parents=True, exist_ok=True)

    rows_out: list[ManifestRow] = []
    with tempfile.TemporaryDirectory(prefix="pte_fig_") as tmp:
        tmpdir = Path(tmp)
        # normalize like table mode: sideways pixel tables get handed over to this mode,
        # so the render must be upright or the cell gate misreads them as pure charts
        norm_pdf = tmpdir / "normalized.pdf"
        rotated_pages = set(pdfio.write_normalized(args.pdf, norm_pdf, info))
        if rotated_pages:
            print(f"rotated sideways-table pages upright: {sorted(rotated_pages)}")

        labels = pdfio.scan_labels(norm_pdf)
        caps = pdfio.caption_labels(labels)
        if not caps:
            return _no_labels_exit(info)
        fig_caps = [l for l in caps if l.kind == "figure"]
        pages = sorted({l.page for l in fig_caps})
        for pno in pages:
            page_caps = [l for l in fig_caps if l.page == pno]
            lab = page_caps[0]
            tid = csv_writer.table_id(lab, pno, 0)
            page_info = info.pages[pno - 1]

            # keyword tier 1: caption/legend
            cap_hits = common.keyword_hits(" ".join(l.text for l in page_caps), keywords) if keywords else []
            raster = _dominant_raster(page_info)

            png = tmpdir / f"p{pno:02d}.png"
            pdfio.render_page(norm_pdf, pno, png, dpi=args.dpi)
            raw = engine_paddle.detect_cells(png)
            if len(raw) < common.CELL_GATE_MIN:
                rows_out.append(csv_writer.failure_row(
                    table_id_=tid, label=lab, page=pno, source_type="image",
                    reason=f"no table grid in the image ({len(raw)} cells < {common.CELL_GATE_MIN}), judged a pure chart, skipped",
                ))
                continue

            # raster legibility gate: dpi alone mispredicts (two 120-dpi figures split
            # 100% vs garbage); NATIVE pixels per detected cell is what decides OCR viability
            if raster is not None and raster.rect.height > 0:
                med_h = statistics.median(b[3] - b[1] for b, _ in raw)
                # if WE rotated this page, the render's vertical axis maps to the raster's width
                rect_h = raster.rect.width if pno in rotated_pages else raster.rect.height
                native_px = raster.px_width if pno in rotated_pages else raster.px_height
                native_cell_h = med_h * native_px / (rect_h * args.dpi / 72)
                if native_cell_h < common.NATIVE_CELL_MIN_PX:
                    rows_out.append(csv_writer.failure_row(
                        table_id_=tid, label=lab, page=pno, source_type="image",
                        reason=f"low_resolution_llm_only: ~{native_cell_h:.1f} native px per cell (<{common.NATIVE_CELL_MIN_PX}); the image itself is the information ceiling, read it with an LLM/vision pass",
                    ))
                    continue

            # keyword tier 2: coarse OCR of the page text
            matched_on = "caption" if cap_hits else ""
            img_hits: list[str] = []
            if keywords and not cap_hits:
                blob = engine_paddle.plain_text(png)
                img_hits = common.keyword_hits(blob, keywords)
                if not img_hits:
                    rows_out.append(csv_writer.failure_row(
                        table_id_=tid, label=lab, page=pno, source_type="image",
                        reason="grid present but neither caption nor in-image text matched the keywords, skipped",
                    ))
                    continue
                matched_on = "image_text"

            boxes = common.iou_dedupe(raw)
            boxes, cut_dropped = common.split_panel(boxes)
            boxes = common.drop_giant_boxes(boxes)
            merged, flags, (n_rows, n_cols) = _dual_read(png, boxes, tmpdir / "scratch_cell.png")
            if not merged:
                rows_out.append(csv_writer.failure_row(
                    table_id_=tid, label=lab, page=pno, source_type="image",
                    reason=f"{len(raw)} cells detected but the merged grid came out empty",
                ))
                continue

            cand = TableCandidate(page=pno, rect=Rect(0, 0, page_info.width, page_info.height), source_type="image", label=lab, origin="page_render")
            table = ExtractedTable(candidate=cand, rows=merged, extractor="dual_ocr")
            table.matched_keywords = cap_hits or img_hits
            table.matched_on = matched_on
            n_conf, n_unres = evaluation.conflict_summary(flags)
            table.notes.append(f"conflicts={n_conf}(unresolved={n_unres})")
            if len(page_caps) > 1:
                table.notes.append(f"page has {len(page_caps)} figure captions; single grid attributed to {lab.raw}, check manually")
            if cut_dropped:
                table.notes.append(f"panel_split_dropped={cut_dropped} boxes (side panel, e.g. a bar chart)")
            mismatch = evaluation.grid_shape_mismatch(n_rows, n_cols, table.n_rows, table.n_cols)
            if mismatch:
                table.notes.append(mismatch)
                table.grid_status = "shape_disagree"
            evaluation.audit(table, region_chars=0, ocr_page=True)  # OCR-derived: low confidence by definition

            name = csv_writer.csv_name(tid, pno)
            csv_writer.write_table(outdir, table, name)
            csv_writer.write_flags(outdir, name.replace(".csv", "_flags.csv"), flags)
            rows_out.append(csv_writer.to_manifest_row(table, table_id_=tid, csv_path=name))
            if args.dump_pages:
                pdfio.render_page(norm_pdf, pno, outdir / f"page_{pno:02d}.png", dpi=args.dpi)

        for lab2 in pdfio.referenced_only(labels):
            hits = common.cooccurs_in_sentence(lab2.text, lab2.raw, keywords) if keywords else []
            if hits or not keywords:
                rows_out.append(csv_writer.failure_row(
                    table_id_=csv_writer.table_id(lab2, lab2.page, 0), label=lab2, page=lab2.page,
                    reason=(
                        "label_referenced_but_absent: the body cites it"
                        + (f" (same sentence as [{';'.join(hits)}])" if hits else "")
                        + " but it is not in this PDF; probably a separate supplementary file"
                    ),
                ))

    query = csv_writer.describe_query("figure", keywords)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for r in rows_out:
        r.query = query
        r.run_at = stamp
    path = csv_writer.write_manifest(outdir, rows_out, query=query)
    csv_writer.print_summary(rows_out, outdir, manifest=path, query=query)
    return 0
