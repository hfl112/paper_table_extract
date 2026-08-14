# pdf-table-extract

Extract tables from English scientific paper PDFs as CSV — including tables typeset
as figure panels (heat maps and similar), which layout tools normally cannot read.

Domain knowledge is never built in: you supply keywords; the tool relies only on
typesetting conventions and structural signals.

## Modes

```bash
# 1. inventory: every table/figure label + caption. stdout IS the CSV, metadata on stderr
python pdf_table_extract.py paper.pdf --mode list > labels.csv

# 2. text-layer tables (docling + pdfplumber second opinion); no keywords = export all
python pdf_table_extract.py paper.pdf --mode table --keywords-file keywords/response.txt -o out --prefix paper1

# 3. table-like figure panels (cell detection + dual OCR reads, per-cell conflict flags)
python pdf_table_extract.py paper.pdf --mode figure -o out --prefix paper1
```

Options: `--mode {list,table,figure}`, `--keywords WORD ...`, `--keywords-file FILE`,
`-o/--outdir`, `--prefix`, `--dpi` (figure mode, default 300), `--dump-pages`
(save source-page PNGs for manual checking). Exit codes: 2 usage, 4 no labels found.

## Output

One folder per PDF: per-table CSVs, `manifest.csv`, and `*_flags.csv` for figure panels.

- The manifest never drops a table silently: every failure or skip gets a row with the
  reason (pure chart, low-resolution raster, supplementary file not in this PDF, ...).
- The manifest accumulates across runs; the `query` column records what each run searched.
- Figure panels are read twice (conservative + enhanced OCR on one grid geometry) and
  merged; per-cell flags mark `agree` / `conflict_*` cells so downstream verification
  knows exactly which values to distrust.
- Confidence is graded high/medium/low; anything OCR-derived is low by definition.

## What it handles

Sideways (rotated) table pages, tables continued across pages, multi-line legends,
scanned PDFs with publisher OCR text layers, preprint detection (warns, never refuses),
dual-panel figures (table + bar chart), and color-swatch columns (read by RGB sampling).
Low-resolution rasters (< ~10.5 native px per cell, measured) are skipped with a manifest
row — that content needs a vision/LLM pass instead.

## Install

Python >= 3.11 with `docling`, `paddleocr` + `paddlex[ocr]`, `pymupdf`, `pdfplumber`.
CPU is enough. torch.compile is disabled by the tool itself (slow for one-shot runs and
broken on old g++ toolchains).

## Keywords

`keywords/*.txt`: one word per line, `#` comments. Matching is whole-word plus English
inflections (`xenograft` also hits `xenografts`); hyphens count as word characters, so
`PR` does not match `PR-104`. Edit these files, not the code, to tune recall.

## Architecture

`common.py` (shared data structures + pre-extraction judgement) -> `pdfio.py` (all PyMuPDF
access) -> `engine_docling/paddle/plumber.py` (heavy lifting) -> `evaluation.py`
(reliability of the extracted CSVs) -> `csv_writer.py` (all output). `mode_*.py` files
orchestrate; every judgement call with a threshold carries its measured evidence in a
comment. `tests/` and `AGENTS.md` still describe the pre-refactor layout (known debt).
