"""CLI entry: argument parsing and mode dispatch only; no business logic.

Exit codes: 0 ok, 2 usage error, 4 no table/figure labels found (table/figure modes).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import common

common.silence()  # before any engine import

from . import mode_figure, mode_list, mode_table  # noqa: E402

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_NO_LABELS = 4


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdf_table_extract",
        description="Extract tables from an English paper PDF: list labels, extract text-layer tables, or read figure panels.",
    )
    p.add_argument("pdf", type=Path, help="input PDF")
    p.add_argument("--mode", required=True, choices=("list", "table", "figure"), help="list = inventory of labels/captions; table = text-layer tables; figure = table-like figure panels")
    p.add_argument("--keywords", nargs="*", default=[], metavar="WORD", help="keywords (OR); table/figure without keywords export everything")
    p.add_argument("--keywords-file", type=Path, action="append", default=[], metavar="FILE", help="keyword file, one per line, # comments; repeatable - files merge into one run (docling runs once)")
    p.add_argument("-o", "--outdir", type=Path, help="output root, default current dir; results land in <outdir>/<prefix>/")
    p.add_argument("--prefix", help="subfolder name for this paper, default the PDF stem")
    p.add_argument("--dpi", type=int, default=300, help="page render dpi for figure mode (300 measured best)")
    p.add_argument("--cpus", type=int, help="CPU threads for the model engines; default: Slurm allocation, else all cores")
    p.add_argument("--dump-pages", action="store_true", help="also save a PNG of each source page that yielded output, for manual checking")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.pdf.exists():
        print(f"error: file not found: {args.pdf}", file=sys.stderr)
        return EXIT_USAGE
    if args.cpus:
        # before any engine loads its model: torch reads it at import, paddle at construction
        os.environ["OMP_NUM_THREADS"] = str(args.cpus)
    keywords = list(args.keywords)
    for f in args.keywords_file:
        keywords += common.read_words(f)
    keywords = list(dict.fromkeys(keywords))  # dedupe, keep order
    run = {"list": mode_list.run, "table": mode_table.run, "figure": mode_figure.run}[args.mode]
    return run(args, keywords)


if __name__ == "__main__":
    raise SystemExit(main())
