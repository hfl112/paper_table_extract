"""Launcher so the tool runs as `python pdf_table_extract.py ...` from anywhere, without -m."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pdf_table_extract.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
