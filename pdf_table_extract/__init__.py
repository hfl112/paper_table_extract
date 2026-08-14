"""pdf-table-extract: given an English paper PDF (and optional keywords), export matching tables as CSV.

Generality principle: domain knowledge comes from the user (keyword files); the
tool itself relies only on typesetting conventions and structural signals.
Modes: list (label inventory), table (text-layer tables), figure (panel dual-read).
"""

__version__ = "0.2.0"
