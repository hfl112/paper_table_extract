"""把 1.output/ 下所有 manifest + captions 汇总成一张清单。

    conda run -n gemini python eval/report.py matrix 1.output

五列：文章 | 本篇有几张表/几张图 | response 命中数(图号表号) | xenograft 命中数(图号表号)
     | 只被引用、实体可能在补充材料里的标号
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

QUERIES = ("response", "PDX;xenograft", "Demographic")


def _dedupe(labels: list[str]) -> list[str]:
    """同一标号可能出现两次（跨页续表，如 TABLE I 在 p4 与 p5），算一张。"""
    seen: dict[str, str] = {}
    for l in labels:
        seen.setdefault(l.strip().lower(), l.strip())
    return list(seen.values())


def collect(d: Path) -> dict:
    caps_path = d / "captions.csv"
    present_t: list[str] = []
    present_f: list[str] = []
    absent: list[str] = []
    if caps_path.exists():
        for r in csv.DictReader(caps_path.open(encoding="utf-8")):
            if r["present_in_pdf"] == "yes":
                (present_t if r["kind"] == "table" else present_f).append(r["label"])
            else:
                absent.append(r["label"])

    hits: dict[str, list[str]] = {q: [] for q in QUERIES}
    man = d / "manifest.csv"
    if man.exists():
        for r in csv.DictReader(man.open(encoding="utf-8")):
            if r["query"] in hits and r["csv_path"]:
                hits[r["query"]].append(f"{r['label'] or r['table_id']}({r['matched_on']})")

    return {
        "name": d.name,
        "tables": _dedupe(present_t),
        "figures": _dedupe(present_f),
        "absent": _dedupe(absent),
        "hits": {q: _dedupe(v) for q, v in hits.items()},
    }


def main(root: Path) -> None:
    rows = [collect(d) for d in sorted(root.iterdir()) if d.is_dir()]

    for r in rows:
        print("=" * 108)
        print(f"{r['name']}")
        print(f"  本篇实体: 表 {len(r['tables'])} 张 {r['tables']}")
        print(f"            图 {len(r['figures'])} 张 {r['figures']}")
        for q in QUERIES:
            h = r["hits"][q]
            print(f"  -k {q:<10} 命中 {len(h)} 张  {h if h else '—'}")
        if r["absent"]:
            print(f"  只被引用、实体可能在补充材料里: {len(r['absent'])} 个 {r['absent']}")
    print("=" * 108)
    print("括号里是命中处：caption=靠标题/图注，cell=靠表内内容，image_text=靠图内 OCR")
    print("同一标号跨页续表（如 TABLE I 在 p4 与 p5）算一张。")


if __name__ == "__main__":
    # 取**最后一个**参数，不是 argv[1]。
    # 本模块 docstring 与 `eval/matrix.sh:28` 都按 `report.py matrix 1.output` 调用，
    # 原来写死 argv[1] 会取到 `matrix`，然后 `Path("matrix").iterdir()` 抛
    # FileNotFoundError —— **照文档跑必崩**，而且一直没人发现。
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(Path(args[-1] if args else "1.output"))
