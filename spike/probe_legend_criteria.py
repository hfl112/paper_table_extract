"""探针：legend 续段判据换什么维度（F-013）。

    conda run -n gemini python spike/probe_legend_criteria.py

现行判据「下一块最长行 >= 40 字符」在 90 篇上**余量归零、两类错同时存在**：
    停住的最高 39 ← 阈值 40 → 吸收的最低 40
    该吸收却停住 5 处（`genotoxic stress or oncogenic signals.` 38 字）
    不该吸收却吸了 8 处（`Number of patients (n)` 41 字、`Test ID`、目录点引导线）
根因：**表头能很长、legend 续段能很短，长度这个维度分不开它们。**

本探针把所有紧贴阈值的决策点连**整块文本**一起抓出来，人工标好类别，
再拿它当测试集比几个候选维度。只读不写。
"""

from __future__ import annotations

import re
import statistics
from pathlib import Path

import fitz

from pdf_table_extract import rules

CORPORA = [
    ("dev", Path("0.pdf_input")),
    ("holdout", Path("/Users/funanhe/Downloads/Zotero_Precancer_PDFs")),
    ("thresh", Path("/Users/funanhe/Downloads/pte_threshold_corpus")),
]

# 点引导线：目录页的 `…………` / `. . . .`
DOT_LEADER = re.compile(r"[.…]{6,}|(?:\.\s){5,}")
SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s*$")


def blocks_after_caption(page, pat):
    """复刻 pdfio._absorb_legend_tail 的遍历，产出每个 (caption块, 候选下一块) 决策点。"""
    bl = [b for b in page.get_text("dict")["blocks"] if "lines" in b]
    bl.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
    for i, b in enumerate(bl):
        lines = ["".join(s["text"] for s in ln["spans"]).strip() for ln in b["lines"]]
        if not any(rules.match_label_at_line_start(x, pat) for x in lines):
            continue
        bottom, bx0, bx1 = b["bbox"][3], b["bbox"][0], b["bbox"][2]
        for nb in bl[i + 1:]:
            nx0, ny0, nx1, ny1 = nb["bbox"]
            if ny0 < bottom - 1 or min(bx1, nx1) - max(bx0, nx0) <= 0:
                continue
            nlines = ["".join(s["text"] for s in ln["spans"]).strip() for ln in nb["lines"]]
            if any(rules.match_label_at_line_start(t, pat) for t in nlines):
                break
            yield lines[0], nlines, ny0 - bottom
            if rules.is_legend_continuation(
                ny0 - bottom, max((len(t) for t in nlines), default=0)
            ):
                bottom = ny1
            else:
                break


def features(nlines: list[str]) -> dict:
    text = " ".join(nlines).strip()
    lens = [len(t) for t in nlines if t]
    return {
        "max_line": max(lens, default=0),
        "n_lines": len(lens),
        "mean_line": statistics.mean(lens) if lens else 0,
        "total": len(text),
        "sentence_end": bool(SENTENCE_END.search(text)),
        "has_period": "." in text,
        "dot_leader": bool(DOT_LEADER.search(text)),
        "text": text,
    }


def main() -> None:
    pat = rules.compile_label_pattern()
    rows = []
    for tag, d in CORPORA:
        for f in sorted(d.glob("*.pdf")):
            doc = fitz.open(f)
            for pno, page in enumerate(doc, 1):
                for cap, nlines, gap in blocks_after_caption(page, pat):
                    if gap >= rules.LEGEND_MAX_BLOCK_GAP:
                        continue                      # gap 已经挡住，不是行长在决定
                    fe = features(nlines)
                    if 36 <= fe["max_line"] <= 44:    # 紧贴阈值 40
                        rows.append((f"{tag}:{f.stem[:22]} p{pno}", round(gap, 1), cap, fe))
            doc.close()

    rows.sort(key=lambda r: r[3]["max_line"])
    print(f"紧贴阈值（gap<15 且 最长行 36-44）的决策点共 {len(rows)} 个")
    print()
    hdr = "%-30s %5s %5s %3s %6s %5s %4s %4s  %s"
    print(hdr % ("页", "gap", "最长", "行", "均长", "总字", "句末", "点引", "整块文本"))
    print("-" * 130)
    for name, gap, cap, fe in rows:
        verdict = "吸收" if fe["max_line"] >= rules.LEGEND_MIN_PROSE_LINE else "停住"
        print(hdr % (f"[{verdict}]{name}", gap, fe["max_line"], fe["n_lines"],
                     f"{fe['mean_line']:.0f}", fe["total"],
                     "是" if fe["sentence_end"] else "",
                     "是" if fe["dot_leader"] else "", fe["text"][:56]))

    print()
    print("=" * 130)
    print("候选维度在这批边界样本上的分布（人工标签见 eval/findings.md F-013）")
    print("=" * 130)
    absorb = [r for r in rows if r[3]["max_line"] >= 40]
    stop = [r for r in rows if r[3]["max_line"] < 40]
    for label, group in (("现行判据吸收的", absorb), ("现行判据停住的", stop)):
        if not group:
            continue
        print(f"\n  {label} {len(group)} 个：")
        for key, name in (("sentence_end", "整块以句末标点结尾"), ("dot_leader", "含点引导线"),):
            n = sum(1 for r in group if r[3][key])
            print(f"      {name:20s} {n}/{len(group)}")
        ml = [r[3]["mean_line"] for r in group]
        nl = [r[3]["n_lines"] for r in group]
        print(f"      平均行长              {min(ml):.0f} – {max(ml):.0f}")
        print(f"      行数                  {min(nl)} – {max(nl)}")


if __name__ == "__main__":
    main()
