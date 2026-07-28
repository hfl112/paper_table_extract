"""探针：转正判据能否改用「竖排文字里是否含 Table 标号」。

    conda run -n gemini python spike/probe_rotation_caption.py

背景：25 篇留出集跑 --list 时，现有判据（竖排行数 >= 30 且 占比 >= 50%，AGENTS.md 事实 2c）
命中 3 页。人工核对结果：
  - Bruhm p9   真横向表（竖排首行就是 `Table 3 | Clinically validated ctDNA tests...`）
  - Li p6      误报 —— 1555 行竖排、占比 85%，内容是 `Percentage of SBSs` / `ACA` 反复重复，
               那是多面板图的 Y 轴标签。**两条现有判据同时失效。**
  - Williams(cells of) p15  误报 —— 90 行、占比 67%，内容是 `% donors` / `1q` / `20q` / `Xp`
               等染色体臂标签。

试过但不成立的替代判据：
  - 竖排串唯一率：真 36–89% vs 误报 2%（Li）与 **50%（Williams，落进真表区间）** → 失手
  - 竖排串中位长：真 1–8 vs 误报 3 → 重叠
  - 长串（>=8 字）占比：真 3–50% vs 误报 1–7% → 重叠（pbc_29304 p4 只有 3%）

本探针验证的候选：竖排行数 >= 30 **且** 竖排文字里含 Table 标号（复用 AGENTS.md 事实 #2
那条行首正则）。语义上也站得住：横向表的 caption 本身就是跟着表一起转过来的。
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz

from pdf_table_extract import rules

HOLDOUT = Path("/Users/funanhe/Downloads/Zotero_Precancer_PDFs")
DEV = Path("0.pdf_input")

CAPTION = re.compile(
    r"^(?:Supplementary\s+)?(?:Table|TABLE|Tab\.)\s+(?:S?\d+|[IVXivx]{1,4})(?:[.:|\s]|$)"
)


def page_lines(page) -> tuple[list[str], int]:
    """返回 (竖排文字行, 水平行数)。方向判定见 AGENTS.md 事实 #3：靠 dir，不靠 page.rotation。"""
    vertical: list[str] = []
    horizontal = 0
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            text = "".join(s["text"] for s in line["spans"]).strip()
            if not text:
                continue
            if line["dir"] == (1, 0):
                horizontal += 1
            else:
                vertical.append(text)
    return vertical, horizontal


def main() -> None:
    pdfs = sorted(HOLDOUT.glob("*.pdf")) + sorted(DEV.glob("*.pdf"))
    current: list[tuple[str, int, int, float, bool]] = []
    candidate: list[tuple[str, int, int, float, bool]] = []

    for path in pdfs:
        doc = fitz.open(path)
        for pno, page in enumerate(doc, 1):
            vert, horiz = page_lines(page)
            if not vert:
                continue
            ratio = len(vert) / (len(vert) + horiz)
            has_caption = any(CAPTION.match(v) for v in vert)
            enough = len(vert) >= rules.VERTICAL_LINES_THRESHOLD
            row = (path.stem[:26], pno, len(vert), ratio, has_caption)
            if enough and ratio >= rules.VERTICAL_RATIO_THRESHOLD:
                current.append(row)
            if enough and has_caption:
                candidate.append(row)
        doc.close()

    print(f"语料：留出集 {len(list(HOLDOUT.glob('*.pdf')))} 篇 + 开发集 {len(list(DEV.glob('*.pdf')))} 篇")
    print()
    print(f"现有判据（竖排 >= {rules.VERTICAL_LINES_THRESHOLD} 行 且 占比 >= "
          f"{rules.VERTICAL_RATIO_THRESHOLD:.0%}）命中 {len(current)} 页：")
    for name, pno, nv, ratio, cap in current:
        verdict = "有 Table 标号" if cap else "无 Table 标号  <-- 误报"
        print(f"    {name:28s} p{pno:<3d} 竖排{nv:5d} 占比{ratio:4.0%}   {verdict}")

    print()
    print(f"候选判据（竖排 >= {rules.VERTICAL_LINES_THRESHOLD} 行 且 竖排含 Table 标号）"
          f"命中 {len(candidate)} 页：")
    for name, pno, nv, ratio, _ in candidate:
        print(f"    {name:28s} p{pno:<3d} 竖排{nv:5d} 占比{ratio:4.0%}")

    cur_set = {(n, p) for n, p, *_ in current}
    cand_set = {(n, p) for n, p, *_ in candidate}
    print()
    print(f"现有独有（候选会放过）：{sorted(cur_set - cand_set)}")
    print(f"候选独有（现有会放过）：{sorted(cand_set - cur_set)}")


if __name__ == "__main__":
    main()
