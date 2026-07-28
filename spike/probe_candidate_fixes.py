"""逐项离线验证候选修法：**能不能修好留出集的问题** + **会不会动开发集的结果**。

    conda run -n gemini python spike/probe_candidate_fixes.py

**不改包代码。** 每个候选都在本文件里独立实现一遍，跟现行实现并排跑，输出两栏：

    修复效果  —— 留出集上那个问题解决了吗
    回归影响  —— 开发集 12 篇的结果有没有变（**必须为 0，除非是有意改基线**）

候选来自 `eval/findings.md`：
  A. F-009  标号正则加期刊变体（Extended Data / Supplemental / eTable / Additional file）
  B. F-002  转正判据加第三条（竖排文字里含 Table 标号）
  C. F-007  caption 为空的表不得为 high
  D. F-008  过滤 docling 漏出来的字形名（/emspaceMean）
"""

from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

import fitz

from pdf_table_extract import pdfio, rules

DEV_PDF = Path("0.pdf_input")
HOLD_PDF = Path("/Users/funanhe/Downloads/Zotero_Precancer_PDFs")
DEV_OUT = Path("1.output")
HOLD_OUT = Path("holdout")

BAR = "=" * 100


def head(title: str) -> None:
    print()
    print(BAR)
    print(title)
    print(BAR)


# ═════════════════════ A. F-009 标号正则加变体 ═════════════════════

NEW_LABEL_CORE = (
    r"(?:Supplementary\s+|Supplemental\s+|Supplementary\s+Data\s+|Extended\s+Data\s+|Online\s+)?"
    r"(?:Tables?|TABLES?|Tabs?\.|Figures?|FIGURES?|Figs?\.?)\s+"
    r"(?:S?\d+|[IVXivx]{1,4})"
)


def probe_a() -> None:
    head("A. F-009 标号正则加期刊变体")

    print("① 修复效果：现在认不认得那些写法")
    old_p, new_p = rules.compile_label_pattern(), rules.compile_label_pattern(NEW_LABEL_CORE)
    cases = [
        "Table 1. Foo", "TABLE I. Foo", "Supplementary Table S2", "Tab. 3 Foo", "Table S1",
        "Extended Data Table 1", "Extended Data Fig. 3", "Supplemental Table 1",
        "Supplementary Data Table 2", "eTable 1", "Additional file 1: Table S1",
    ]
    for s in cases:
        o, n = bool(old_p.match(s)), bool(new_p.match(s))
        mark = "  同" if o == n else ("  **修好**" if n else "  ✗变差")
        print(f"    {'✓' if o else '✗'} → {'✓' if n else '✗'}  {s!r}{mark}")

    print()
    print("② 回归影响：开发集 12 篇的标号集合有没有变")
    changed = 0
    for f in sorted(DEV_PDF.glob("*.pdf")):
        old = pdfio.scan_labels(f)
        new = pdfio.scan_labels(f, NEW_LABEL_CORE)
        ok = {l.key for l in old if l.is_caption}
        nk = {l.key for l in new if l.is_caption}
        ro = {l.key for l in pdfio.referenced_only(old)}
        rn = {l.key for l in pdfio.referenced_only(new)}
        if ok != nk or ro != rn:
            changed += 1
            print(f"    ⚠ {f.stem[:34]:34s} caption +{sorted(nk-ok)} -{sorted(ok-nk)}"
                  f"   仅引用 +{sorted(rn-ro)} -{sorted(ro-rn)}")
    print(f"    ⇒ 开发集 12 篇里 {changed} 篇结果有变" + ("（**零回归**）" if changed == 0 else ""))

    print()
    print("③ 留出集增量：新认出来多少标号")
    add_cap = add_ref = 0
    papers = set()
    for f in sorted(HOLD_PDF.glob("*.pdf")):
        old, new = pdfio.scan_labels(f), pdfio.scan_labels(f, NEW_LABEL_CORE)
        oc = {l.key for l in old if l.is_caption}
        nc = {l.key for l in new if l.is_caption}
        ro = {l.key for l in pdfio.referenced_only(old)}
        rn = {l.key for l in pdfio.referenced_only(new)}
        if nc - oc or rn - ro:
            papers.add(f.stem[:26])
            add_cap += len(nc - oc)
            add_ref += len(rn - ro)
            if nc - oc:
                print(f"    + {f.stem[:34]:34s} 新增真 caption {sorted(nc-oc)[:5]}")
    print(f"    ⇒ 留出集新增真 caption {add_cap} 个、仅引用标号 {add_ref} 个，涉及 {len(papers)} 篇")
    print("    ⚠ 仅引用标号变多 = `entity_not_in_pdf` 提示会变吵，这是这条修法的代价")


# ═════════════════════ B. F-002 转正加第三条判据 ═════════════════════

CAPTION_IN_VERT = re.compile(
    r"^(?:Supplementary\s+|Extended\s+Data\s+)?(?:Table|TABLE|Tab\.)\s+(?:S?\d+|[IVXivx]{1,4})"
)


def page_vertical_lines(page) -> tuple[list[str], int]:
    """复制 pdfio._line_dirs 的口径：只有 round 后落在 _VERT_DIRS 的才算竖排。"""
    vert, horiz = [], 0
    for b in page.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            t = "".join(s["text"] for s in ln["spans"]).strip()
            if not t:
                continue
            d = tuple(round(x) for x in ln["dir"])
            if d == (1, 0):
                horiz += 1
            elif d in pdfio._VERT_DIRS:
                vert.append(t)
    return vert, horiz


def probe_b() -> None:
    head("B. F-002 转正判据加第三条：竖排文字里含 Table 标号")

    rows = []
    for tag, d in (("dev", DEV_PDF), ("holdout", HOLD_PDF)):
        for f in sorted(d.glob("*.pdf")):
            doc = fitz.open(f)
            prev_rot = False
            for pno, page in enumerate(doc, 1):
                vert, horiz = page_vertical_lines(page)
                old = bool(rules.rotation_for_page(len(vert), (0, -1), horiz))
                has_cap = any(CAPTION_IN_VERT.match(v) for v in vert)
                # 候选：原两条 且（竖排含 Table 标号 或 上一页已判横向表）
                new = old and (has_cap or prev_rot)
                if old or new:
                    ratio = len(vert) / max(len(vert) + horiz, 1)
                    rows.append((tag, f.stem[:28], pno, len(vert), ratio, old, new, has_cap))
                prev_rot = new
            doc.close()

    print(f"{'语料':8s} {'PDF':30s} {'页':>4s} {'竖排':>5s} {'占比':>5s}  现行 → 候选   竖排含Table")
    print("-" * 100)
    for tag, name, pno, nv, ratio, old, new, cap in rows:
        arrow = f"{'转' if old else '—'} → {'转' if new else '—'}"
        flag = "" if old == new else "   ← **变了**"
        print(f"{tag:8s} {name:30s} {pno:4d} {nv:5d} {ratio:5.0%}  {arrow:12s} "
              f"{'有' if cap else '无'}{flag}")

    # 地面真值：人眼核实过的（findings F-002 + 本轮渲染核对）
    TRUE = {("pbc_21078", 4), ("pbc_21078", 5), ("pbc_21296", 4), ("pbc_26870", 18),
            ("pbc_26870", 19), ("pbc_29304", 3), ("pbc_29304", 4)}
    TRUE_HOLD = {("Bruhm", 9)}   # 只用短前缀 —— name 被截到 28 字符，长 key 会 startswith 失败

    def truth(name, pno):
        return (name, pno) in TRUE or any(name.startswith(k) and pno == v for k, v in TRUE_HOLD)

    print()
    for label, idx in (("现行", 5), ("候选", 6)):
        tp = sum(1 for r in rows if r[idx] and truth(r[1], r[2]))
        fp = sum(1 for r in rows if r[idx] and not truth(r[1], r[2]))
        fn = sum(1 for r in rows if not r[idx] and truth(r[1], r[2]))
        print(f"  {label}：真表命中 {tp}，误报 {fp}，**漏判 {fn}**")
    print("  （地面真值来自人眼渲染核对，见 findings.md F-002）")


# ═════════════════════ C. F-007 无 caption 不得 high ═════════════════════

def probe_c() -> None:
    head("C. F-007 caption 为空的表不得为 high（最多 medium + notes=no_caption）")

    for tag, out in (("dev", DEV_OUT), ("holdout", HOLD_OUT)):
        hits, total = [], 0
        for man in sorted(out.glob("*/manifest.csv")):
            for r in csv.DictReader(man.open()):
                if not (r.get("csv_path") or ""):
                    continue
                total += 1
                if r.get("confidence") == "high" and not (r.get("caption") or "").strip():
                    hits.append((man.parent.name[:30], r.get("table_id"), r.get("n_rows"),
                                 (r.get("notes") or "")[:34]))
        print(f"  {tag}：{total} 张已导出的表，其中 **caption 为空且 confidence=high 的 {len(hits)} 张**")
        for n, tid, nr, notes in hits:
            print(f"      {n:32s} {tid:14s} {nr:>4s} 行   notes={notes or '(空)'}")
    print()
    print("  ⇒ 改动会把上面这些从 high 降到 medium。**开发集里有几张就是几张回归**，")
    print("     每一张都要人核实它到底该不该是 high。")


# ═════════════════════ D. F-008 字形名过滤 ═════════════════════

# ⚠ 第一版写成 `^/[a-zA-Z]{2,15}(?=[A-Z0-9(\[]|$)` —— **贪婪匹配把内容一起吃掉了**：
#   `/emspaceMean` → `''`（不是 `Mean`），而留出集里有 22 格是 `/emspaceClear cell`、
#   `/emspaceHigh-grade serous` 这类**卵巢癌组织学分型**，会被整格删空 = 丢数据。
#   改成**只认已知的空白字形名**，换成一个空格、保留后面的内容。
_SPACE_GLYPHS = "em|en|thin|hair|figure|punctuation|third|quarter|sixth|zerowidth|nb|no-break"
GLYPH = re.compile(rf"^/(?:{_SPACE_GLYPHS})space\s*", re.I)


def probe_d() -> None:
    head("D. F-008 过滤 docling 漏出来的字形名")

    print("① 修复效果（第二版：只认空白字形名，保留内容）")
    for s in ["/emspaceMean", "/emspaceRange", "/emspaceI", "/emspaceClear cell",
              "/emspaceHigh-grade serous", "/emspaceCystadenoma/adenofibroma",
              "/bowtie", "/enspace2.5", "Mean"]:
        out = GLYPH.sub("", s)
        mark = "" if out else "   ← ⚠ 变空了"
        print(f"    {s!r:36s} → {out!r}{mark}")

    print()
    print("② 影响面：全语料有多少格子会被改")
    for tag, out in (("dev", DEV_OUT), ("holdout", HOLD_OUT)):
        n_cell = n_hit = 0
        samples = []
        for p in sorted(out.glob("*/p*.csv")):
            if p.name in ("manifest.csv", "captions.csv"):
                continue
            for row in csv.reader(p.open(newline="")):
                for c in row:
                    if not c.strip():
                        continue
                    n_cell += 1
                    if GLYPH.match(c):
                        n_hit += 1
                        if len(samples) < 4:
                            samples.append(f"{p.parent.name[:18]}: {c[:24]!r}")
        print(f"    {tag}：{n_cell} 个非空格子，命中 {n_hit} 个  {samples}")

    print()
    print("③ 误伤检查：所有以 `/` 开头的格子，改完变成什么")
    for tag, out in (("dev", DEV_OUT), ("holdout", HOLD_OUT)):
        slash = set()
        for p in sorted(out.glob("*/p*.csv")):
            if p.name in ("manifest.csv", "captions.csv"):
                continue
            for row in csv.reader(p.open(newline="")):
                for c in row:
                    if c.strip().startswith("/"):
                        slash.add(c.strip())
        emptied = [c for c in slash if c and not GLYPH.sub("", c).strip()]
        untouched = [c for c in slash if GLYPH.sub("", c) == c]
        print(f"    {tag}：以 / 开头 {len(slash)} 种")
        for c in sorted(slash)[:8]:
            print(f"        {c[:34]!r:38s} → {GLYPH.sub('', c)[:30]!r}")
        print(f"      被清空的 {len(emptied)} 种 {sorted(emptied)[:3]}"
              f"   没动的 {len(untouched)} 种 {sorted(untouched)[:3]}")


if __name__ == "__main__":
    probe_a()
    probe_b()
    probe_c()
    probe_d()
