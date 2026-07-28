"""三个候选修法的 spike 验证。**不改包代码。**

    conda run -n gemini python spike/probe_fix_candidates.py --only A   # docling 行错：坐标拆分
    conda run -n gemini python spike/probe_fix_candidates.py --only B   # F-017 断行连字符
    conda run -n gemini python spike/probe_fix_candidates.py --only C   # 图片表三个子候选（慢）

范式照抄 `spike/probe_candidate_fixes.py`：每个候选独立实现一遍，跟现行实现并排跑，
输出两栏 ——

    修复效果  —— 目标问题解决了吗
    回归影响  —— 其它表的结果变了吗（**必须为 0，除非是有意改基线**）

**只有"效果为正 且 回归为 0"的候选，才允许在下一轮写回产品。**

═══ 三个候选对应的三类错（log.md §26-29）═══

    A  docling 错在「行」—— TableFormer 把两个数据行合成一个 cell（F-001）
    B  docling 错在「拼接」—— 断行连字符没接回（F-017）
    C  paddle OCR 错在「字」—— 低分辨率图认错字
"""

from __future__ import annotations

import argparse
import csv
import pickle
import re
import statistics
import sys
import unicodedata
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import geom, gold as gold_io  # noqa: E402
from pdf_table_extract import engine_docling  # noqa: E402
from pdf_table_extract.models import Rect  # noqa: E402

CACHE = Path("/tmp/pte_rowbands_cache")
GOLD_DIR = Path("/Users/funanhe/pte_gold")
DEV = ROOT / "0.pdf_input"

CELL_TALL = 1.5          # 候选 A：cell 高于同列中位数这么多倍算可疑
WRAP_MAX_COLS = 2        # 候选 A：折行续段最多占这么多列（实测折行 1-2 列、压行 6+ 列）

# 候选 A 的**适用范围闸门**（用户 2026-07-27 提出，留出集实测定的）。
#
# 「比同列中位数高 1.5x」这个信号的**前提**是：同列其它格子的高度本来是整齐的。
# 前提不成立时（文本型综述表，每格都是长句、行高天然参差），这个信号毫无意义 ——
# 实测留出集 27 张表命中 28 次、拆分 14 次**全错**，开发集 28 张表只命中 1 次。
#
# 闸门：某列只有在「排除可疑行与表头后，极差/中位 <= 本阈值」时，才算作证据列。
#
# 实测（留出集 14 处误拆 + 开发集 1 处真压行）：
#     该拆      极差/中位  6.5% / 7.1%
#     不该拆    最小 13.9%，多数 92%-409%；每行取最大后再取最小 = 114.8%
#     → 余量 7.1% <- 30% -> 114.8%，**16 倍**
#
# 30% 取的是两侧的**几何中心**（sqrt(7.1 x 114.8) = 28.6%），不偏向任何一侧。
# ⚠ **这个阈值是在留出集上量出来的** —— 留出集对它不再是独立证据，要用新料验。
HEIGHT_EVEN_MAX = 0.30
MIN_EVEN_SAMPLES = 4     # 同列可用样本少于这么多就判定不出整齐度 → 保守放弃


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    for a, b in (("−", "-"), ("–", "-"), ("—", "-")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def docling_dump(stem: str) -> list[dict]:
    ck = CACHE / f"{stem}.pkl"
    if not ck.exists():
        raise SystemExit(f"缺 docling 缓存 {ck} —— 先跑 spike/probe_row_bands.py")
    return pickle.loads(ck.read_bytes())


# ═══════════════════════════ 候选 B：断行连字符 ═══════════════════════════

# 只匹配「小写字母 + 连字符 + 空白 + 小写字母」。
# 关键区别（AGENTS.md 已知限制 #5）：本语料的连字符 token 全是标识符
# （`PR-104` / `NB-SD` / `ALL-11` / `KT-13`），但**那些连字符后面没有空白**。
# 这个区别是本判据成立的全部依据 —— 所以下面的「回归影响」栏必须把三个语料里
# 全部出现逐条列出来人工确认，不能靠推理。
_HYPHEN_BREAK = re.compile(r"([a-z])-\s+([a-z])")

# 拼接后要在文字层里找得到才允许接 —— **这一条是第一版实测逼出来的**。
#
# 第一版只用正则，三个语料 5 种命中里 **3 种是假阳性**：
#   `lineage- defining` → `lineagedefining`   ❌ `lineage-defining` 是合法连字符复合词
#   `patient- derived`  → `patientderived`    ❌ 同上
#   `Log-rank p- vaiue` → `Log-rank pvaiue`   ❌ `p-value` 的连字符该保留
# 只有 `cyclophos- phamide` → `cyclophosphamide` 是真断行。
#
# **surface pattern 完全相同，正则分不开。** 区别在于：真断行拼接后是**一个真词**，
# 合法复合词拼接后是个不存在的字符串。所以判据必须去文字层里查一次。
_WORDISH = re.compile(r"[a-z]+")


def fix_hyphen(s: str, page_words: set[str] | None = None) -> str:
    """接回断行连字符。`page_words` 为 None 时退回纯正则（只用于展示第一版的问题）。"""
    def repl(m: re.Match) -> str:
        if page_words is None:
            return m.group(1) + m.group(2)
        # 取连字符两侧的完整词片段，拼起来查文字层
        left = s[:m.start(1) + 1]
        right = s[m.end(2) - 1:]
        lw = (_WORDISH.findall(left.lower()) or [""])[-1]
        rw = (_WORDISH.findall(right.lower()) or [""])[0]
        joined = lw + rw
        if joined and joined in page_words:
            return m.group(1) + m.group(2)
        return m.group(0)  # 查不到 → 不动，保留原样
    return _HYPHEN_BREAK.sub(repl, s)



# —— 文字层词表（按篇缓存）——
_PDF_DIRS = [ROOT / "0.pdf_input", Path("/Users/funanhe/Downloads/Zotero_Precancer_PDFs")]
_WORDS_CACHE: dict[str, set[str]] = {}


def doc_words(out_dir_name: str) -> set[str] | None:
    """某篇文档文字层里出现过的「纯字母词」集合（小写、已去连字符与空白）。

    作用域取**整篇**而不是单页：断行的词有时前后半落在不同页（跨页表）。
    这里只查「这个拼接结果是不是一个真词」，不是拿它验证网格，
    所以不受 F-004「文字层比对必须按页」那条约束 —— 那条针对的是**逐格验证**。
    """
    if out_dir_name in _WORDS_CACHE:
        return _WORDS_CACHE[out_dir_name]
    from pdf_table_extract.emit import sanitize_name
    import fitz
    pdf = None
    for d in _PDF_DIRS:
        if not d.exists():
            continue
        for cand in d.glob("*.pdf"):
            if cand.stem == out_dir_name or sanitize_name(cand.stem) == out_dir_name:
                pdf = cand
                break
        if pdf:
            break
    if pdf is None:
        _WORDS_CACHE[out_dir_name] = None
        return None
    doc = fitz.open(pdf)
    try:
        txt = " ".join(doc[i].get_text() for i in range(len(doc)))
    finally:
        doc.close()
    # **不要先把连字符接掉。** 第一版这么做了，结果词表里凭空出现 `lineagedefining`,
    # 于是判据自己证明自己 —— 循环论证。按非字母切词即可：
    # 文字层里 `lineage-defining` 切出 `lineage` + `defining`（**不会**有 `lineagedefining`），
    # 而 `cyclophosphamide` 在别处以完整词出现过，能查到。
    words = set(_WORDISH.findall(txt.lower()))
    _WORDS_CACHE[out_dir_name] = words
    return words


def cand_b() -> int:
    print("=" * 100)
    print("候选 B：断行连字符（F-017）—— `([a-z])-\\s+([a-z])` → `\\1\\2`")
    print("=" * 100)

    # ——— 回归影响：先做这一栏。它是判据成立与否的关键 ———
    print("\n【回归影响】三个语料里 `[a-z]-\\s+[a-z]` 的**全部**出现，逐条确认是断行还是标识符")
    print("-" * 100)
    hits: list[tuple[str, str, str]] = []
    for root in (ROOT / "1.output", ROOT / "holdout"):
        if not root.exists():
            continue
        for csv_path in sorted(root.rglob("*.csv")):
            if csv_path.name in ("manifest.csv", "captions.csv"):
                continue
            try:
                rows = list(csv.reader(csv_path.open(newline="")))
            except Exception:
                continue
            for r in rows:
                for cell in r:
                    if _HYPHEN_BREAK.search(cell):
                        pw = doc_words(csv_path.parent.name)
                        hits.append((f"{csv_path.parent.name}/{csv_path.name}",
                                     cell.strip(), fix_hyphen(cell).strip(),
                                     fix_hyphen(cell, pw).strip()))
    if not hits:
        print("  三个语料里零命中 —— 判据无从验证（也说明它极少触发）")
    seen = set()
    uniq = []
    for src, before, naive, guarded in hits:
        if before in seen:
            continue
        seen.add(before)
        uniq.append((src, before, naive, guarded))
    print(f"  共 {len(hits)} 处出现，去重后 {len(uniq)} 种。")
    print("  「纯正则」是第一版；「查文字层」是收紧后的版本。\n")
    changed = 0
    for src, before, naive, guarded in uniq:
        mark = "**改了**" if guarded != before else "不动"
        changed += int(guarded != before)
        print(f"    {src[:50]:52s} → {mark}")
        m = _HYPHEN_BREAK.search(before)
        frag = before[max(0, m.start() - 14):m.end() + 14] if m else before[:40]
        print(f"        片段     …{frag}…")
        print(f"        纯正则   {'改' if naive != before else '不动'}"
              f"    查文字层  {'改' if guarded != before else '不动'}")
    print(f"\n  收紧后只改 {changed}/{len(uniq)} 种（第一版会改 "
          f"{sum(1 for _s,b,n,_g in uniq if n != b)} 种）")
    print("\n  ⚠ **上面每一条都要人眼确认是断行、而不是标识符。** 出现任何标识符即判据不合格。")

    # ——— 修复效果：pbc_28772 的 gold 对齐率 ———
    print("\n【修复效果】`pbc_28772 p03_table_1` 的 gold 对齐率与命中率")
    print("-" * 100)
    gp = GOLD_DIR / "pbc_28772__p03_table_1.gold"
    if not gp.exists():
        print("  找不到 gold，跳过")
        return 0
    g = gold_io.load(gp)
    src = ROOT / "1.output" / "pbc_28772" / g.csv_name
    rows = list(csv.reader(src.open(newline="")))
    pw = doc_words("pbc_28772")
    fixed = [[fix_hyphen(c, pw) for c in r] for r in rows]

    n_gold = sum(1 for a in g.values if g.sources.get(a, "agreed") in {"human", "agreed"})
    denom = max(1, n_gold * len(g.check_cols))
    print("%-10s %8s %8s %9s %8s %8s %9s" %
          ("", "计分", "命中", "命中率", "漏行", "多出行", "对齐率"))
    for tag, rr in (("修前", rows), ("修后", fixed)):
        res = gold_io.score_against(g, rr, {"human", "agreed"})
        t = res.totals
        rate = f"{t['match'] / t['scored']:.1%}" if t["scored"] else "—"
        print("%-10s %8d %8d %9s %8d %8d %9.0f%%" %
              (tag, t["scored"], t["match"], rate, len(res.missing_rows),
               len(res.extra_rows), 100 * t["scored"] / denom))
    return 0


# ═══════════════════════════ 候选 A：坐标拆分 ═══════════════════════════


def column_evenness(items: list[tuple[int, float]], skip_row: int) -> float | None:
    """同列（排除表头与可疑行）高度的 极差/中位。None = 样本不够、判定不出。"""
    hs = [h for r0, h in items if r0 >= 1 and r0 != skip_row]
    if len(hs) < MIN_EVEN_SAMPLES:
        return None
    med = statistics.median(hs)
    return None if med <= 0 else (max(hs) - min(hs)) / med


def tall_rows(cells: list[dict], gate: bool = True) -> dict[int, list[int]]:
    """定位可疑行：排除表头行后，同一数据行 >=2 列的 cell 高于该列中位数 CELL_TALL 倍。

    表头行**必须排除** —— 实测 `pbc_21296` p4 表头合法折行，6 列全部超标 2.07-2.24x，
    不排除就是 100% 误报。

    `gate=True` 时再加一道**适用范围闸门**：某列只有在自身高度分布够整齐时才算证据列
    （见 HEIGHT_EVEN_MAX 的注释）。`gate=False` 用来复现加闸门之前的行为做对照。
    """
    by_col: dict[int, list[tuple[int, float]]] = {}
    for c in cells:
        if not c["bbox"] or c["rspan"] != 1:
            continue
        h = abs(c["bbox"][3] - c["bbox"][1])
        if h > 0:
            by_col.setdefault(c["c0"], []).append((c["r0"], h))
    sus: dict[int, list[int]] = {}
    for col, items in by_col.items():
        if len(items) < 6:
            continue
        med = statistics.median(h for _, h in items)
        if med <= 0:
            continue
        for r0, h in items:
            if r0 < 1 or h <= CELL_TALL * med:
                continue
            if gate:
                ev = column_evenness(items, r0)
                if ev is None or ev > HEIGHT_EVEN_MAX:
                    continue   # 该列本身就不整齐 → 这个"偏高"没有意义
            sus.setdefault(r0, []).append(col)
    return {r: cols for r, cols in sus.items() if len(cols) >= 2}


def split_row(pdf: Path, t: dict, row_idx: int, words) -> list[list[str]] | None:
    """把 docling 第 row_idx 行按词坐标拆成多行。返回 None 表示判定为折行、不该拆。

    折行 vs 压行的判据（实测差一个数量级）：
      折行续段  占 1-2 列
      真数据行  占 6+ 列（`pbc_21296` 那行 10 列里有 6 列超标）
    """
    cells = [c for c in t["cells"] if c["r0"] == row_idx and c["bbox"]]
    if not cells:
        return None
    ncol = max(c["c0"] for c in t["cells"]) + 1
    # 每个 cell 内的词，按 y 聚成带
    per_cell: dict[int, list[list]] = {}
    for c in cells:
        x0, y0, x1, y1 = c["bbox"]
        inside = [w for w in words if x0 - 1 <= w.cx <= x1 + 1 and y0 - 1 <= w.cy <= y1 + 1]
        if not inside:
            continue
        per_cell[c["c0"]] = geom.bands_by_center(inside, "y", 3.0)

    nband = max((len(b) for b in per_cell.values()), default=0)
    if nband < 2:
        return None
    # 有 >=2 个带的列数：折行只有 1-2 列，压行会有很多列
    multi = [col for col, bands in per_cell.items() if len(bands) >= 2]
    if len(multi) <= WRAP_MAX_COLS:
        return None  # 折行，不拆

    out = []
    for bi in range(nband):
        row = [""] * ncol
        for col, bands in per_cell.items():
            if bi < len(bands):
                row[col] = " ".join(w.text for w in sorted(bands[bi], key=lambda w: w.x0))
        out.append(row)
    return out


def cand_a() -> int:
    print("=" * 100)
    print("候选 A：docling 行错（F-001）—— 用词坐标把压成一格的行拆开")
    print("=" * 100)
    print("判据：cell 高 > 同列中位 %.1fx 且同行 >=2 列超标 ⇒ 可疑；"
          "格内词按 y 聚带，>%d 列有多带 ⇒ 判为压行并拆开" % (CELL_TALL, WRAP_MAX_COLS))
    print()
    print("%-24s %5s %8s %8s %8s  %s" %
          ("表", "页", "docling", "拆分后", "人核真值", "判定"))
    print("-" * 100)

    TRUTH = {("pbc_21296", 4): 44, ("pbc_28772", 3): 38, ("pbc_26870", 18): 26,
             ("pbc_21078", 4): 43, ("pbc_21078", 6): 31, ("pbc_21078", 8): 35,
             ("pbc_24724", 4): 41, ("pbc_30017", 7): 12}
    fixed_ok = regress = 0
    for (stem, page), truth in sorted(TRUTH.items()):
        try:
            dump = docling_dump(stem)
        except SystemExit as e:
            print(f"  {e}")
            return 1
        cands = [t for t in dump if t["page"] == page and t.get("rows")]
        if not cands:
            continue
        t = max(cands, key=lambda x: len(x["rows"]))
        nrow = len(t["rows"]) - 1
        sus = tall_rows(t["cells"])
        with geom.normalized_pdf(DEV / f"{stem}.pdf") as (npdf, _i):
            words = geom.page_words(npdf, page)
        rect = Rect(*t["rect"])
        words = geom.words_in_rect(words, rect)

        added = 0
        detail = []
        for r in sorted(sus):
            got = split_row(DEV / f"{stem}.pdf", t, r, words)
            if got and len(got) >= 2:
                added += len(got) - 1
                detail.append((r, got))
        after = nrow + added
        if stem == "pbc_21296" and page == 4:
            verdict = "✓ 修好了" if after == truth else f"✗ 应为 {truth}"
            fixed_ok = int(after == truth)
        else:
            verdict = "✓ 无回归" if after == nrow else f"✗ **回归** 多了 {added} 行"
            regress += int(after != nrow)
        print("%-24s %5d %8d %8d %8d  %s" % (stem[:24], page, nrow, after, truth, verdict))
        for r, got in detail:
            for g in got:
                print("        拆出 %s" % [c[:12] for c in g[:6]])
    print("-" * 100)

    # ——— 拿 gold 给拆分后的表打分：光是行数对了不够，各列值也得对 ———
    gp = GOLD_DIR / "pbc_21296__p04_table_i.gold"
    if gp.exists():
        g = gold_io.load(gp)
        src = ROOT / "1.output" / "pbc_21296" / g.csv_name
        rows = list(csv.reader(src.open(newline="")))
        dump = docling_dump("pbc_21296")
        t = max([x for x in dump if x["page"] == 4 and x.get("rows")],
                key=lambda x: len(x["rows"]))
        with geom.normalized_pdf(DEV / "pbc_21296.pdf") as (npdf, _i):
            ws = geom.words_in_rect(geom.page_words(npdf, 4), Rect(*t["rect"]))
        # 重建：把可疑行替换成拆出来的多行（docling 行号 r 对应 CSV 第 r 行）
        newrows = [rows[0]]
        for ri, r in enumerate(rows[1:], start=1):
            got = split_row(DEV / "pbc_21296.pdf", t, ri, ws) if ri in tall_rows(t["cells"]) else None
            if got and len(got) >= 2:
                newrows.extend(got)
            else:
                newrows.append(r)
        print("\n【修复效果】拿 gold 打分（source ∈ human+agreed）")
        n_gold = sum(1 for a in g.values if g.sources.get(a, "agreed") in {"human", "agreed"})
        denom = max(1, n_gold * len(g.check_cols))
        print("%-10s %8s %8s %9s %8s %8s %9s" %
              ("", "计分", "命中", "命中率", "漏行", "多出行", "对齐率"))
        for tag, rr in (("修前", rows), ("修后", newrows)):
            res = gold_io.score_against(g, rr, {"human", "agreed"})
            tt = res.totals
            rate = f"{tt['match'] / tt['scored']:.1%}" if tt["scored"] else "—"
            print("%-10s %8d %8d %9s %8d %8d %9.0f%%" %
                  (tag, tt["scored"], tt["match"], rate, len(res.missing_rows),
                   len(res.extra_rows), 100 * tt["scored"] / denom))
            for m in res.mismatches[:4]:
                print("        · %-16s %-20s 抽到 %r" % (m.anchor[:16], m.col[:20], m.ours[:20]))

    print(f"\n【行数判定】pbc_21296 p4: {'通过' if fixed_ok else '未通过'}")
    print(f"【回归影响】其余 7 张表出现回归 {regress} 处（**必须为 0**）")
    print("\n⚠ **正例只有 1 个**（误报侧有 27 张表的证据）。结论只能写成「在这 1 个实例上成立」。")
    return 0




# ═══════════════════════════ 候选 C：图片表 ═══════════════════════════
#
# C1  换 `RT-DETR-L_wireless_table_cell_det`（期刊表是三线表、无竖线，理论上更对路）
# C2  逐格 OCR（cell bbox 一直在手上但被 `cells_summary` 丢了）
# C3  删末尾全空行（根因 `engine_paddle._html_to_rows` 保留全空字符串的 <tr>）
#
# 用 `spike/out_images/` 与 `spike/lowres/` 的现成图，不复现取图链路。

IMGS = {
    "pbc_24724 fig_1 (300dpi)": ROOT / "spike/out_images/pbc_24724_p5_1.png",
    "pbc_21296 fig_1 (120dpi)": ROOT / "spike/lowres/pbc_21296.pd_p3_1_x3_crop.png",
}
GOLD_FOR = {
    "pbc_24724 fig_1 (300dpi)": "pbc_24724__p05_fig_1",
    "pbc_21296 fig_1 (120dpi)": "pbc_21296__p03_fig_1",
}
RESP = {"cr", "mcr", "pr", "sd", "pd", "pd1", "pd2", "ne", "int", "low", "high"}
CELL_PAD = 10        # 逐格裁剪的留边。**紧贴裁剪会让识别更差**（实测）
CELL_UPSCALE = 2     # 逐格放大倍数


def _cell_boxes(img: Path, model: str) -> list[tuple[float, float, float, float]]:
    from paddlex import create_model
    m = create_model(model_name=model)
    out = []
    for r in m.predict(str(img), batch_size=1):
        for b in r["boxes"]:
            x0, y0, x1, y1 = (float(v) for v in b["coordinate"])
            out.append((x0, y0, x1, y1))
    return out


def _bands(vals: list[float], tol: float) -> list[float]:
    """一维聚类，返回每个带的中心。"""
    if not vals:
        return []
    vs = sorted(vals)
    groups = [[vs[0]]]
    for v in vs[1:]:
        if v - groups[-1][-1] > tol:
            groups.append([])
        groups[-1].append(v)
    return [sum(g) / len(g) for g in groups]


def _cellwise_ocr(img: Path, boxes) -> list[list[str]]:
    """逐格 OCR 并按行列带重组。**padding + 放大 + 完整 det+rec** 缺一不可。"""
    from PIL import Image
    from rapidocr import RapidOCR
    import numpy as np
    eng = RapidOCR()
    im = Image.open(img).convert("RGB")
    hs = [b[3] - b[1] for b in boxes]
    ws = [b[2] - b[0] for b in boxes]
    ytol = statistics.median(hs) * 0.5 if hs else 5
    xtol = statistics.median(ws) * 0.5 if ws else 5
    rows_c = _bands([(b[1] + b[3]) / 2 for b in boxes], ytol)
    cols_c = _bands([(b[0] + b[2]) / 2 for b in boxes], xtol)
    grid = [["" for _ in cols_c] for _ in rows_c]
    for x0, y0, x1, y1 in boxes:
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        ri = min(range(len(rows_c)), key=lambda i: abs(rows_c[i] - cy))
        ci = min(range(len(cols_c)), key=lambda i: abs(cols_c[i] - cx))
        crop = im.crop((max(0, x0 - CELL_PAD), max(0, y0 - CELL_PAD),
                        min(im.width, x1 + CELL_PAD), min(im.height, y1 + CELL_PAD)))
        crop = crop.resize((crop.width * CELL_UPSCALE, crop.height * CELL_UPSCALE),
                           Image.LANCZOS)
        try:
            r = eng(np.array(crop))
            txt = " ".join(r.txts) if r and r.txts else ""
        except Exception:
            txt = ""
        if txt and len(txt) > len(grid[ri][ci]):
            grid[ri][ci] = txt.strip()
    return grid


def _score_codes(gold_name: str, rows: list[list[str]]) -> str:
    gp = GOLD_DIR / f"{gold_name}.gold"
    if not gp.exists() or not rows:
        return "—"
    g = gold_io.load(gp)
    from collections import Counter
    tc = Counter(c.strip().lower() for r in rows for c in r if c.strip().lower() in RESP)
    gc = Counter(v.strip().lower() for d in g.values.values() for v in d.values()
                 if v.strip().lower() in RESP)
    if not sum(gc.values()):
        return "—"
    return f"{sum((tc & gc).values())}/{sum(gc.values())} = {sum((tc & gc).values()) / sum(gc.values()):.0%}"


def cand_c() -> int:
    from pdf_table_extract import engine_paddle
    print("=" * 104)
    print("候选 C：图片表（C1 wireless 检测器 / C2 逐格 OCR / C3 删全空行）")
    print("=" * 104)
    for name, img in IMGS.items():
        if not img.exists():
            print(f"\n{name}: 找不到 {img}")
            continue
        gold_name = GOLD_FOR[name]
        print(f"\n══ {name}   {img.name}")
        print("-" * 104)

        # 基线：现行管线（wired 检测器 + 密集带裁剪 + 整表还原）
        base_grid = engine_paddle.cells_summary(img)
        print("  基线 wired 检测器：%d 格，行带 %d / 列带 %d"
              % (base_grid.n_cells, base_grid.n_row_bands, base_grid.n_col_bands))
        try:
            base_rows = engine_paddle.recognize_table(img)
        except Exception as e:
            base_rows = []
            print("    整表还原抛异常:", str(e)[:50])
        shape = f"{len(base_rows)}x{max((len(r) for r in base_rows), default=0)}" if base_rows else "失败"
        print("    整表还原 %-10s 响应码重叠 %s" % (shape, _score_codes(gold_name, base_rows)))
        if base_rows:
            print("    表头: %s" % [c[:16] for c in base_rows[0][:6]])

        # C1：wireless 检测器
        try:
            wl = _cell_boxes(img, "RT-DETR-L_wireless_table_cell_det")
            print("  C1 wireless 检测器：%d 格（wired %d）" % (len(wl), base_grid.n_cells))
        except Exception as e:
            wl = []
            print("  C1 wireless 检测器失败:", str(e)[:60])

        # C2：逐格 OCR（用 wired 的框，与基线同源，只换识别方式）
        try:
            wd = _cell_boxes(img, "RT-DETR-L_wired_table_cell_det")
            grid = _cellwise_ocr(img, wd)
            shape2 = f"{len(grid)}x{max((len(r) for r in grid), default=0)}"
            print("  C2 逐格 OCR：%-10s 响应码重叠 %s" % (shape2, _score_codes(gold_name, grid)))
            print("    表头: %s" % [c[:20] for c in grid[0][:6]])
            # C3 叠加在 C2 上
            ne = [r for r in grid if any(c.strip() for c in r)]
            print("  C3 删全空行（叠在 C2 上）：%d → %d 行" % (len(grid), len(ne)))
        except Exception as e:
            import traceback
            print("  C2 逐格 OCR 失败:", str(e)[:80])
            traceback.print_exc(limit=1)

        # C3 单独：叠在基线上
        if base_rows:
            ne = [r for r in base_rows if any(c.strip() for c in r)]
            print("  C3 删全空行（叠在基线上）：%d → %d 行" % (len(base_rows), len(ne)))
    print("-" * 104)
    print("**两张图片表都跑了** —— 第 3 步已证明同为图片表、两张失效模式完全不同。")
    return 0




# ═══════════════════════ 留出集泛用性检验 ═══════════════════════
#
# 留出集的职责（用户 2026-07-27 澄清）：**拿一批与开发无关的文章测阈值**。
# 允许据此调阈值，但**每次调完都要记下"这个阈值是在留出集上定的"** ——
# 否则它对后续改动就不再是独立证据了。
#
# ⚠ `AGENTS.md` 现在写的是「它是 test set，不是 validation set / 跑完不许调参」，
# 与上面这条不一致。**那份文档该改**，否则下一个 session 会按另一套规矩走。
#
# 本次跑的时候 A/B 的判据与常数与开发集上**完全一致**，没有任何调整。

HOLDOUT_PDFS = Path("/Users/funanhe/Downloads/Zotero_Precancer_PDFs")
HOLDOUT_CACHE = Path("/tmp/pte_holdout_cache")
CORPUS3_PDFS = Path("/Users/funanhe/Downloads/pte_threshold_corpus")
CORPUS3_CACHE = Path("/tmp/pte_corpus3_cache")


def holdout_dump(pdf: Path, cache: Path = HOLDOUT_CACHE) -> list[dict]:
    """跑一次 docling 拿 tables + per-cell bbox，缓存到 /tmp。"""
    cache.mkdir(exist_ok=True)
    ck = cache / f"{pdf.stem[:60]}.pkl"
    if ck.exists():
        return pickle.loads(ck.read_bytes())
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from pdf_table_extract import quiet
    quiet.hush_loggers()
    opts = PdfPipelineOptions(); opts.do_ocr = False; opts.do_table_structure = True
    conv = DocumentConverter(format_options={"pdf": PdfFormatOption(pipeline_options=opts)})
    out = []
    with geom.normalized_pdf(pdf) as (norm, _info):
        doc = conv.convert(str(norm)).document
        for ti in doc.tables:
            if not ti.prov:
                continue
            prov = ti.prov[0]
            ph = float(doc.pages[prov.page_no].size.height)
            try:
                rows = engine_docling._df_to_rows(ti.export_to_dataframe(doc))
            except Exception:
                rows = []
            cells = []
            for c in (ti.data.table_cells or []):
                bb = None
                if c.bbox:
                    r = engine_docling._to_topleft(c.bbox, ph)
                    bb = (r.x0, r.y0, r.x1, r.y1)
                cells.append({"text": c.text or "", "r0": c.start_row_offset_idx,
                              "r1": c.end_row_offset_idx, "c0": c.start_col_offset_idx,
                              "c1": c.end_col_offset_idx, "rspan": c.row_span, "bbox": bb})
            out.append({"page": prov.page_no,
                        "rect": engine_docling._to_topleft(prov.bbox, ph).as_tuple(),
                        "rows": rows, "cells": cells})
    ck.write_bytes(pickle.dumps(out))
    return out


def cand_holdout(pdf_dir: Path = HOLDOUT_PDFS, cache: Path = HOLDOUT_CACHE,
                 tag: str = "留出集") -> int:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"{pdf_dir} 下没有 PDF")
        return 1
    print("=" * 104)
    print("%s 泛用性检验（%d 篇）—— **只报结果，不调任何阈值**" % (tag, len(pdfs)))
    print("=" * 104)

    n_tab = 0
    n_ungated = [0]
    a_hits: list[tuple] = []
    b_hits: list[tuple] = []
    for pdf in pdfs:
        try:
            dump = holdout_dump(pdf, cache)
        except Exception as e:
            print("  %-40s docling 失败: %s" % (pdf.stem[:40], str(e)[:40]))
            continue
        words_doc = None
        for t in dump:
            if not t.get("rows"):
                continue
            n_tab += 1
            # ——— A：可疑行检测 + 拆分 ———
            # 同时统计不加闸门时会命中多少 —— 让"闸门有没有用"直接体现在产物里
            n_ungated[0] += len(tall_rows(t["cells"], gate=False))
            sus = tall_rows(t["cells"])
            if sus:
                with geom.normalized_pdf(pdf) as (npdf, _i):
                    ws = geom.words_in_rect(geom.page_words(npdf, t["page"]),
                                            Rect(*t["rect"]))
                for r in sorted(sus):
                    got = split_row(pdf, t, r, ws)
                    a_hits.append((pdf.stem, t["page"], r, len(sus[r]),
                                   t["rows"][r] if r < len(t["rows"]) else [], got))
            # ——— B：断行连字符 ———
            for r in t["rows"]:
                for cell in r:
                    if _HYPHEN_BREAK.search(cell):
                        if words_doc is None:
                            words_doc = _pdf_words(pdf)
                        b_hits.append((pdf.stem, t["page"], cell,
                                       fix_hyphen(cell, words_doc)))

    print("\n扫过 %d 篇 / %d 张 docling 表\n" % (len(pdfs), n_tab))

    print("【A 类：两行压一格】检测到 %d 处可疑行"
          "（**不加行高均匀闸门时会是 %d 处**）" % (len(a_hits), n_ungated[0]))
    print("-" * 104)
    split_n = 0
    for stem, page, r, ncol, orig, got in a_hits:
        acted = bool(got and len(got) >= 2)
        split_n += acted
        print("  %-34s p%-3d 行%-3d  %d 列超标  →  %s"
              % (stem[:34], page, r, ncol, "**拆成 %d 行**" % len(got) if acted else "判为折行，不拆"))
        print("      原格: %s" % [c[:18] for c in orig[:5]])
        for g in (got or []):
            print("      拆出: %s" % [c[:18] for c in g[:5]])
    if not a_hits:
        print("  （零命中）")
    print("  → 检测 %d 处，其中判为压行并拆分 %d 处" % (len(a_hits), split_n))

    print("\n【B 类：断行连字符】命中 %d 处" % len(b_hits))
    print("-" * 104)
    changed = 0
    seen_b = set()
    for stem, page, before, after in b_hits:
        if before in seen_b:
            continue
        seen_b.add(before)
        m = _HYPHEN_BREAK.search(before)
        frag = before[max(0, m.start() - 16):m.end() + 16] if m else before[:40]
        act = after != before
        changed += act
        print("  %-34s p%-3d  %s" % (stem[:34], page, "**改了**" if act else "不动"))
        print("      …%s…" % frag)
    if not b_hits:
        print("  （零命中）")
    print("  → 去重 %d 种，其中改了 %d 种" % (len(seen_b), changed))
    print("\n" + "-" * 104)
    print("上面每一条都要人眼核。判据与常数与开发集上跑的完全一致，本次未做任何调整。")
    return 0


def _pdf_words(pdf: Path) -> set[str]:
    import fitz
    doc = fitz.open(pdf)
    try:
        txt = " ".join(doc[i].get_text() for i in range(len(doc)))
    finally:
        doc.close()
    return set(_WORDISH.findall(txt.lower()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=("A", "B", "C", "holdout", "corpus3"), required=True)
    args = ap.parse_args()
    if args.only == "holdout":
        return cand_holdout()
    if args.only == "corpus3":
        return cand_holdout(CORPUS3_PDFS, CORPUS3_CACHE, "阈值验证集(53篇)")
    if args.only == "A":
        return cand_a()
    if args.only == "B":
        return cand_b()
    return cand_c()


if __name__ == "__main__":
    sys.exit(main())
