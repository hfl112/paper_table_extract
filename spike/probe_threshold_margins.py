"""探针：留出集跑完之后，各阈值的真实余量还剩多少 / 哪些根本没被碰到。

    conda run -n gemini python spike/probe_threshold_margins.py

不凭记忆列 `log.md` §11 那张表 —— 在**开发集 12 篇 + 留出集 25 篇**上重新实测每个决策点的
两侧观测值，看：

  1. 哪些阈值的余量被留出集**收窄或击穿**了
  2. 哪些阈值**留出集根本没碰到**（`--list` 不经过 / 语料里没有该场景）
  3. 哪些只验了单侧（比如只有反例、没有正例）

只做**读取与统计**，不改任何代码、不写任何产物。
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import fitz

from pdf_table_extract import pdfio, rules

DEV = Path("0.pdf_input")
HOLDOUT = Path("/Users/funanhe/Downloads/Zotero_Precancer_PDFs")
# 阈值验证集：按性质定向挑的，**统计量偏**，只用来看逐条的两侧观测值
THRESH = Path("/Users/funanhe/Downloads/pte_threshold_corpus")

CAPTION = re.compile(
    r"^(?:Supplementary\s+)?(?:Table|TABLE|Tab\.)\s+(?:S?\d+|[IVXivx]{1,4})(?:[.:|\s]|$)"
)


def corpora() -> list[tuple[str, Path]]:
    return ([("dev", p) for p in sorted(DEV.glob("*.pdf"))]
            + [("holdout", p) for p in sorted(HOLDOUT.glob("*.pdf"))]
            + [("thresh", p) for p in sorted(THRESH.glob("*.pdf"))])


# ————————————————————— 转正：竖排行数 / 占比 —————————————————————

def probe_rotation() -> None:
    print("=" * 100)
    print("① 转正判据  VERTICAL_LINES_THRESHOLD=%d  且  VERTICAL_RATIO_THRESHOLD=%.2f"
          % (rules.VERTICAL_LINES_THRESHOLD, rules.VERTICAL_RATIO_THRESHOLD))
    print("=" * 100)
    真, 假 = [], []
    for tag, path in corpora():
        doc = fitz.open(path)
        for pno, page in enumerate(doc, 1):
            vert, horiz = [], 0
            for b in page.get_text("dict")["blocks"]:
                for ln in b.get("lines", []):
                    t = "".join(s["text"] for s in ln["spans"]).strip()
                    if not t:
                        continue
                    if ln["dir"] == (1, 0):
                        horiz += 1
                    else:
                        vert.append(t)
            if len(vert) < rules.VERTICAL_LINES_THRESHOLD:
                continue
            ratio = len(vert) / (len(vert) + horiz)
            # 地面真值：竖排文字里含 Table 标号 ⇒ 真横向表（7/7 命中、3/3 排除，见 findings F-002）
            # 例外：pbc_26870 p19 是续表页、caption 为空，人核实过是真表
            is_true = any(CAPTION.match(v) for v in vert) or (path.stem == "pbc_26870" and pno == 19)
            rec = (f"{tag}:{path.stem[:24]} p{pno}", len(vert), ratio)
            (真 if is_true else 假).append(rec)

        doc.close()

    def span(rows, i):
        vals = [r[i] for r in rows]
        return (min(vals), max(vals)) if vals else (None, None)

    print(f"  真横向表 {len(真)} 页：竖排行数 {span(真,1)}   占比 {span(真,2)[0]:.0%}–{span(真,2)[1]:.0%}")
    print(f"  误  报  {len(假)} 页：竖排行数 {span(假,1)}   占比 {span(假,2)[0]:.0%}–{span(假,2)[1]:.0%}")
    print()
    for name, rows in (("真", 真), ("假", 假)):
        for n, v, r in sorted(rows, key=lambda x: x[1]):
            print(f"    [{name}] {n:34s} 竖排{v:5d}  占比{r:4.0%}")
    print()
    # ⚠ 上面「假」那一栏是**通过了行数判据的候选**，不等于工具的误报 ——
    #   工具要求两条**同时**满足。口径搞混会得出"两条判据都失效了"这种夸大结论。
    real_fp = [r for r in 假 if r[2] >= rules.VERTICAL_RATIO_THRESHOLD]
    real_tp = [r for r in 真 if r[2] >= rules.VERTICAL_RATIO_THRESHOLD]
    print()
    print("  —— 按工具的实际判据（两条**同时**满足）——")
    print(f"  真横向表 {len(real_tp)}/{len(真)} 页被正确判出；"
          f"漏判 {len(真) - len(real_tp)} 页")
    print(f"  **实际误报 {len(real_fp)} 页**：")
    for n, v, r in sorted(real_fp, key=lambda x: -x[2]):
        print(f"      {n:34s} 竖排{v:5d}  占比{r:4.0%}")
    print()
    lo_true, _ = span(真, 1)
    _, hi_false = span(假, 1)
    print(f"  ⇒ **单看行数**：真表最低 {lo_true}，通过行数判据的非真表最高 {hi_false}"
          f" —— 完全重叠，单用必错（AGENTS.md 事实 2c 早已说明）")
    lo_tr, _ = span(真, 2)
    hi_fa = max(r[2] for r in 假)
    print(f"  ⇒ **单看占比**：真表最低 {lo_tr:.0%}，非真表最高 {hi_fa:.0%}"
          f" —— 也重叠，单用必错")
    if real_fp:
        print(f"  ⇒ **两条与在一起**：仍有 {len(real_fp)} 页误报（占比 "
              f"{min(r[2] for r in real_fp):.0%}–{max(r[2] for r in real_fp):.0%}，"
              f"全都落在真表区间 {lo_tr:.0%}–{span(真,2)[1]:.0%} 之内）")
        print(f"     → **这才是新情况**：以前是「单条不够、两条够」，现在两条也不够了。")
    else:
        print("  ⇒ 两条与在一起：0 误报，判据仍然成立")


# ————————————————————— preprint —————————————————————

def probe_preprint() -> None:
    print()
    print("=" * 100)
    print(f"② preprint  PREPRINT_MIN_PAGE_RATIO={pdfio.PREPRINT_MIN_PAGE_RATIO}")
    print("=" * 100)
    rows = []
    for tag, path in corpora():
        info = pdfio.read_doc(path)
        rows.append((tag, path.stem[:40], info.preprint_pages / info.n_pages))
    rows.sort(key=lambda r: -r[2])
    for tag, name, r in rows[:6]:
        flag = "← 判为 preprint" if r >= pdfio.PREPRINT_MIN_PAGE_RATIO else ""
        print(f"    {tag:8s} {name:42s} {r:5.0%}  {flag}")
    print("    ...")
    above = [r for r in rows if r[2] >= pdfio.PREPRINT_MIN_PAGE_RATIO]
    below = [r for r in rows if r[2] < pdfio.PREPRINT_MIN_PAGE_RATIO]
    print(f"  ⇒ 判为 preprint 的最低 {min(r[2] for r in above):.0%}；"
          f"未判的最高 {max(r[2] for r in below):.0%}   "
          f"余量 {min(r[2] for r in above) - max(r[2] for r in below):.0%}")


# ————————————————————— OCR 文字层 —————————————————————

def probe_ocr_layer() -> None:
    print()
    print("=" * 100)
    print(f"③ OCR 文字层  FULL_PAGE_IMAGE_COVER={rules.FULL_PAGE_IMAGE_COVER}  "
          f"且  OCR_LAYER_MIN_CHARS={rules.OCR_LAYER_MIN_CHARS}")
    print("=" * 100)
    hit = defaultdict(list)
    for tag, path in corpora():
        info = pdfio.read_doc(path)
        for pg in info.pages:
            if pg.ocr_text_layer:
                hit[tag].append(f"{path.stem[:22]} p{pg.page} ({pg.n_chars}字)")
    for tag in ("dev", "holdout", "thresh"):
        print(f"    {tag:8s} 判为 OCR 文字层的页: {len(hit[tag]):4d}  {hit[tag][:2]}")
    n_pos = sum(len(v) for v in hit.values())
    print(f"  ⇒ 正例合计 {n_pos} 页。")
    print("     **但这条判据的两侧余量仍然测不出** —— 阈值是「整页位图 >=95% **且** 文字层 >=1000 字」，")
    print("     要量余量得看**贴着阈值**的页（位图覆盖 90-95%、或文字层 800-1200 字），")
    print("     而不是看命中了多少页。见下。")
    # 真正的余量：贴着两个阈值的页
    near = []
    for tag, path in corpora():
        doc = fitz.open(path)
        for pno, page in enumerate(doc, 1):
            w, h = page.rect.width, page.rect.height
            n = len(page.get_text())
            cov = max(((im["bbox"][2] - im["bbox"][0]) / w) * ((im["bbox"][3] - im["bbox"][1]) / h)
                      for im in page.get_image_info()) if page.get_image_info() else 0
            full = cov >= rules.FULL_PAGE_IMAGE_COVER ** 2
            if full and 600 <= n <= 2600:      # 贴着字数阈值
                near.append((f"{tag}:{path.stem[:22]} p{pno}", n, n >= rules.OCR_LAYER_MIN_CHARS))
        doc.close()
    near.sort(key=lambda x: x[1])
    print(f"     贴着字数阈值（600-2600 字）且整页被位图覆盖的页共 {len(near)} 个：")
    for nm, n, yes in near[:14]:
        print(f"        {nm:36s} {n:5d} 字  {'判为OCR层' if yes else '未判'}")
    lo = [x for x in near if not x[2]]
    hi = [x for x in near if x[2]]
    if lo and hi:
        print(f"     ⇒ 真实余量：未判的最高 {max(x[1] for x in lo)} 字 ← 阈值 "
              f"{rules.OCR_LAYER_MIN_CHARS} → 判了的最低 {min(x[1] for x in hi)} 字")


# ————————————————————— legend 续段（余量最紧的一条）—————————————————————

def probe_legend() -> None:
    print()
    print("=" * 100)
    print(f"④ legend 续段  gap < {rules.LEGEND_MAX_BLOCK_GAP}  且  "
          f"最长行 >= {rules.LEGEND_MIN_PROSE_LINE}   ← log.md §11 记为余量最紧的一条")
    print("=" * 100)
    # **必须复制 pdfio._absorb_legend_tail 的两道过滤**，否则测出来的 pair 不是工具真实
    # 考虑的 pair：① `ny0 < bottom - 1` 跳过（不在下方，这也是负 gap 到不了判据的原因）
    # ② 横向不重叠时**跳过而不是终止**（pbc_30017 的竖排水印 block 会插在中间）
    absorbed, stopped = [], []
    for tag, path in corpora():
        doc = fitz.open(path)
        for pno, page in enumerate(doc, 1):
            blocks = [b for b in page.get_text("dict")["blocks"] if "lines" in b]
            blocks.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
            for i, b in enumerate(blocks):
                lines = ["".join(s["text"] for s in ln["spans"]).strip() for ln in b["lines"]]
                if not any(CAPTION.match(x) for x in lines):
                    continue
                bottom = b["bbox"][3]
                bx0, bx1 = b["bbox"][0], b["bbox"][2]
                for nb in blocks[i + 1:]:
                    nx0, ny0, nx1, ny1 = nb["bbox"]
                    if ny0 < bottom - 1:
                        continue
                    if min(bx1, nx1) - max(bx0, nx0) <= 0:
                        continue
                    nlines = ["".join(s["text"] for s in ln["spans"]).strip() for ln in nb["lines"]]
                    if any(CAPTION.match(t) for t in nlines):
                        break
                    gap, mx = ny0 - bottom, max((len(t) for t in nlines), default=0)
                    rec = (f"{tag}:{path.stem[:22]} p{pno}", gap, mx)
                    if rules.is_legend_continuation(gap, mx):
                        absorbed.append(rec)
                        bottom = ny1
                    else:
                        stopped.append(rec)
                        break
        doc.close()

    print(f"  吸收（判为 legend 续段）{len(absorbed)} 处：")
    for n, g, m in sorted(absorbed, key=lambda x: -x[1])[:6]:
        print(f"      {n:32s} gap={g:7.1f}  最长行={m:4d}")
    print(f"  停住 {len(stopped)} 处，其中**擦边**的（gap<25 或 最长行 30-50）：")
    edge = [r for r in stopped if r[1] < 25 or 30 <= r[2] <= 50]
    for n, g, m in sorted(edge, key=lambda x: x[1])[:8]:
        why = []
        if g >= rules.LEGEND_MAX_BLOCK_GAP:
            why.append(f"gap {g:.1f}≥{rules.LEGEND_MAX_BLOCK_GAP}")
        if m < rules.LEGEND_MIN_PROSE_LINE:
            why.append(f"行长 {m}<{rules.LEGEND_MIN_PROSE_LINE}")
        print(f"      {n:32s} gap={g:7.1f}  最长行={m:4d}   停因: {'+'.join(why)}")
    if absorbed:
        print(f"  ⇒ 吸收的最长行最小值 = {min(m for _, _, m in absorbed)}，"
              f"阈值 {rules.LEGEND_MIN_PROSE_LINE}")
    near = [r for r in stopped if rules.LEGEND_MIN_PROSE_LINE - 12 <= r[2] < rules.LEGEND_MIN_PROSE_LINE]
    if near:
        print(f"  ⚠ 有 {len(near)} 处因「行长差一点」被停住，最高 {max(m for _,_,m in near)} "
              f"（阈值 {rules.LEGEND_MIN_PROSE_LINE}）—— 余量就这么点")


# ————————————————————— 没被 --list 碰到的 —————————————————————

def probe_untouched() -> None:
    print()
    print("=" * 100)
    print("⑤ 留出集这一趟**根本没碰到**的阈值（--list 不跑抽取）")
    print("=" * 100)
    from pdf_table_extract import engine_paddle
    rows = [
        ("CROP_PAD_RATIO", engine_paddle.CROP_PAD_RATIO, "图片裁剪留边", "单点拟合（Heat Map 那一列）"),
        ("MIN_RIVAL_OVERLAP", rules.MIN_RIVAL_OVERLAP, "caption 归属校验", "单点拟合"),
        ("MIN_IMAGE_COVER_H", rules.MIN_IMAGE_COVER_H, "logo 过滤", "3.9% vs 13.2%，约 2×"),
        ("TABLE_CELLS_THRESHOLD", rules.TABLE_CELLS_THRESHOLD, "OCR 闸门", "0–6 vs 99–300，宽"),
        ("TEXT_IN_REGION_THRESHOLD", rules.TEXT_IN_REGION_THRESHOLD, "文字/图片分流", "0 vs 1036–3190，宽"),
        ("SPARSE_BAND_RATIO", engine_paddle.SPARSE_BAND_RATIO, "密集单元格带", "稀疏 1 vs 密集 47"),
        ("COVERAGE_LOW", rules.COVERAGE_LOW, "内容覆盖率降级", "实测拦不住真 merged_row（事实 #15）"),
    ]
    print(f"    {'阈值':28s} {'当前值':>8s}  {'管什么':16s} 余量状态（来自 log.md §11）")
    print("    " + "-" * 94)
    for name, val, what, margin in rows:
        print(f"    {name:28s} {val!s:>8s}  {what:16s} {margin}")
    print()
    print("  ⇒ 这 7 个要验，必须跑**完整抽取**（eval 的事），留出集趟一给不了任何证据。")


if __name__ == "__main__":
    probe_rotation()
    probe_preprint()
    probe_ocr_layer()
    probe_legend()
    probe_untouched()
