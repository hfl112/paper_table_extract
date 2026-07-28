"""留出集阈值体检：四个数，异常就说明某个阈值掉进了没见过的中间档。

    conda run -n gemini python holdout/report.py --pdf-dir /Users/funanhe/Downloads/Zotero_Precancer_PDFs

**只依赖 `pdf_table_extract` 本身，不碰 `eval/`。** 它检的不是"抽对没抽对"（留出集没有
ground truth），而是**阈值有没有明显失准**。

`log.md` §8 的三个泛化 bug 全是这么抓到的，模式每次都一样：
**只见过两个极端，阈值落进了没见过的中间档。**

判定不通过时 exit 1，所以可以直接当门禁用。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pdf_table_extract import pdfio, rules

# 验收区间。任一不满足 ⇒ 有阈值失准。
#
# ⚠ **区间必须按「人核实过的正确值」定，不能按「跑出来的观测值」定。**
#   第一版我把 rot 设成 0–3，因为实际就是 3 —— 于是永远✓通过。但那 3 页里有 2 页是已核实的
#   误报。照观测值定验收线 = 这个体检永远不会失败 = 等于没有。
#
# 每条后面写清核实依据；换一批语料时要重新人核，不能照抄。
EXPECT: dict[str, tuple[int, int, str]] = {
    "fail": (0, 0, "解析失败篇数"),
    "rot": (0, 1, "判需转正的页 —— **已渲染页面人眼核实**：只有 Bruhm p9 是真横向表"
                  "（Table 3 竖排占左半页，右半页是正文，所以占比只有 57%）。"
                  "F-002 修好后（加了「竖排里含 Table 标号」第三条判据），"
                  "Li p6 与 Williams p15 那两个误报已消除"),
    "ocr": (0, 0, "判 OCR 文字层的页 —— 这批全是现代排版 PDF，应为 0"),
    "pre": (0, 1, "判 preprint 的篇 —— **人核实过 Williams(cells from) 确是 preprint**"
                  "（34 页里 20 页含水印词 = 59%），判对了；多于 1 篇才算失准"),
}

# 已知会失败的项，以及对应的 finding。这样报告能自己说清"这次红是老问题还是新问题"。
# 修好一条就从这里删一条 —— 2026-07-26 删掉了 "rot"（F-002 已修，本项已转绿）。
KNOWN_FAILING: dict[str, str] = {}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pdf-dir", type=Path, required=True)
    ap.add_argument("--raw", action="store_true",
                    help="只打观测值、不做通过/不通过判定。**在留出集以外的语料上必须加这个** —— "
                         "EXPECT 的区间是按「随机的现代论文」定的，拿到定向挑选的语料上"
                         "（如 ~/Downloads/pte_threshold_corpus）会全红，那是预期行为、不是失准")
    args = ap.parse_args()

    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"✗ {args.pdf_dir} 下没有 PDF")
        return 2

    hdr = "%-46s %3s %3s %3s %6s %7s %7s %8s"
    print(hdr % ("PDF", "页", "表", "图", "仅引用", "转正页", "OCR层页", "preprint"))
    print("-" * 104)
    tot = dict(tables=0, figs=0, rot=0, ocr=0, pre=0, zero=0, fail=0)
    detail: dict[str, list[str]] = {"rot": [], "ocr": [], "pre": [], "fail": []}

    for p in pdfs:
        try:
            info = pdfio.read_doc(p)
            labels = pdfio.scan_labels(p)
        except Exception as exc:
            print(f"{p.stem[:46]:46s} ✗ {type(exc).__name__}: {exc}")
            tot["fail"] += 1
            detail["fail"].append(p.stem[:34])
            continue

        caps = pdfio.caption_labels(labels)
        refs = pdfio.referenced_only(labels)
        n_t = len({x.key for x in caps if x.kind == "table"})
        n_f = len({x.key for x in caps if x.kind == "figure"})
        rot = [x.page for x in info.pages if x.needs_rotation]
        ocr = [x.page for x in info.pages if x.ocr_text_layer]
        pre = pdfio.is_preprint(info)

        tot["tables"] += n_t
        tot["figs"] += n_f
        tot["rot"] += len(rot)
        tot["ocr"] += len(ocr)
        tot["pre"] += int(pre)
        tot["zero"] += int(n_t == 0 and n_f == 0)
        detail["rot"] += [f"{p.stem[:30]} p{x}" for x in rot]
        detail["ocr"] += [f"{p.stem[:30]} p{x}" for x in ocr]
        if pre:
            detail["pre"].append(p.stem[:30])

        print(hdr % (p.stem[:46], info.n_pages, n_t, n_f, len(refs),
                     len(rot), len(ocr), "是" if pre else ""))

    n = len(pdfs)
    print("-" * 104)
    print(f"共 {n} 篇，解析失败 {tot['fail']} 篇，零标号 {tot['zero']} 篇")
    print(f"合计：表 {tot['tables']} / 图 {tot['figs']}；"
          f"平均每篇 {tot['tables']/n:.1f} 张表、{tot['figs']/n:.1f} 张图")
    print()

    if args.raw:
        print("--raw：只报观测值，不做判定")
        print("（EXPECT 的区间是按「随机的现代论文」定的，只在留出集上有意义）")
        print("-" * 104)
        for key, (lo, hi, why) in EXPECT.items():
            print(f"  {why.split(chr(8212))[0].strip()}: {tot[key]}")
            if detail.get(key):
                print(f"      {detail[key][:6]}")
        return 0

    print("阈值体检")
    print("-" * 104)
    bad = new_bad = 0
    for key, (lo, hi, why) in EXPECT.items():
        v = tot[key]
        ok = lo <= v <= hi
        known = KNOWN_FAILING.get(key)
        if not ok:
            bad += 1
            if not known:
                new_bad += 1
        mark = "✓" if ok else ("▲" if known else "✗")
        print(f"  {mark} {why}")
        print(f"        实际 {v}，期望 {lo}–{hi}")
        if not ok and detail.get(key):
            print(f"        命中: {detail[key][:6]}")
        if not ok and known:
            print(f"        ▲ 这是**已知问题**，不是新退化：{known}")

    print()
    print("当前阈值：")
    print(f"  VERTICAL_LINES_THRESHOLD={rules.VERTICAL_LINES_THRESHOLD}  "
          f"VERTICAL_RATIO_THRESHOLD={rules.VERTICAL_RATIO_THRESHOLD}  "
          f"（转正判据，AGENTS.md 事实 2c）")
    print(f"  FULL_PAGE_IMAGE_COVER={rules.FULL_PAGE_IMAGE_COVER}  "
          f"OCR_LAYER_MIN_CHARS={rules.OCR_LAYER_MIN_CHARS}  （OCR 文字层，事实 2d）")
    print(f"  PREPRINT_MIN_PAGE_RATIO={pdfio.PREPRINT_MIN_PAGE_RATIO}  （preprint，事实 2e）")

    print()
    print('⚠ 本报告**检不到**（别把"跑过了"当成"验过了"）：')
    print("  · 图片路径的任何东西 —— --list 不跑抽取")
    print("  · CROP_PAD_RATIO / MIN_RIVAL_OVERLAP / MIN_IMAGE_COVER_H —— --list 根本不经过")
    print('  · **标号漏检** —— 没有 ground truth，漏了只显示成"这篇表少一点"，看起来完全合理')

    print()
    if new_bad:
        print(f"✗ {new_bad} 项**新失准** —— 有阈值掉进了没见过的中间档，先查清再往下走")
    elif bad:
        print(f"▲ {bad} 项失准，但全是已知问题（见上），没有新退化")
    else:
        print("✓ 全部通过")
    # 只有**新**失准才 exit 1 —— 已知问题天天红会让人对红色脱敏
    return 1 if new_bad else 0


if __name__ == "__main__":
    sys.exit(main())
