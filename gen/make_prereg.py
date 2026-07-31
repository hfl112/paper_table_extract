#!/usr/bin/env python3
"""生成 `gen/prereg.csv` 的**草稿**。必须在跑抽取之前跑，跑完就冻结。

判定规则（每条都留下 `basis`，好让别人复核我是怎么定的）：

| 条件 | 结论 | 为什么可信 |
|---|---|---|
| caption 命中（**用工具自己的 matcher**，不是 grep） | `hit` / `caption` | caption 文本就是工具要匹配的输入，没有歧义 |
| **整页文字层**都没有这个词 | `miss` | 表是页的子集，页里没有 ⇒ 表里一定没有。这是**确定的负例** |
| 页里有词，且出现了只可能是**表头**的字符串（`Xenograft line` / `Response activity` / `Objective Response` …） | `hit` / `cell` | 这些串正文不会这么写 |
| 页里有词，但定位不到表头串 | **`?`** | PPTP 正文满篇都是 `response`，分不清在表里还是正文里 |

`?` 一条都不算断言。它的条数就是本次结论强度的上限 —— 事后才定下来的不计入。

⚠️ 这份是 **agent 读 PDF** 得出的，不是人核的。对应 `eval/gold.py` 的
`human` / `agreed` 分档，只能算 `agreed`。
"""

from __future__ import annotations

import csv
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz  # noqa: E402

from pdf_table_extract.rules import keyword_hits  # noqa: E402

B = Path.home() / "Downloads" / "pte_pptp_gen"
DEEP = ["pbc_26825", "pbc_22188", "pbc_22576", "pbc_22741", "s00280_011_1618_8",
        "0008_5472_can_16_0122", "1078_0432_ccr_18_2675", "blood_2016_03_707414",
        "pbc_22921", "j_pharmthera_2024_108742"]
HDR = {
    "xeno": ["Xenograft line", "Xenograft Line", "xenograft line"],
    "resp": ["Response activity", "Median group response", "Obj. Response",
             "Objective Response", "Objective response rate", "Responder"],
}
FIELDS = ("paper query label expect page expect_rows expect_cols expect_matched_on "
          "expect_confidence expect_stitch basis why").split()


def main() -> int:
    amap = {r["staged_name"]: r["article_id"]
            for r in csv.DictReader((B / "corpus.csv").open())}
    out: list[dict] = []
    for s in DEEP:
        d = fitz.open(B / f"deep10/{s}.pdf")
        caps = [r for r in csv.DictReader((B / "list97" / amap[s] / "captions.csv").open())
                if r["kind"] == "table" and r["present_in_pdf"] == "yes"]
        for r in caps:
            pg = int(r["page"])
            page = re.sub(r"\s+", " ", d[pg - 1].get_text())
            cap = r["caption_legend"]
            for q, kws, hk in (("response", ["response"], "resp"),
                               ("PDX;xenograft", ["xenograft", "PDX"], "xeno")):
                hdr = next((h for h in HDR[hk] if h in page), None)
                if keyword_hits(cap, kws):
                    e, mo, basis = "hit", "caption", "caption 命中（工具 matcher）"
                elif not keyword_hits(page, kws):
                    e, mo, basis = "miss", "", "整页文字层无该词 ⇒ 表内一定没有"
                elif hdr:
                    e, mo, basis = "hit", "cell", f"表头串「{hdr}」"
                else:
                    e, mo, basis = "?", "", "整页有该词但分不清在表内还是正文"
                out.append(dict(paper=s, query=q, label=r["label"], expect=e, page=pg,
                                expect_rows="", expect_cols="", expect_matched_on=mo,
                                expect_confidence="", expect_stitch="",
                                basis=basis, why=""))

    # ── 人核过的，写死确定值 ────────────────────────────────────────────
    for o in out:
        if o["paper"] == "pbc_22188" and o["label"] == "TABLE I":
            o.update(expect_rows="45", expect_cols="12", expect_confidence="high",
                     expect_matched_on="cell",
                     basis="人读 p2 渲染图逐行数：45 数据行 × 12 列；"
                           "表头 Xenograft line…Median group response…Response activity")
        if o["paper"] == "pbc_26825" and o["label"] == "TABLE 3" and o["page"] == 5:
            o["expect_stitch"] = "caption_continued"
            o["basis"] += "；p6 caption 为 `TABLE 3 (Continued)`，判据一该命中"
        # 两种新续表写法：判据一（要求字面括号）**一定不命中**，只能靠判据二兜。
        # 写 `?` 后缀表示"这是个预测，不作为形状断言"。
        if o["paper"] == "s00280_011_1618_8" and o["label"] == "Table 2" and o["page"] == 6:
            o["expect_stitch"] = "same_columns?"
            o["why"] = "p7 是 `Table 2 continued`（无括号），判据一必不中"
        if o["paper"] == "0008_5472_can_16_0122" and o["label"] == "Table 1" and o["page"] == 2:
            o["expect_stitch"] = "same_columns?"
            o["why"] = "p3 是 `Table 1 …(Cont'd )`，判据一必不中"

    p = Path("gen/prereg.csv")
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(out)

    c = Counter(o["expect"] for o in out)
    sure = c["hit"] + c["miss"]
    print(f"写出 {p}：{len(out)} 条 = hit {c['hit']} / miss {c['miss']} / ? {c['?']}")
    print(f"确定 {sure} 条，待定 {c['?']} 条。"
          f"**「?」不构成断言** —— 本次结论强度的上限就是这 {sure} 条。")
    print("\n待定的（跑完要靠人核补判，且要在 log 里标明是事后定的）:")
    for o in out:
        if o["expect"] == "?":
            print(f"   ? {o['paper']:<26}{o['query']:<14}{o['label']} p{o['page']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
