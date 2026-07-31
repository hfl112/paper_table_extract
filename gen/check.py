#!/usr/bin/env python3
"""预登记 ↔ 实际产出对账。

    python gen/check.py --outdir ~/Downloads/pte_pptp_gen/out_noocr --prereg gen/prereg.csv

**为什么不用 `eval/checks.py regression`。** 方向相反，而且它有两个硬假设对不上：

1. 它断言的是「行为没变」—— 拿产出对**现状快照**（`expected.csv` 里钉着已知错误的当前值）。
   本次要断言的是「行为符合**跑之前**写下的预期」，没有现成的东西做这件事。
2. 它的 docstring 明说假设「每个 prefix 只跑一条命令，不会出现多 query」，`manifest_row()`
   返回第一条匹配行。本次每篇跑两组关键词 ⇒ `matched_on` 的断言会静默取到先跑的那一组。
   而 2 组 × 10 篇 × 最多 9 张表，**逐 query 的负例正是本次的主要价值**。

跟 `eval/checks.py detect` 一样：**永远 exit 0**，是报告不是门禁。

预登记的三态 `expect`（二值不够用）：
  `hit`      —— 必须有 manifest 行且有 CSV
  `fail_row` —— 必须有 manifest 行但 `csv_path` 为空（铁律 #2：caption 命中却抽不出，
                要留失败行。本次约四成有意思的行是这一类 —— 每个命中关键词的图 legend）
  `miss`     —— 压根不该有行
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

HDR = ("paper query label expect page expect_rows expect_cols "
       "expect_matched_on expect_confidence expect_stitch basis why").split()


def norm_label(s: str) -> str:
    """`TABLE I.` / `Table I` / `table_i` → `table i`，好让预登记与 manifest 对得上。"""
    s = re.sub(r"[_.]+", " ", (s or "").strip().lower())
    return re.sub(r"\s+", " ", s).strip()


def load_prereg(p: Path) -> list[dict]:
    rows = list(csv.DictReader(p.open()))
    missing = [r for r in rows if not (r.get("basis") or "").strip()]
    if missing:
        # 反循环装置：basis 记录这条预期是**怎么来的**（人读第几页渲染图 / 文字层 /
        # expected.csv）。空 basis 的行直接丢掉，**不是改正** —— 这是 holdout/report.py
        # 的 EXPECT docstring 用真实教训记下的规矩：照观测值定的验收线永远不会失败。
        print(f"▲ 丢弃 {len(missing)} 条 basis 为空的预登记行（不可追溯，不作为依据）")
        for r in missing:
            print(f"    {r['paper']} / {r['query']} / {r['label']}")
        rows = [r for r in rows if (r.get("basis") or "").strip()]
    return rows


def load_manifest(outdir: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for d in sorted(p for p in outdir.iterdir() if p.is_dir()):
        m = d / "manifest.csv"
        if m.exists():
            out[d.name] = list(csv.DictReader(m.open()))
    return out


def data_rows(csv_path: Path) -> int:
    """数据行数 = 文件行数 - 1（表头）。

    注意 `eval/expected.csv` 的 `rows` 列是**含表头的文件行数**（`checks.py` 用
    `sum(1 for _ in path.open())`）。预登记里写的是**数据行数**，别照抄那个约定，
    否则每个数都差 1。
    """
    if not csv_path.exists():
        return -1
    with csv_path.open() as f:
        return max(sum(1 for _ in f) - 1, 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--prereg", type=Path, default=Path("gen/prereg.csv"))
    a = ap.parse_args()

    pre = load_prereg(a.prereg) if a.prereg.exists() else []
    man = load_manifest(a.outdir)
    if not man:
        print(f"✗ {a.outdir} 下没有 manifest.csv")
        return 0

    # ── 索引 manifest：(篇, query, 规范化 label) → 行 ───────────────────
    idx: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for paper, rows in man.items():
        for r in rows:
            idx[(paper, r["query"], norm_label(r["label"] or r["table_id"]))].append(r)

    buckets: dict[str, list[str]] = defaultdict(list)
    n_checked = 0
    n_unknown: list[str] = []
    blank_rows = 0

    for e in pre:
        key = (e["paper"], e["query"], norm_label(e["label"]))
        got = idx.get(key, [])
        want = (e["expect"] or "").strip()
        tag = f"{e['paper']:<26} {e['query']:<14} {e['label']:<14}"
        if want == "?":
            # 跑之前就判不了的，**不构成断言**，不算对也不算错。只统计条数：
            # 它是本次结论强度的上限。
            n_unknown.append(f"{tag} 出了 {len(got)} 行" if got else f"{tag} 没出")
            continue
        n_checked += 1

        with_csv = [r for r in got if (r["csv_path"] or "").strip()]
        fail_only = [r for r in got if not (r["csv_path"] or "").strip()]

        if want == "hit":
            if not got:
                buckets["1_漏抽"].append(f"{tag} 预期命中，manifest 里**一行都没有**")
            elif not with_csv:
                buckets["1_漏抽"].append(
                    f"{tag} 预期出表，只有失败行: {fail_only[0]['notes'][:70]}")
            else:
                r = with_csv[0]
                got_rows = data_rows(a.outdir / e["paper"] / Path(r["csv_path"]).name)
                for fld, col, val in (
                    ("expect_rows", "数据行", got_rows),
                    ("expect_cols", "列数", r["n_cols"]),
                    ("expect_matched_on", "matched_on", r["matched_on"]),
                    ("expect_confidence", "confidence", r["confidence"]),
                ):
                    w = (e.get(fld) or "").strip()
                    if not w:
                        blank_rows += 1
                        continue
                    if str(w) != str(val):
                        buckets["3_形状或自评不符"].append(
                            f"{tag} {col}: 预期 {w} 实际 {val}")
                w = (e.get("expect_stitch") or "").strip()
                if w and not w.endswith("?") and w not in (r["notes"] or ""):
                    buckets["3_形状或自评不符"].append(
                        f"{tag} 拼接: 预期 {w}，notes={r['notes'][:60]!r}")
        elif want == "fail_row":
            if not got:
                buckets["4_该留失败行却什么都没有"].append(
                    f"{tag} 违反铁律 #2：caption 命中却连失败行都没有")
            elif with_csv:
                buckets["2_误报"].append(
                    f"{tag} 预期只留失败行，却真出了表 {with_csv[0]['csv_path']}")
        elif want == "miss":
            if got:
                buckets["2_误报"].append(
                    f"{tag} 预期不该有行，却有 {len(got)} 行"
                    f"（{got[0]['matched_on']} / {got[0]['matched_keywords']}）")
        else:
            buckets["0_预登记本身有问题"].append(f"{tag} expect={want!r} 不认识")

    # ── 结构检查（预登记之外，纯看产出自身是否自洽）────────────────────
    for paper, rows in man.items():
        seen_csv = set()
        for r in rows:
            cp = (r["csv_path"] or "").strip()
            if not cp:
                continue
            f = a.outdir / paper / Path(cp).name
            seen_csv.add(Path(cp).name)
            if not f.exists():
                buckets["5_manifest 指向不存在的文件"].append(f"{paper} {cp}")
                continue
            # F-019：`1 行 0 列` 之类的退化 CSV —— 绕过铁律 #2 的失败行，看起来像成功。
            # `eval/checks.py` 的 check_blank_rows 从 rows[1:] 开始遍历，看不见它。
            with f.open() as fh:
                grid = list(csv.reader(fh))
            n_data = len(grid) - 1
            n_col = max((len(g) for g in grid), default=0)
            body_chars = sum(len(c.strip()) for g in grid[1:] for c in g)
            if n_data <= 0 or n_col <= 1 or body_chars == 0:
                buckets["6_退化CSV"].append(
                    f"{paper} {cp}: {n_data} 数据行 × {n_col} 列，正文字符 {body_chars}"
                    f"  confidence={r['confidence']} notes={r['notes'][:40]!r}")
            # 错而不举手：high 且 notes 空 —— 这一档错了最危险
            if r["confidence"] == "high" and not (r["notes"] or "").strip():
                buckets["7_high且notes空(未举手)"].append(
                    f"{paper} {cp}  {r['n_rows']}x{r['n_cols']}  {r['matched_on']}")
        for f in sorted((a.outdir / paper).glob("p*_*.csv")):
            if f.name not in seen_csv:
                buckets["5_manifest 指向不存在的文件"].append(
                    f"{paper} {f.name} 在盘上但 manifest 里没有对应行")

    # ── 输出 ───────────────────────────────────────────────────────────
    print(f"\n对账 {a.outdir}")
    print(f"预登记 {len(pre)} 条：构成断言的 {n_checked} 条，"
          f"跑前判不了的 {len(n_unknown)} 条（不算对错，是结论强度的上限）")
    print(f"manifest 篇数 {len(man)}，总行数 {sum(len(v) for v in man.values())}")
    if blank_rows:
        print(f"▲ 预登记里有 {blank_rows} 个字段留空未核 —— 这部分不构成断言")
    total = 0
    for k in sorted(buckets):
        v = buckets[k]
        total += len(v)
        print(f"\n── {k[2:]}（{len(v)}）")
        for line in v:
            print(f"   {line}")
    if not total:
        print("\n✓ 全部符合预登记，且无结构问题")
    print(f"\nVERDICT items={total} 漏抽={len(buckets['1_漏抽'])} "
          f"误报={len(buckets['2_误报'])} 退化={len(buckets['6_退化CSV'])} "
          f"未举手={len(buckets['7_high且notes空(未举手)'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
