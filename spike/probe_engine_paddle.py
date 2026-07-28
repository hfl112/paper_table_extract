"""第 3 步：paddle OCR（图片表）的准确性 —— 图片路径的第一次真实测量。

    conda run -n gemini python spike/probe_engine_paddle.py --variants   # 放大×裁剪矩阵
    conda run -n gemini python spike/probe_engine_paddle.py              # gold 打分

═══ 为什么这一步最不确定 ═══

图片表现在**连对齐都做不到** —— `pbc_24724 fig_1` 锚点重复 8 组（`PD1` 出现 8 次当锚点）、
`pbc_21296 fig_1` 的瘤系名被 OCR 读成 `K1-18` / `Rhabdold`。
上一版报告把它们写成 `0`，那是**错误的表述**：它们是"测不出"，不是"全错"。

三级对齐，**报告必须注明用的哪一级**：

  L1 精确锚点  —— 大概率失败
  L2 模糊锚点  —— 归一化后编辑距离 <=2 且 <=30% 长度，贪心一一配对且**要求最优唯一**
  L3 纯聚合    —— 放弃逐格，只报行数/列数/**响应码多重集重叠率**

**L3 对目标 (b) 仍然有用**："这张图里的响应码，我们抽对了几个（不管位置）"本身就是有意义的数字。

═══ 先解开一个文档与代码的矛盾 ═══

`AGENTS.md` 事实 9d 与 `engine_paddle.py:249-250` 都记「裁剪 + **4x** → `45x6`」，
但 `upscale_factor(120)` 实际返回 **3**（`ceil(300/120)`），`tests/test_rules.py:605` 把 3 钉死了。
→ **文档记的实验配置，生产根本没在跑。** `--variants` 用 `spike/lowres/` 现成的 16 张图复现。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pdf_table_extract import quiet  # noqa: E402

quiet.silence()  # 必须在 import 引擎之前，否则 paddle 日志淹没输出

from eval import gold as gold_io  # noqa: E402
from pdf_table_extract import engine_paddle  # noqa: E402

GOLD_DIR = Path("/Users/funanhe/pte_gold")
LOWRES = ROOT / "spike" / "lowres"

# 目标 (b) 关心的响应码。L3 聚合用 —— 只数多重集重叠，不管位置。
RESP_CODES = {"cr", "mcr", "pr", "sd", "pd", "pd1", "pd2", "ne", "int", "low", "high"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    for a, b in (("−", "-"), ("–", "-"), ("—", "-")):
        s = s.replace(a, b)
    return re.sub(r"\s+", "", s).lower()


def edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def fuzzy_map(tool_keys: list[str], gold_keys: list[str]) -> tuple[dict[str, str], list[str]]:
    """L2：把工具侧锚点模糊配到 gold 锚点。返回 (映射, 被拒的原因清单)。

    **要求最优唯一** —— 次优距离必须严格更大，否则拒绝该行。
    OCR 把 `KT-13` 读成 `K1-18` 时，它跟 `KT-16` 的距离可能一样近，
    这种情况配上去就是瞎猜，宁可不配。
    """
    used: set[str] = set()
    out: dict[str, str] = {}
    rejects: list[str] = []
    for tk in tool_keys:
        n = norm(tk)
        if not n:
            continue
        scored = sorted(
            ((edit_distance(n, norm(gk)), gk) for gk in gold_keys if gk not in used),
            key=lambda x: x[0],
        )
        if not scored:
            continue
        d, best = scored[0]
        limit = max(1, int(len(n) * 0.3))
        if d > min(2, limit):
            rejects.append(f"{tk!r} 最近的也差 {d}")
            continue
        if len(scored) > 1 and scored[1][0] == d:
            rejects.append(f"{tk!r} 最优不唯一（{best!r} 与 {scored[1][1]!r} 距离都是 {d}）")
            continue
        out[tk] = best
        used.add(best)
    return out, rejects


def cmd_variants(_args) -> int:
    imgs = sorted(LOWRES.glob("*.png"))
    if not imgs:
        print(f"{LOWRES} 下没有图")
        return 1
    print("放大 × 裁剪 变体矩阵（`spike/lowres/` 现成对照图，不用重跑取图链路）")
    print("=" * 96)
    print("生产实测：upscale_factor(120dpi) = %d，**不是文档记的 4**"
          % engine_paddle.upscale_factor(120))
    print("-" * 96)
    print("%-34s %10s %10s  %s" % ("图", "还原形状", "首列样例", "表头样例"))
    print("-" * 96)
    for p in imgs:
        try:
            rows = engine_paddle.recognize_table(p)
        except Exception as e:
            print("%-34s  还原抛异常: %s" % (p.name[:34], str(e)[:40]))
            continue
        if not rows:
            print("%-34s %10s" % (p.name[:34], "**失败(0表)**"))
            continue
        shape = "%dx%d" % (len(rows), max(len(r) for r in rows))
        first = rows[1][0][:14] if len(rows) > 1 and rows[1] else ""
        head = " | ".join(c[:10] for c in rows[0][:3])
        print("%-34s %10s %10r  %s" % (p.name[:34], shape, first, head[:34]))
    print("-" * 96)
    return 0


def score_image_gold(g, rows: list[list[str]]) -> None:
    """L1 → L2 → L3 逐级降级，注明用的哪一级。"""
    name = g.path.stem
    tool_keys, _ = gold_io.anchors_of(
        gold_io.apply_index_specs(rows, g.anchor_cols, g.check_cols),
        g.anchor_cols, g.section_rows)
    gold_keys = list(g.values)

    # L1
    res = gold_io.score_against(g, rows, {"human", "agreed"})
    if not res.anchor_not_unique and res.totals["scored"] and not res.missing_rows:
        t = res.totals
        print(f"{name:26s} **L1 精确**  {t['match']}/{t['scored']} = "
              f"{t['match'] / t['scored']:.1%}")
        return

    why_l1 = (f"锚点重复 {len(res.anchor_not_unique)} 组" if res.anchor_not_unique
              else f"漏 {len(res.missing_rows)} 行")
    # L2
    mapping, rejects = fuzzy_map(tool_keys, gold_keys)
    rate = len(mapping) / max(1, len(gold_keys))
    if rate >= 0.8:
        idx = gold_io.apply_index_specs(rows, g.anchor_cols, g.check_cols)
        acol = idx[0].index(g.anchor_cols[0])
        relabelled = [idx[0]] + [
            ([mapping.get(r[acol], r[acol])] + r[1:]) if acol == 0 else r
            for r in idx[1:]
        ]
        r2 = gold_io.score_against(g, relabelled, {"human", "agreed"})
        t = r2.totals
        got = f"{t['match']}/{t['scored']} = {t['match'] / t['scored']:.1%}" if t["scored"] else "—"
        print(f"{name:26s} **L2 模糊**（L1 不行：{why_l1}）配对率 {rate:.0%}  {got}")
        for rj in rejects[:3]:
            print(f"      拒配: {rj}")
        return

    # L3
    print(f"{name:26s} **L3 聚合**（L1 不行：{why_l1}；L2 配对率仅 {rate:.0%}）")
    print(f"      行数 工具 {len(rows) - 1} / gold 期望 {g.expect_data_rows}"
          f"   列数 工具 {max(len(r) for r in rows)} / gold 期望 {g.expect_cols}")
    tool_codes = [norm(c) for r in rows[1:] for c in r if norm(c) in RESP_CODES]
    gold_codes = [norm(v) for d in g.values.values() for v in d.values() if norm(v) in RESP_CODES]
    from collections import Counter
    tc, gc = Counter(tool_codes), Counter(gold_codes)
    overlap = sum((tc & gc).values())
    print(f"      **响应码多重集重叠 {overlap} / gold {sum(gc.values())} "
          f"= {overlap / max(1, sum(gc.values())):.0%}**（不管位置，只看数量）")
    blanks = sum(1 for r in rows[1:] if not any(c.strip() for c in r))
    if blanks:
        print(f"      全空行 {blanks} 个 —— 根因 `engine_paddle._html_to_rows`（保留全空字符串的 <tr>）")


def cmd_score(_args) -> int:
    golds = sorted(GOLD_DIR.glob("*.gold")) if GOLD_DIR.exists() else []
    print("第 3 步：paddle OCR 在 gold 上的分（三级对齐，注明用的哪级）")
    print("=" * 96)
    for gp in golds:
        g = gold_io.load(gp)
        out_csv = ROOT / "1.output" / Path(g.pdf).stem / g.csv_name
        if not out_csv.exists():
            continue
        man = out_csv.parent / "manifest.csv"
        stype = "?"
        if man.exists():
            for r in csv.DictReader(man.open(newline="")):
                if Path(r.get("csv_path") or "").name == g.csv_name:
                    stype = r.get("source_type", "?")
                    break
        if stype != "image":
            continue
        rows = list(csv.reader(out_csv.open(newline="")))
        score_image_gold(g, rows)
    print("-" * 96)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", action="store_true", help="跑放大×裁剪矩阵")
    args = ap.parse_args()
    return cmd_variants(args) if args.variants else cmd_score(args)


if __name__ == "__main__":
    sys.exit(main())
