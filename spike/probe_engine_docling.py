"""第 1 步：docling（文字表主干）的准确性统计。

    conda run -n gemini python spike/probe_engine_docling.py

**一次只测一个方法。** 这个探针只管 docling，plumber 和 paddle 各有自己的探针。
上一版把三个引擎混在一张表里输出，结果 plumber 那列全是 `—`、OCR 两张全是 `0` ——
**那不是"分低"，是没给它们能对齐的打分方式，等于没测却看着像测了。**

═══ 两条报告纪律 ═══

1. **「测不出」和「0 分」必须分开写。** 图片表现在是「锚点对不齐 ⇒ 无法打分」，
   不是「全错」。前者要写原因，后者才是分数。
2. **每一处漏行都要能归因。** 漏行不写原因，报告就退化成"有问题"三个字。
   本探针用规则自动归因到 F-001 / F-016 / F-017，**归不了的显式标 `原因不明`** ——
   那才是需要人去查的。

═══ 归因规则（靠 gold 锚点与实际锚点的差集比对，不靠人工贴标签）═══

对每个「漏行」M，去「多出行」E 里找能解释它的：

- **F-001 行合并**：存在 E 同时包含两个漏行 M1、M2 的锚点片段
  （`ALL-8 ALL-16` 同时含 `ALL-8` 和 `ALL-16`）
- **F-017 断行连字符**：E 去掉 `- ` 之后等于 M
  （`EPZ011989 + cyclophos- phamide` → `EPZ011989 + cyclophosphamide`）
- **F-016 脚注上标**：E 去掉尾部孤立标记（空格 + 单个字母/数字）之后等于 M
  （`P+E 1` → `P+E`）

═══ 非对称契约 ═══

打分走 `eval/gold.score_against`，它只返回我们这边的值。本探针**不打印 gold 的期望值**。
锚点本身来自 gold，属于"行的标识"而非"格子的答案"，打印它是归因所必需的。
"""

from __future__ import annotations

import csv
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import gold as gold_io  # noqa: E402

GOLD_DIR = Path("/Users/funanhe/pte_gold")

_HYPHEN_JOIN = re.compile(r"([a-z])-\s+([a-z])")
_TAIL_MARK = re.compile(r"\s+[0-9a-z]\Z")


def _dehyphen(s: str) -> str:
    return _HYPHEN_JOIN.sub(r"\1\2", s)


def attribute(missing: list[str], extra: list[str]) -> dict[str, str]:
    """把每个漏行归因到一条 finding。归不了的标 `原因不明`。"""
    out: dict[str, str] = {}
    for m in missing:
        why = None
        for e in extra:
            if _dehyphen(e) == m:
                why = "F-017 断行连字符（`x- y` 没接回）"
                break
            if _TAIL_MARK.sub("", e) == m:
                why = "F-016 脚注上标编进文字层"
                break
        if why is None:
            # F-001：两个漏行被并进同一个多出行
            for e in extra:
                others = [x for x in missing if x != m]
                if m.split(gold_io.ANCHOR_SEP)[-1] in e and any(
                    o.split(gold_io.ANCHOR_SEP)[-1] in e for o in others
                ):
                    why = "F-001 两个数据行被压成一格"
                    break
        out[m] = why or "**原因不明（需人工查）**"
    return out


def source_type_of(csv_path: Path) -> str:
    man = csv_path.parent / "manifest.csv"
    if not man.exists():
        return "?"
    for r in csv.DictReader(man.open(newline="")):
        if Path(r.get("csv_path") or "").name == csv_path.name:
            return r.get("source_type", "?")
    return "?"


def main() -> int:
    golds = sorted(GOLD_DIR.glob("*.gold")) if GOLD_DIR.exists() else []
    if not golds:
        print(f"{GOLD_DIR} 下没有 gold")
        return 1

    print("第 1 步：docling 准确性（gold 打分）")
    print("=" * 104)
    print("%-24s %-6s %-30s %s" % ("gold", "类型", "human 档", "human+agreed 档"))
    print("-" * 104)

    rows_out = []
    for gp in golds:
        g = gold_io.load(gp)
        csv_path = ROOT / "1.output" / Path(g.pdf).stem / g.csv_name
        if not csv_path.exists():
            print(f"{gp.stem:24s} 找不到产物 {csv_path}")
            continue
        rows = list(csv.reader(csv_path.open(newline="")))
        stype = source_type_of(csv_path)

        cells = {}
        for tier, srcs in (("human", {"human"}), ("both", {"human", "agreed"})):
            res = gold_io.score_against(g, rows, srcs)
            t = res.totals
            if res.anchor_not_unique:
                cells[tier] = ("无法对齐", f"锚点重复 {len(res.anchor_not_unique)} 组")
            elif t["scored"] == 0 and res.missing_rows:
                cells[tier] = ("无法对齐", f"gold 的 {len(res.missing_rows)} 个锚点全找不到")
            elif t["scored"] == 0:
                cells[tier] = ("未标注", "该档没有可计分的格子")
            else:
                cells[tier] = (f"{t['match']}/{t['scored']} = {t['match'] / t['scored']:.1%}",
                               f"漏{len(res.missing_rows)} 多{len(res.extra_rows)}")
            if tier == "both":
                rows_out.append((gp.stem, stype, g, res))

        print("%-24s %-6s %-30s %s" % (
            gp.stem[:24], stype,
            f"{cells['human'][0]}  ({cells['human'][1]})",
            f"{cells['both'][0]}  ({cells['both'][1]})"))

    print("-" * 104)
    print("**「无法对齐」不是 0 分** —— 是锚点配不上、根本没法逐格比，原因见括号。")

    print()
    print("漏行归因（human+agreed 档）")
    print("=" * 104)
    any_unknown = False
    for stem, stype, g, res in rows_out:
        if not res.missing_rows:
            print(f"{stem:24s} 无漏行 ✓")
            continue
        if res.anchor_not_unique:
            print(f"{stem:24s} 锚点重复 {len(res.anchor_not_unique)} 组 → 归因无意义，"
                  f"先解决对齐（{stype} 路径）")
            continue
        att = attribute(res.missing_rows, res.extra_rows)
        print(f"{stem:24s} 漏 {len(res.missing_rows)} 行：")
        for m, why in att.items():
            if "原因不明" in why:
                any_unknown = True
            print(f"      {m[:44]:46s} → {why}")

    print("-" * 104)
    if any_unknown:
        print("⚠ 有「原因不明」的漏行 —— **第 1 步验收不通过**，先查清再往下走。")
        return 1
    print("✓ 全部漏行都已归因，无「原因不明」")
    return 0


if __name__ == "__main__":
    sys.exit(main())
