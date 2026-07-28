"""把 Zotero 扫出来的候选组装成一份**阈值验证集**，复制到 ~/Downloads/。

    conda run -n gemini python spike/build_threshold_corpus.py [--dry-run]

输入是 `spike/probe_find_testmaterial.py` 落盘的 `/tmp/pte_material.json`。

═══ 这不是留出集，别混用 ═══

`holdout/` 那 25 篇是**随机的一批没见过的论文**，所以在它上面算"误报率""命中率"有意义。

本语料是**按性质定向挑出来的**（专门找扫描件、专门找同页多表…），
**统计量天然是偏的** —— 在它上面算比例毫无意义，只能用来回答
「这条判据在这类样本上表现如何」这种**逐条**的问题。

名字里带 `thresholds` 就是为了提醒这一点。

═══ 各类挑多少、为什么 ═══

  ① 扫描件            全要        —— 最缺的正例（原本只有 2 个），且含 1909–1965 的古董
  ② 同页多张表        取 12       —— 原本 37 篇里只有 1 处；优先同页 >=3 张表的
  ③ 竖排方向 (0,1)    取 10       —— 优先**真的触发 270° 分支**的，其余当负例
  ④ 首页竖排水印      全要        —— arXiv 型 preprint，F-010 原本只有 1 个正例
  ⑤ 别的标号写法      每种取 2    —— Extended Data Table / Supplemental Table / Additional file
"""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

SRC_JSON = Path("/tmp/pte_material.json")
DEST = Path.home() / "Downloads" / "pte_threshold_corpus"

QUOTA = {"scanned": None, "multi_table": 12, "dir_up": 10, "p1_watermark": None, "alt_label": 6}

LABEL = {
    "scanned": ("扫描件", "FULL_PAGE_IMAGE_COVER + OCR_LAYER_MIN_CHARS 的**正例**（原本只有 2 个）"),
    "multi_table": ("同页多张表", "MIN_RIVAL_OVERLAP（caption 归属校验，原本单点拟合）"),
    "dir_up": ("竖排方向 (0,1)", "rotation_for_page 的 270° 分支（AGENTS.md 已知限制 #3，原本 0 样本）"),
    "p1_watermark": ("首页竖排水印", "F-010 arXiv 型 preprint（原本只有 1 个正例）"),
    "alt_label": ("别的标号写法", "F-009 DEFAULT_LABEL_CORE 的期刊变体"),
}


def score(kind: str, meta: str) -> int:
    """挑样本时的优先级：信息量大的排前面。"""
    try:
        v = ast.literal_eval(meta)
    except Exception:
        return 0
    if kind == "multi_table":          # 同页表数越多越有价值
        return max((n for _, n in v), default=0) * 10 + len(v)
    if kind == "dir_up":               # 竖排行数越多越可能触发
        return max((n for _, n in v), default=0)
    if kind == "scanned":              # 扫描页数越多越像真扫描件
        return len(v)
    if kind == "alt_label":
        return sum(n for _, n in v)
    return 1


def pick(kind: str, rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    quota = QUOTA.get(kind)
    rows = sorted(rows, key=lambda r: -score(kind, r[1]))
    if kind == "alt_label":
        # 不平均取 —— 按「这条写法还需不需要验」排优先级（结论见 findings.md F-009）：
        #   Extended Data Table  ← **最需要**。提议的正则里加了它，但 37 篇语料里 0 实例
        #                            （只有 Extended Data Fig），这个分支从未被真样本验过
        #   Supplemental Table   ← 需要。开发集只有 blood 一篇 1 次
        #   Web/Online Table     ← 需要。0 实例
        #   Additional file      ← 已决定**不加**（工具靠内嵌的 `Table SN` 已处理对），
        #                            留 1 篇当**负例**：证明不加是对的
        #   eTable               ← 已决定不加（无实例）；有的话留 1 篇
        PRIORITY = {"Extended Data Table": (0, 3), "Supplemental Table": (1, 2),
                    "Web Table / Online Table": (2, 2), "eTable (JAMA)": (3, 1),
                    "Additional file (BMC)": (4, 1)}
        by_conv: dict[str, list] = defaultdict(list)
        for p, m in rows:
            try:
                for name, _ in ast.literal_eval(m):
                    by_conv[name].append((p, m))
            except Exception:
                pass
        out, seen = [], set()
        for name in sorted(by_conv, key=lambda k: PRIORITY.get(k, (9, 1))[0]):
            take = PRIORITY.get(name, (9, 1))[1]
            for p, m in by_conv[name][:take]:
                if p not in seen:
                    seen.add(p)
                    out.append((p, m))
        return out[:quota] if quota else out
    return rows[:quota] if quota else rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not SRC_JSON.exists():
        print(f"✗ {SRC_JSON} 不存在 —— 先跑 spike/probe_find_testmaterial.py")
        return 2
    found = json.loads(SRC_JSON.read_text())

    chosen: dict[str, list[tuple[str, str]]] = {}
    for kind in LABEL:
        chosen[kind] = pick(kind, [tuple(r) for r in found.get(kind, [])])

    # 一篇可能同时满足多类 —— 记全部用途，只复制一次
    uses: dict[str, list[str]] = defaultdict(list)
    detail: dict[str, dict[str, str]] = defaultdict(dict)
    for kind, rows in chosen.items():
        for p, m in rows:
            uses[p].append(kind)
            detail[p][kind] = m

    print(f"选中 {len(uses)} 篇（去重后），各类：")
    for kind, rows in chosen.items():
        name, why = LABEL[kind]
        print(f"  {name:18s} {len(rows):3d} 篇 / 库里共 {len(found.get(kind, [])):3d} 篇   {why}")

    if args.dry_run:
        print("\n--dry-run，不复制")
        for p in sorted(uses):
            print(f"  {'+'.join(uses[p]):28s} {Path(p).name[:70]}")
        return 0

    DEST.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 阈值验证集（pte_threshold_corpus）",
        "",
        "从 `/Users/funanhe/Zotero/storage` 的 1940 篇里**按性质**挑出来的，",
        "专门用来验证 `holdout/` 那 25 篇覆盖不到的阈值。",
        "",
        "## ⚠ 这不是留出集，别混用",
        "",
        "`holdout/` 那 25 篇是**随机的一批没见过的论文**，在它上面算「误报率」有意义。",
        "本语料是**按性质定向挑的**（专门找扫描件、专门找同页多表…），**统计量天然偏**，",
        "算比例毫无意义。它只能回答「这条判据在这类样本上表现如何」这种**逐条**的问题。",
        "",
        "## 各文件对应验哪条阈值",
        "",
        "| 文件 | 验什么 | 扫描时测到的 |",
        "|---|---|---|",
    ]
    n_copy = 0
    for p in sorted(uses, key=lambda x: (uses[x], Path(x).name)):
        src = Path(p)
        if not src.exists():
            print(f"  ⚠ 源文件不存在，跳过: {src}")
            continue
        dst = DEST / src.name
        if dst.exists():                       # Zotero 有同名重复，加 key 区分
            dst = DEST / f"{src.parent.name}_{src.name}"
        shutil.copy2(src, dst)
        n_copy += 1
        kinds = "；".join(LABEL[k][0] for k in uses[p])
        meta = "；".join(f"{LABEL[k][0]}={detail[p][k][:44]}" for k in uses[p])
        lines.append(f"| `{dst.name}` | {kinds} | {meta} |")

    lines += [
        "",
        "## 各类的量与用途",
        "",
        "| 类别 | 本语料 | Zotero 全库 | 验哪条阈值 |",
        "|---|---|---|---|",
    ]
    for kind, rows in chosen.items():
        name, why = LABEL[kind]
        lines.append(f"| {name} | {len(rows)} | {len(found.get(kind, []))} | {why} |")
    lines += [
        "",
        "## 怎么用",
        "",
        "```bash",
        "# 阈值体检（复用 holdout 的报告，换个语料目录）",
        f"conda run -n gemini python holdout/report.py --pdf-dir {DEST}",
        "```",
        "",
        "**但注意**：`holdout/report.py` 的 `EXPECT` 验收区间是按「随机的现代论文」定的，",
        "在本语料上会大量报警 —— 那是**预期行为**（本语料就是专门挑异常样本挑出来的）。",
        "看的是**逐条的两侧观测值**，不是通过/不通过。",
    ]
    (DEST / "README.md").write_text("\n".join(lines) + "\n")
    print(f"\n✓ 复制 {n_copy} 篇到 {DEST}")
    print(f"  清单写在 {DEST / 'README.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
