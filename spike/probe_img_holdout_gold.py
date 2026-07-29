"""留出集图片路径的**分母**：那 108 条「结构还原失败」里，到底有几条真是表？

背景：在留出集 24 篇上跑 `--all` 全量抽取，图片候选 202 条，结局是
  108  检出网格但结构还原失败
   83  闸门判为纯图表（跳过 OCR）
    5  出了 CSV 但是 1 行 0 列的退化输出
    3  出了非退化的表

**光看这个分布下不了任何结论** —— 108 条失败里，可能全是热图/多面板图
（单元格检测器在柱子和网格线上检出伪单元格，见 PDF 事实 #9b），那"失败"就是**正确行为**；
也可能有大量真表被漏掉，那就是**静默漏表**、违反铁律 #2 的精神。
这两种情况对策完全相反，而没有 ground truth 就分不开 —— 这正是
`AGENTS.md` 里那条「分母不对，比例就是假的」。

所以按用户的建议用 LLM 当**独立第二读者**给留出集造 gold。只问一个问题：
**这一块区域里有没有一张数据表？** 有的话报形状和首列。
不要求它逐格精确 —— 它自己也是一种 OCR、低分辨率上一样会错（这条写进 findings）。
它值钱的地方在于**跟我们的 paddle 管线不共享失败模式**，所以"是不是表"这个
判断可以当独立证据。

`eval/` 允许调 LLM（AGENTS.md 非目标里的明确例外），产品本体一行网络代码都没有。

用法（key 从 .env 读，绝不回显）：
    set -a; . /Users/funanhe/00_MyCode/idea_generator/.env; set +a
    conda run -n gemini python spike/probe_img_holdout_gold.py --n 15
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SURVEY = Path("/tmp/pte_img_survey")
PDF_DIR = Path("/Users/funanhe/Downloads/Zotero_Precancer_PDFs")
MODEL = "claude-opus-4-8"
DPI = 200

SCHEMA = {
    "type": "object",
    "properties": {
        "is_table": {"type": "boolean",
                     "description": "这块区域里有没有一张**数据表**（成行成列的文字/数值，"
                                    "有表头或行标签）。纯折线图/散点图/热图色块/流程图/"
                                    "显微照片都算没有"},
        "why": {"type": "string", "description": "一句话说明看到了什么"},
        "n_rows": {"type": "integer", "description": "数据行数，不含表头；不是表填 0"},
        "n_cols": {"type": "integer", "description": "列数；不是表填 0"},
        "header": {"type": "array", "items": {"type": "string"},
                   "description": "表头各列文字，从左到右；不是表留空数组"},
        "first_col": {"type": "array", "items": {"type": "string"},
                      "description": "首列（行标签列）从上到下的取值，最多前 12 个；"
                                     "不是表留空数组"},
    },
    "required": ["is_table", "why", "n_rows", "n_cols", "header", "first_col"],
    "additionalProperties": False,
}

PROMPT = """这是一篇英文文献 PDF 的第 {page} 页整页原图。

页面上有一个标号为 `{label}` 的图/表。请只看**它**，回答：它里面有没有一张数据表？

判断标准（严格一点）：
- **是表**：成行成列排布的文字或数值，有表头行或行标签列。
  期刊常把「表格 + 条形图」拼成一张图，这种**算是表**（左半那半）。
- **不是表**：折线图、散点图、箱线图、生存曲线、流程示意图、显微/组织照片、
  纯色块热图（只有颜色没有文字数值）、基因组轨迹图。
- 图例（legend）、坐标轴标签、图内的小注释**不算表**。

是表的话，报出它的行数、列数、表头、以及首列从上到下的取值（最多 12 个）。
按原文照抄，不要解读、不要补全缩写。
"""


def collect() -> list[dict]:
    """从留出集普查产物里收集全部图片候选，按结局分类。"""
    from pdf_table_extract.emit import sanitize_name

    stems: dict[str, Path] = {}
    for p in PDF_DIR.glob("*.pdf"):
        stems.setdefault(p.stem, p)
        stems.setdefault(sanitize_name(p.stem), p)
    out = []
    for d in sorted(x for x in SURVEY.iterdir() if x.is_dir()):
        pdf = stems.get(d.name)
        man = d / "manifest.csv"
        if pdf is None or not man.exists():
            continue
        for r in csv.DictReader(man.open(newline="")):
            if r.get("source_type") != "image":
                continue
            notes = r.get("notes", "")
            rows, cols = int(r.get("n_rows") or 0), int(r.get("n_cols") or 0)
            if r.get("csv_path"):
                kind = "OK" if (rows >= 2 and cols >= 2) else "DEGEN"
            elif "结构还原失败" in notes:
                kind = "FAIL"
            elif "未检出表格网格" in notes:
                kind = "GATED"
            else:
                continue
            m = re.search(r"检出 (\d+) 个|单元格 (\d+) 个", notes)
            out.append({
                "kind": kind, "paper": d.name, "pdf": pdf,
                "page": int(r["page"]), "label": r.get("label") or r["table_id"],
                "table_id": r["table_id"], "shape": f"{rows}x{cols}",
                "cells": int(m.group(1) or m.group(2)) if m else 0,
            })
    return out


def page_png(pdf: Path, page: int) -> bytes:
    import fitz

    doc = fitz.open(str(pdf))
    try:
        return doc[page - 1].get_pixmap(dpi=DPI).tobytes("png")
    finally:
        doc.close()


def ask(client, t: dict) -> dict:
    img = base64.standard_b64encode(page_png(t["pdf"], t["page"])).decode()
    msg = client.messages.create(
        model=MODEL, max_tokens=4000,
        tools=[{"name": "report", "description": "报告判读结果",
                "input_schema": SCHEMA}],
        tool_choice={"type": "tool", "name": "report"},
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": img}},
            {"type": "text", "text": PROMPT.format(page=t["page"], label=t["label"])},
        ]}],
    )
    for b in msg.content:
        if b.type == "tool_use":
            return b.input
    return {"is_table": False, "why": "(无返回)", "n_rows": 0, "n_cols": 0,
            "header": [], "first_col": []}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="从 FAIL 里抽多少条")
    ap.add_argument("--out", type=Path, default=Path("/tmp/pte_img_holdout_gold.json"))
    args = ap.parse_args()

    import anthropic

    all_c = collect()
    fails = sorted([c for c in all_c if c["kind"] == "FAIL"], key=lambda c: c["cells"])
    # **分层抽样**：按检出单元格数排序后等距取，避免只抽到某一档。
    # 直接取前 N 会全是小网格、取后 N 会全是 300 格封顶的热图，两种都会得出错误比例。
    step = max(1, len(fails) // args.n)
    sample = fails[::step][: args.n]
    picked = ([c for c in all_c if c["kind"] == "OK"]
              + [c for c in all_c if c["kind"] == "DEGEN"] + sample)
    print(f"留出集图片候选 {len(all_c)} 条："
          f"OK {sum(c['kind'] == 'OK' for c in all_c)} / "
          f"DEGEN {sum(c['kind'] == 'DEGEN' for c in all_c)} / "
          f"FAIL {len(fails)} / GATED {sum(c['kind'] == 'GATED' for c in all_c)}")
    print(f"送 LLM 判读 {len(picked)} 条（FAIL 分层抽 {len(sample)} 条，"
          f"单元格数 {sample[0]['cells']}~{sample[-1]['cells']}）\n")

    client = anthropic.Anthropic()
    with ThreadPoolExecutor(max_workers=4) as ex:
        verdicts = list(ex.map(lambda t: ask(client, t), picked))

    res = []
    for t, v in zip(picked, verdicts):
        res.append({**{k: str(t[k]) for k in ("kind", "paper", "page", "label",
                                              "shape", "cells")}, **v})
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=1))

    for kind in ("OK", "DEGEN", "FAIL"):
        grp = [r for r in res if r["kind"] == kind]
        if not grp:
            continue
        yes = sum(1 for r in grp if r["is_table"])
        print(f"=== {kind}（{len(grp)} 条）→ LLM 判定是表的 **{yes}** 条")
        for r in grp:
            mark = "表" if r["is_table"] else "非表"
            print(f"  [{mark}] {r['paper'][:30]:<30} p{r['page']:<3} {r['label'][:18]:<18} "
                  f"工具 {r['shape']:>7}/{r['cells']}格  LLM {r['n_rows']}x{r['n_cols']}"
                  f"  {r['why'][:52]}")
            if r["is_table"] and r["first_col"]:
                print(f"        首列: {r['first_col'][:8]}")
    print(f"\n明细写到 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
