"""第 2 步：pdfplumber 在框内抽格的准确性 —— 第一次逐格量它。

    conda run -n gemini python spike/probe_engine_plumber.py

═══ 为什么此前从没量过 ═══

`engine_plumber.extract_in_bbox`（L34-67）**本来就返回完整二维网格**，但被
`columns_in_bbox`（L70-75）`max(len(r) for r in rows)` 降成一个 int 就扔了。
生产里 plumber 只贡献"列数第二意见"这一个标量，`log.md` 里**没有任何逐格实测**。

而实测发现它一直被低估：`pbc_28772` p3 抽出 83 行看着像垃圾，其实是
**38 个数据行中间夹着空行** + 表头拆成两行，内容与 docling **逐字相同**。

**更关键的**：`pbc_21296` p4 上它**正确抽出了 docling 丢掉的 `ALL-8` / `ALL-16` 两行**
（docling 把它们压成一格，就是 F-001）。反过来它在 `ALL-7` 那行把 `10.8` 切成了
`1` + `0.8`，而 docling 是对的。**两个引擎各有胜负。**

═══ 归一化：三道，且必须坐标无关 ═══

plumber 的行**没有坐标**，所以套不了 `probe_row_bands.py` 那个「用 docling cell 顶边划
表头区」的办法。改用**内容定位**：

  1. 丢掉全空行（实测 83→45、48→46、53→27）
  2. **按 gold 锚点值找出锚点列在第几列** —— 逐列数"有多少格能匹配上某个锚点分量"，取最高的
  3. **第一个匹配上的行 = 首个数据行**，它上面的全是表头 → 按列纵向拼接成一行表头

第 2 步用到 gold 的**锚点值**（不是格子答案）来定位列。这会给引擎在"列识别"上开后门，
所以**发现的列下标必须打印出来**，让人看得见这个让步。

═══ 不许调参凑 ═══

归一化后行数若仍对不上人核真值，**标「无法归一化」并写原因**，不要调参数把数字凑对。
`findings.md` F-013 记过这个坑。
"""

from __future__ import annotations

import csv
import pickle
import sys
import unicodedata
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval import geom, gold as gold_io  # noqa: E402
from pdf_table_extract import engine_plumber  # noqa: E402
from pdf_table_extract.models import Rect  # noqa: E402

GOLD_DIR = Path("/Users/funanhe/pte_gold")
CACHE = Path("/tmp/pte_rowbands_cache")

# 人核真值（单页），来自 AGENTS.md 测试料表 + 本轮渲染图人眼复核
TRUTH = {("pbc_21296", 4): 44, ("pbc_28772", 3): 38, ("pbc_26870", 18): 26}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    for a, b in (("−", "-"), ("–", "-"), ("—", "-")):
        s = s.replace(a, b)
    return "".join(s.split()).lower()


def find_anchor_col(rows: list[list[str]], anchor_vals: set[str]) -> tuple[int, int]:
    """逐列数「有多少格能匹配某个锚点分量」，返回 (最佳列下标, 命中数)。"""
    best, best_n = -1, 0
    ncol = max((len(r) for r in rows), default=0)
    for j in range(ncol):
        n = sum(1 for r in rows if j < len(r) and norm(r[j]) in anchor_vals)
        if n > best_n:
            best, best_n = j, n
    return best, best_n


def is_section_like(row: list[str]) -> bool:
    return bool(row) and bool(row[0].strip()) and not any(c.strip() for c in row[1:])


def normalize_grid(
    raw: list[list[str]], anchor_vals: set[str], docling_header: list[str]
) -> tuple[list[list[str]] | None, str, int]:
    """归一化 plumber 的原始网格。返回 (网格, 说明, 锚点列下标)。

    ═══ 关键实测：plumber 的**数据行优秀、表头不可用** ═══

    `pbc_21296` p4 的数据行与 docling **逐字相同**
    （`BT-29 | Rhabdoid | >EPg | <0.001 | >3.1 | 1.5 | 0.24 | Int | Int | Int`），
    但表头是 `'M FST/Cc fina'` / `'edian T lRTVd volu'` 这种乱码。

    **根因不是 bbox 切边** —— 实测 pad=0/1/2/3/5/8 全都一样（`mor Line` / `tage 1` 不变）。
    真正的原因是 **plumber 的列边界是从数据行的对齐推出来的，而表头与分节标签是居中排版的**，
    字符按 x 落到边界外就被切掉。数据行左对齐，所以完好。

    → **表头一律改用 docling 的**（同一个 bbox、列数一致时位置可对应）；
      列数不一致时退回用 plumber 自己的表头名做匹配，仍失败就标"测不出"。
    """
    ne = [r for r in raw if any(c.strip() for c in r)]
    if len(ne) < 3:
        return None, f"去空行后只剩 {len(ne)} 行", -1

    j, hits = find_anchor_col(ne, anchor_vals)
    if j < 0 or hits < 3:
        return None, f"找不到锚点列（最佳列只命中 {hits} 个锚点值）", -1

    first = next((i for i, r in enumerate(ne) if j < len(r) and norm(r[j]) in anchor_vals), None)
    if first is None or first == 0:
        return None, "定位不到首个数据行（表头区为空）", j

    # 分节行在首个数据行之前，不能被当成表头吞掉 —— 往回走，把连续的分节行留在数据区
    while first > 0 and is_section_like(ne[first - 1]):
        first -= 1

    ncol = max(len(r) for r in ne)
    # 表头优先用 docling 的（列数一致时位置对应）；否则用 plumber 自己拼的
    if len(docling_header) == ncol:
        header, hsrc = list(docling_header), "表头借用 docling"
    else:
        header = [" ".join(ne[i][c].strip() for i in range(first)
                           if c < len(ne[i]) and ne[i][c].strip()) for c in range(ncol)]
        hsrc = f"表头用 plumber 自己的（列数 {ncol}≠docling {len(docling_header)}）"
    data = [r + [""] * (ncol - len(r)) for r in ne[first:]]
    return [header] + data, f"去空行 {len(raw)}→{len(ne)}；{hsrc}", j


def main() -> int:
    golds = sorted(GOLD_DIR.glob("*.gold")) if GOLD_DIR.exists() else []
    if not golds:
        print(f"{GOLD_DIR} 下没有 gold")
        return 1

    print("第 2 步：pdfplumber（框内抽格）")
    print("=" * 112)
    print("%-22s %5s %6s %7s %9s  %s" %
          ("gold", "真值", "docling", "plumber", "锚点列", "归一化说明"))
    print("-" * 112)

    scored_tables = []
    for gp in golds:
        g = gold_io.load(gp)
        stem = Path(g.pdf).stem
        page = g.pages[0]
        out_csv = ROOT / "1.output" / stem / g.csv_name
        if not out_csv.exists():
            continue

        # 只测文字表 —— 图片表 docling 没网格、区域内也没文字层，plumber 无从下手
        d_rows = list(csv.reader(out_csv.open(newline="")))
        ck = CACHE / f"{stem}.pkl"
        if not ck.exists():
            print("%-22s  缺 docling 缓存，跳过" % gp.stem[:22])
            continue
        cands = [t for t in pickle.loads(ck.read_bytes()) if t["page"] == page and t.get("rows")]
        if not cands:
            print("%-22s  docling 在 p%d 无网格（图片路径）→ plumber 不适用" % (gp.stem[:22], page))
            continue
        t0 = max(cands, key=lambda t: len(t["rows"]))

        anchor_vals = {norm(part) for k in g.values for part in k.split(gold_io.ANCHOR_SEP) if part}
        with geom.normalized_pdf(ROOT / "0.pdf_input" / g.pdf) as (npdf, _i):
            raw = engine_plumber.extract_in_bbox(npdf, page, Rect(*t0["rect"]))
        grid, note, jcol = normalize_grid(raw, anchor_vals, t0["rows"][0])

        truth = TRUTH.get((stem, page), "?")
        d_n = len(t0["rows"]) - 1
        p_n = len(grid) - 1 if grid else "—"
        print("%-22s %5s %6s %7s %9s  %s" %
              (gp.stem[:22], truth, d_n, p_n, jcol if jcol >= 0 else "—", note))
        if grid:
            scored_tables.append((gp.stem, g, grid, d_rows, jcol))

    print("-" * 112)
    print("「锚点列」是**按 gold 锚点值内容发现**的下标 —— 这给了引擎在列识别上一个让步，"
          "所以显式打印出来。")

    # ————— 打分 —————
    print()
    print("plumber 打分（同一份 gold，source ∈ human+agreed）")
    print("=" * 112)
    print("%-22s %8s %8s %9s %9s %7s %8s  %s" %
          ("gold", "计分", "命中", "严格", "忽略空格", "漏行", "多出行", "备注"))
    print("-" * 112)
    for stem, g, grid, d_rows, jcol in scored_tables:
        # 把发现的锚点列改名成 gold 声明的名字，check_cols 按归一化列名找
        hdr = list(grid[0])
        # 分节表：plumber 的分节标签被截断（`Stage 1` → `tage 1`），锚点构造不出来。
        # **不做模糊修复** —— 那是拿 gold 的知识去补引擎的输出，会把"测不出"粉饰成"测出来了"。
        if g.section_rows:
            want = {k.split(gold_io.ANCHOR_SEP)[0] for k in g.values}
            got = {r[0].strip() for r in grid[1:] if is_section_like(r)}
            if not (want & got):
                print("%-22s  **分节标签被截断（%s vs gold 的 %s）→ 测不出**"
                      % (stem[:22], sorted(got)[:2], sorted(want)[:2]))
                continue
        miss = []
        for c in list(g.anchor_cols) + list(g.check_cols):
            if c in hdr:
                continue
            hit = next((i for i, h in enumerate(hdr) if norm(h) == norm(c)), None)
            if hit is None:
                miss.append(c)
            else:
                hdr[hit] = c
        if miss:
            print("%-22s  **表头里定位不到 %s** —— 不调参凑，标为测不出" % (stem[:22], miss))
            continue
        loose = gold_io.score_against(g, [hdr] + grid[1:], {"human", "agreed"},
                                      ignore_spaces=True)
        res = gold_io.score_against(g, [hdr] + grid[1:], {"human", "agreed"})
        if res.fatal:
            print("%-22s  %s" % (stem[:22], res.fatal[:70]))
            continue
        t, tl = res.totals, loose.totals
        rate = f"{t['match'] / t['scored']:.1%}" if t["scored"] else "—"
        lrate = f"{tl['match'] / tl['scored']:.1%}" if tl["scored"] else "—"
        n_gold_cells = max(1, len(g.check_cols) * sum(
            1 for a in g.values if g.sources.get(a, "agreed") in {"human", "agreed"}))
        align = t["scored"] / n_gold_cells
        note = f"锚点重复{len(res.anchor_not_unique)}组; " if res.anchor_not_unique else ""
        note += f"对齐率 {align:.0%}"
        print("%-22s %8d %8d %9s %9s %7d %8d  %s" %
              (stem[:22], t["scored"], t["match"], rate, lrate,
               len(res.missing_rows), len(res.extra_rows), note))
        for m in res.mismatches[:6]:
            print(f"        · {m.anchor[:34]:36s} {m.col[:16]:18s} plumber 抽到 {m.ours[:18]!r}")
    print("-" * 112)

    # ————— 跨引擎逐格对比（全列，不只 check_cols）—————
    #
    # 为什么要看全列：`ALL-7` 那处切列错（plumber 把 `10.8` 拆成 `1` + `0.8`）落在
    # `Median final RTV` 列上，而它**不是 check_col** —— 只看打分列就永远发现不了。
    print()
    print("跨引擎逐格对比（对齐上的行，全部列）")
    print("=" * 112)
    for stem, g, grid, d_rows, jcol in scored_tables:
        hdr = list(grid[0])
        try:
            pk, pdata = gold_io.anchors_of(hdr_named(hdr, g) + grid[1:], g.anchor_cols, g.section_rows)
            dk, ddata = gold_io.anchors_of(d_rows, g.anchor_cols, g.section_rows)
        except gold_io.GoldError as e:
            print("%-22s  对不上，跳过（%s）" % (stem[:22], str(e)[:50]))
            continue
        pmap = {k: pdata[i] for i, k in enumerate(pk)}
        dmap = {k: ddata[i] for i, k in enumerate(dk)}
        common = [k for k in dk if k in pmap]
        same = ws_only = real = 0
        examples = []
        for k in common:
            dr, pr = dmap[k], pmap[k]
            for c in range(min(len(dr), len(pr))):
                a, b = dr[c].strip(), pr[c].strip()
                if a == b:
                    same += 1
                elif "".join(a.split()) == "".join(b.split()):
                    ws_only += 1
                else:
                    real += 1
                    if len(examples) < 8:
                        col = d_rows[0][c] if c < len(d_rows[0]) else f"#{c}"
                        examples.append((k, col, a, b))
        print("%-22s 共同行 %d/%d(docling)  格子：相同 %d，仅空格差 %d，**真分歧 %d**"
              % (stem[:22], len(common), len(dk), same, ws_only, real))
        for k, col, a, b in examples:
            print("      %-26s %-22s docling=%-16r plumber=%-16r" % (k[:26], col[:22], a[:16], b[:16]))
        only_p = [k for k in pk if k not in dmap]
        only_d = [k for k in dk if k not in pmap]
        if only_p:
            print("      **只有 plumber 抽到的行**：%s" % only_p[:4])
        if only_d:
            print("      只有 docling 抽到的行：%s" % only_d[:4])
    print("-" * 112)
    print("「真分歧」= 两个引擎都抽到该格但值不同（已排除纯空格差）。这些格子的对错由 gold 裁 ——")
    print("落在 check_cols 上的看上一节的失配清单，不在 check_cols 上的需要人工看。")
    return 0


def hdr_named(hdr, g):
    """把表头里 gold 声明的列名对齐好，供 anchors_of 用。"""
    out = list(hdr)
    for c in list(g.anchor_cols) + list(g.check_cols):
        if c in out:
            continue
        hit = next((i for i, h in enumerate(out) if norm(h) == norm(c)), None)
        if hit is not None:
            out[hit] = c
    return [out]


if __name__ == "__main__":
    sys.exit(main())
