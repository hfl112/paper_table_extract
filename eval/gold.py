"""声明式 gold 文件的读写与行对齐。

    conda run -n gemini python eval/gold.py --validate-anchors 1.output --pdf-dir 0.pdf_input

**为什么是声明式的**：`NEXT.md` 原方案写死"第一列当行锚点"，实测在它自己挑的表上就不成立 ——
`pbc_28772` 第一列 `Model` 有 8 行是空的、`G401` 重复 8 次；`pbc_30017` 第一列 12 行全是
`Osteosarcoma`；`pbc_26870` `ES-4` 重复 16 次。所以锚点列必须由标注者按每张表的实际情况声明，
脚本加载时**先查唯一性、有重复当场报错**，不能等人抄完 132 个值才发现白抄。

三条从实测逼出来的机制（缺一不可，验证见 --validate-anchors）：

1. **锚点列必须向下填充** —— 合并单元格导致纵跨的那一列只有第一行有值（AGENTS.md 事实 #15）。
   `pbc_28772` `Model+Agent` 不填充 43/44 唯一、填充后 **44/44**；`pbc_29304` 从 23 → **41**；
   `blood` table_2 从 3 → **14**。
2. **分节标题行要单独识别**（整行只有第一格有值，其余全空）。`pbc_26870` 44 行里 3 行是
   `Stage 1/2/3`，真数据行 41 行；`ES-4|Erlotinib` 在 Stage 1 和 Stage 3 **各出现一次**，
   不把 Stage 算进锚点必然撞车。
3. **`check_cols` 每张表不同** —— `pbc_28772` 只核 1 列（`Obj. Response`）；`pbc_29304` 要核 7 列
   （`PD`/`PD1`/`PD2`/`SD`/`PR`/`CR`/`MCR` 各占一列、格子里是**计数**）；`pbc_21078 table_i` 搜
   `Demographic` 核的是 `Patient age`/`Sex`/`Stage`。同为"响应表"结构完全不同，一个固定 header
   套不住。

文件格式（`---` 之上是头部，之下是 CSV 正文）：

    pdf: pbc_28772.pdf
    csv: p03_table_1.csv         # 产物文件名，打分器靠它定位
    label: TABLE 1
    pages: 3-4
    expect_data_rows: 44          # 不含表头，不含分节标题行
    expect_cols: 11
    anchor_cols: Model + Agent    # 用 + 连接；脚本先向下填充再查唯一性
    section_rows: false           # 表内有无分节标题行
    check_cols: Obj. Response     # 用 , 分隔
    ---
    anchor,Obj. Response,source
    G401|Control,PD,human
    G401|EPZ011989,PD2,agreed

列名可以写 `#5` 这种 **0-based 序号** —— 给图片表用，它们的表头本身就是 OCR 垃圾
（`pbc_21296 fig_1` 的表头是 `Linte / Histology / Score / Difference mruTTe / Growp Response`）。

格子的**空值三态**，不区分就会让「漏抽」白得分：
  `<EMPTY>`  期望这格就是空（**不能用 `-`** —— `pbc_28772` 里 `-` 是真实值"不适用"）
  留空       还没标，计入 unlabeled、不进分母
  有值       正常比对

`source` 三档不许混：
  human               —— 人裁决过，是真 gold
  agreed              —— LLM 第二读者与工具一致，**没人看过**
  disputed_unresolved —— 有分歧但还没裁决
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ANCHOR_SEP = "|"
SOURCE_LEVELS = ("human", "agreed", "disputed_unresolved")

# 列可以按名字写，也可以按 0-based 序号写成 `#5`。
# 后者是给**图片表**用的 —— 它们的表头本身就是 OCR 垃圾（`Linte` / `Growp Response`）。
_IDX_SPEC = re.compile(r"^#(\d+)$")

# 空值三态里的「期望这格就是空」。**不能用 `-`** —— `pbc_28772` 里 `-` 是真实值（"不适用"）。
EMPTY_SENTINEL = "<EMPTY>"


class GoldError(Exception):
    """gold 文件本身有问题（锚点不唯一、列名不存在等）。**当场抛，不静默降级。**"""


@dataclass
class Gold:
    path: Path
    pdf: str
    csv_name: str  # 产物文件名，如 `p04_table_i.csv` —— 打分器靠它定位要比的那张表
    label: str
    pages: list[int]
    expect_data_rows: int
    expect_cols: int
    anchor_cols: list[str]
    check_cols: list[str]
    section_rows: bool
    values: dict[str, dict[str, str]] = field(default_factory=dict)  # anchor -> {col: value}
    sources: dict[str, str] = field(default_factory=dict)            # anchor -> source


# ————————————————————— 行结构 —————————————————————

def forward_fill(values: list[str]) -> list[str]:
    """把合并单元格补回来：空格子继承上一个非空值。

    合法的例子（AGENTS.md 事实 #15）：`pbc_30017` 的 `Model` 列，`OS-2` 在原表纵跨
    Control / Copanlisib 两行，导出后第二行该格为空 —— 数据没丢，只是位置合并。
    """
    out, last = [], ""
    for v in values:
        if v.strip():
            last = v.strip()
        out.append(last)
    return out


def is_section_row(row: list[str]) -> bool:
    """分节标题行：只有第一格有值，其余全空。如 `pbc_26870` 的 `Stage 1` / `Stage 2` / `Stage 3`。"""
    return bool(row) and bool(row[0].strip()) and not any(c.strip() for c in row[1:])


def split_rows(rows: list[list[str]], section_rows: bool) -> tuple[list[str], list[list[str]], list[str]]:
    """拆成 (表头, 数据行, 每个数据行所属的分节)。不启用分节时第三个返回值全是空串。"""
    header, body = rows[0], rows[1:]
    data: list[list[str]] = []
    sections: list[str] = []
    current = ""
    for r in body:
        if section_rows and is_section_row(r):
            current = r[0].strip()
            continue
        if not any(c.strip() for c in r):
            # **整行全空的行直接丢掉。** 它不可能对上 gold 里任何一行，留着只有坏处：
            #   1. 锚点变成空串 ⇒ 多个空行互相撞车 ⇒ 打分器报"锚点不唯一"、
            #      然后整张表的逐格分都被标成不可信
            #   2. 虚增 `多出行` 与实际数据行数
            # 实测 `pbc_24724 p05_fig_1` 基线产物 46 个数据行里 **8 行全空**；
            # 去掉它们之后 `#0+#1` 从 38/46 变成 **38/38 唯一**。
            # 注意这**不会掩盖"产品吐空行"这个问题** —— 行数由 `expected.csv` 那边
            # 按 manifest 的 n_rows 断言，那份计的是原始行数。
            continue
        data.append(r)
        sections.append(current)
    return header, data, sections


def anchors_of(
    rows: list[list[str]], anchor_cols: list[str], section_rows: bool
) -> tuple[list[str], list[list[str]]]:
    """算出每个数据行的锚点键。返回 (锚点列表, 数据行)。

    锚点 = [分节] + 各锚点列（**向下填充后**）的值，用 `|` 连接。
    """
    header, data, sections = split_rows(rows, section_rows)
    missing = [c for c in anchor_cols if c not in header]
    if missing:
        raise GoldError(f"锚点列在表头里不存在: {missing}；表头是 {header}")

    filled: list[list[str]] = []
    for col in anchor_cols:
        i = header.index(col)
        filled.append(forward_fill([r[i] if i < len(r) else "" for r in data]))

    keys = []
    for j in range(len(data)):
        parts = ([sections[j]] if section_rows else []) + [f[j] for f in filled]
        keys.append(ANCHOR_SEP.join(parts))
    return keys, data


def check_anchor_unique(keys: list[str]) -> list[tuple[str, int]]:
    """返回重复的锚点键及其出现次数。**空列表才算可用。**"""
    seen: dict[str, int] = {}
    for k in keys:
        seen[k] = seen.get(k, 0) + 1
    return sorted(((k, n) for k, n in seen.items() if n > 1), key=lambda x: -x[1])


# ————————————————————— 读写 —————————————————————

def _parse_header(text: str) -> dict[str, str]:
    """解析头部。**`#` 后面跟数字时不算注释。**

    不能一律把 `#` 当注释起始 —— 列可以写成 `#5` 这种序号（图片表专用）。
    踩过两次：先是 `line.split("#")` 把 `check_cols: #1` 整行吃空，
    改成 `\\s#` 之后又被 `#` 前面那个空格命中，值仍然是空。
    两次都表现为"头部缺字段 ['check_cols']"，跟真正的原因隔着一层，很难查。
    """
    out = {}
    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            continue
        line = re.split(r"\s#(?!\d)", line, maxsplit=1)[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip()] = v.strip()
    return out


def _parse_pages(spec: str) -> list[int]:
    pages: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            pages.extend(range(int(a), int(b) + 1))
        elif part:
            pages.append(int(part))
    return pages


def load(path: Path) -> Gold:
    raw = path.read_text()
    if "\n---" not in raw:
        raise GoldError(f"{path}: 缺少 `---` 分隔线（头部与正文之间）")
    head_txt, body_txt = raw.split("\n---", 1)
    h = _parse_header(head_txt)

    required = ("pdf", "csv", "label", "pages", "expect_data_rows", "expect_cols",
                "anchor_cols", "check_cols")
    lack = [k for k in required if k not in h]
    if lack:
        raise GoldError(f"{path}: 头部缺字段 {lack}")

    gold = Gold(
        path=path,
        pdf=h["pdf"],
        csv_name=h["csv"],
        label=h["label"],
        pages=_parse_pages(h["pages"]),
        expect_data_rows=int(h["expect_data_rows"]),
        expect_cols=int(h["expect_cols"]),
        anchor_cols=[c.strip() for c in h["anchor_cols"].split("+") if c.strip()],
        check_cols=[c.strip() for c in h["check_cols"].split(",") if c.strip()],
        section_rows=h.get("section_rows", "false").strip().lower() in ("true", "yes", "1"),
    )

    body = body_txt.lstrip("\n")
    reader = csv.DictReader(body.splitlines())

    # `#N` 在**头部**指的是抽出来那张表的第 N 列；而 gold 正文的列名用的是人读得懂的真实表头
    # （`gold-draft` 保留了 LLM 读到的全部列）。两边要接上：`#N` → 正文第 N 个数据列
    # （跳过 `anchor` 与 `source`，顺序与表列一致）。
    #
    # 不接上的后果（实测踩到）：两张图片表的 `check_cols: #4` / `#5` 在正文里查无此列，
    # `row.get("#4")` 全返回 None → **gold 值一个都没读进来**，打分器报"响应码 gold 0 个"，
    # 看着像"gold 里没标"，其实是**读取时就丢了**。
    body_cols = [c for c in (reader.fieldnames or []) if c not in ("anchor", "source")]
    col_alias = {}
    for spec in gold.check_cols + gold.anchor_cols:
        m = _IDX_SPEC.match(spec)
        if m and int(m.group(1)) < len(body_cols):
            col_alias[spec] = body_cols[int(m.group(1))]

    for row in reader:
        anchor = (row.get("anchor") or "").strip()
        if not anchor:
            continue
        if anchor in gold.values:
            raise GoldError(f"{path}: gold 正文里锚点 {anchor!r} 出现多次")
        gold.values[anchor] = {c: (row.get(col_alias.get(c, c)) or "").strip()
                               for c in gold.check_cols}
        src = (row.get("source") or "agreed").strip()
        if src not in SOURCE_LEVELS:
            raise GoldError(f"{path}: source={src!r} 不在 {SOURCE_LEVELS}")
        gold.sources[anchor] = src
    return gold


def apply_index_specs(rows: list[list[str]], *specs: list[str]) -> list[list[str]]:
    """把 `#N` 形式的列名在**表头副本**上落实成字面量 `#N`，其余逻辑不用改。

    为什么需要：**图片表的表头本身就是 OCR 垃圾**，没法按名字引用 ——
    `pbc_24724 fig_1` 首列名整个缺失（导致全表左移一位）、
    `pbc_21296 fig_1` 的表头是 `Linte / Histology / Score / Difference mruTTe / Growp Response`。
    不支持列序号，5 张 gold 里那 2 张图片表根本写不出来。
    """
    idxs = sorted({int(m.group(1)) for s in specs for c in s if (m := _IDX_SPEC.match(c))})
    if not idxs or not rows:
        return rows
    out = [list(rows[0])] + [list(r) for r in rows[1:]]
    for i in idxs:
        if i < len(out[0]):
            out[0][i] = f"#{i}"
    return out


@dataclass
class CellVerdict:
    """一格的裁决。**刻意不存 gold 的值** —— 见 score_against 的非对称契约。"""

    anchor: str
    col: str
    ours: str
    kind: str  # mismatch / missing_row


@dataclass
class ScoreResult:
    label: str
    shape: list[str] = field(default_factory=list)
    per_col: dict[str, dict[str, int]] = field(default_factory=dict)
    missing_rows: list[str] = field(default_factory=list)
    extra_rows: list[str] = field(default_factory=list)
    mismatches: list[CellVerdict] = field(default_factory=list)
    anchor_not_unique: list[tuple[str, int]] = field(default_factory=list)
    fatal: str = ""
    # ═══ 整行判据（`rows_*`）—— 逐格命中率会掩盖漏行 ═══
    #
    # 实测：`pbc_21296 p04_table_i` 报 **84/84 = 100%**，同时**漏了 2 个数据行**
    # （`ALL-8`/`ALL-16`，候选 A 那个压行问题）。
    # 100% 只算了"对上号的行"里的格子，漏掉的行根本没进分母 ——
    # 像考试只算你答了的题，没答的不算错，所以能拿满分却少答两道。
    #
    # 而目标(b)要的是「瘤系 × 处理 × 响应 在**同一行上**同时对」。
    # 所以这里按 gold 的**每一个数据行**记账，**漏行直接算错**：
    #   rows_total    分母 = gold 里该计分的行数（不含未标行）
    #   rows_full     该行全部 check_cols 都对
    #   rows_partial  部分对
    #   rows_missing  整行没抽到（= len(missing_rows) 的计分子集）
    rows_total: int = 0
    rows_full: int = 0
    rows_partial: int = 0
    rows_missing: int = 0

    @property
    def totals(self) -> dict[str, int]:
        t = {"scored": 0, "match": 0, "mismatch": 0, "unlabeled": 0}
        for c in self.per_col.values():
            for k in t:
                t[k] += c.get(k, 0)
        return t


def score_against(
    gold: Gold, rows: list[list[str]], source_filter: set[str] | None = None,
    ignore_spaces: bool = False,
) -> ScoreResult:
    """给一张实际抽出的表打分。

    ═══ 非对称契约（隔离机制的核心，改这里要格外小心）═══

    返回的 `ScoreResult` 里**只有我们这边的值**（`CellVerdict.ours`），
    **任何地方都不带 gold 的期望值**。这样 AI 能跑打分、能据此改代码，
    但学不到答案 —— 否则 gold 就退化成第二个开发集，再也不能当独立证据用。
    `tests/test_eval.py` 有一条 `ZZZSECRET` 单测钉死这一点，**那条测试才是真正的保证**，
    比 `.claude/settings.local.json` 的 deny 规则可靠（deny 挡不住 `open()`）。

    ═══ 空值三态（不区分就会让「漏抽」白得分）═══

    - gold 写 `<EMPTY>`  → **期望这格就是空**，实际非空即 mismatch
    - gold 留空          → **还没标**，计入 `unlabeled`，不进分母
    - gold 有值          → 正常比对

    原实现是 `if wv and ...`，把上面前两种混成一种、且都静默跳过。
    """
    src_ok = source_filter or {"human"}
    res = ScoreResult(label=gold.label)
    rows = apply_index_specs(rows, gold.anchor_cols, gold.check_cols)

    try:
        keys, data = anchors_of(rows, gold.anchor_cols, gold.section_rows)
    except GoldError as e:
        res.fatal = str(e)
        return res

    dup = check_anchor_unique(keys)
    if dup:
        # **不再 raise。** `pbc_24724 fig_1` 末尾 3 行全空 → 锚点全是空串 → 必崩，
        # 而那恰恰是最需要打分的一张。记成 finding，继续出聚合数。
        res.anchor_not_unique = dup

    if len(data) != gold.expect_data_rows:
        res.shape.append(f"数据行数 期望 {gold.expect_data_rows} 实际 {len(data)}")
    if len(rows[0]) != gold.expect_cols:
        res.shape.append(f"列数 期望 {gold.expect_cols} 实际 {len(rows[0])}")

    header = rows[0]
    lack = [c for c in gold.check_cols if c not in header]
    if lack:
        res.fatal = f"check_cols 在实际表头里不存在: {lack}（图片表请改用 #N 列序号）"
        return res

    actual = {k: {c: (data[j][header.index(c)] if header.index(c) < len(data[j]) else "")
                  for c in gold.check_cols}
              for j, k in enumerate(keys)}
    res.per_col = {c: {"scored": 0, "match": 0, "mismatch": 0, "unlabeled": 0}
                   for c in gold.check_cols}

    for anchor, want in gold.values.items():
        if gold.sources.get(anchor, "agreed") not in src_ok:
            continue
        got = actual.get(anchor)
        labeled = sum(1 for wv in want.values() if wv)
        if labeled:
            res.rows_total += 1               # 分母只算"至少标了一列"的行
        if got is None:
            res.missing_rows.append(anchor)
            if labeled:
                res.rows_missing += 1         # **漏行算错**，这是整行判据的全部意义
            continue
        row_ok = row_bad = 0
        for col, wv in want.items():
            bucket = res.per_col[col]
            if not wv:
                bucket["unlabeled"] += 1      # 还没标 —— 不进分母
                continue
            bucket["scored"] += 1
            mine = got[col].strip()
            expected_empty = wv == EMPTY_SENTINEL
            if expected_empty:
                ok = mine == ""
            elif ignore_spaces:
                # pdfplumber 会把 `ALV rhabdomyosarcoma` 抽成 `ALVrhabdomyosarcoma` ——
                # **丢空格和抽错值是两类错**，混在一起报会看不出引擎真正的强弱。
                # 两档都跑，差值就是纯空格问题的规模；这不会泄露 gold 的值。
                ok = "".join(mine.split()) == "".join(wv.split())
            else:
                ok = mine == wv
            if ok:
                bucket["match"] += 1
                row_ok += 1
            else:
                bucket["mismatch"] += 1
                row_bad += 1
                res.mismatches.append(CellVerdict(anchor, col, mine, "mismatch"))
        if labeled:
            if row_bad == 0:
                res.rows_full += 1
            elif row_ok:
                res.rows_partial += 1

    res.extra_rows = [a for a in actual if a not in gold.values]
    return res


def validate_against(gold: Gold, rows: list[list[str]]) -> list[str]:
    """旧接口：返回人可读的问题列表。**只在本地裁决时用，它会打印 gold 的值。**

    打分一律走 `score_against` —— 它是非对称的。
    """
    res = score_against(gold, rows, source_filter=set(SOURCE_LEVELS))
    if res.fatal:
        return [res.fatal]
    out = list(res.shape)
    out += [f"漏行: {a}" for a in res.missing_rows]
    out += [f"多出行: {a}" for a in res.extra_rows]
    out += [f"{m.anchor} / {m.col}: 实际 {m.ours!r}" for m in res.mismatches]
    return out


# ————————————————————— 锚点可用性自检 —————————————————————

# 本轮 grill 已在真实数据上验证过的锚点。--validate-anchors 复跑一遍，
# 顺便证明"向下填充"和"分节行"两条机制缺一不可。
KNOWN_ANCHORS: list[tuple[str, str, list[str], bool, int]] = [
    ("pbc_28772", "p03_table_1.csv", ["Model", "Agent"], False, 44),
    ("pbc_30017", "p07_table_1.csv", ["Model", "Agent"], False, 12),
    ("pbc_29304", "p03_table_1.csv", ["Tumor", "Group"], False, 41),
    ("blood_2014_12_518900", "p04_table_2.csv", ["", "Treatment"], False, 14),
    ("blood_2014_12_518900", "p06_table_3.csv", ["", "Treatment"], False, 6),
    ("pbc_26870", "p18_table_1.csv", ["Tumor Line", "Treatment Group"], True, 41),
    ("pbc_21078", "p04_table_i.csv", ["Xenograft"], False, 60),
]


def validate_anchors(outdir: Path) -> int:
    print(f"{'表':38s} {'锚点':44s} {'数据行':>6s} {'唯一':>5s}  {'不填充时':>8s}")
    print("-" * 112)
    bad = 0
    for prefix, name, cols, sect, want_rows in KNOWN_ANCHORS:
        path = outdir / prefix / name
        if not path.exists():
            print(f"{prefix + '/' + name:38s} 跳过（{path} 不存在）")
            continue
        rows = list(csv.reader(path.open(newline="")))
        keys, data = anchors_of(rows, cols, sect)
        dup = check_anchor_unique(keys)

        # 对照组：不做向下填充会怎样
        header, raw_data, sections = split_rows(rows, sect)
        raw_keys = []
        for j, r in enumerate(raw_data):
            parts = ([sections[j]] if sect else []) + [
                (r[header.index(c)] if header.index(c) < len(r) else "") for c in cols
            ]
            raw_keys.append(ANCHOR_SEP.join(parts))
        raw_uniq = len(set(raw_keys))

        ok = not dup and len(data) == want_rows
        bad += 0 if ok else 1
        label = ("+".join(c or "(首列)" for c in cols)) + (" +分节" if sect else "")
        print(f"{prefix + '/' + name:38s} {label:44s} {len(data):6d} "
              f"{len(set(keys)):5d}  {raw_uniq:8d}  {'OK' if ok else '✗'}")
        if dup:
            print(f"      ✗ 重复 {len(dup)} 组: {dup[:3]}")
        if len(data) != want_rows:
            print(f"      ✗ 数据行数 {want_rows} → {len(data)}")

    print("-" * 112)
    print("「不填充时」这一列若小于「唯一」列，就说明**向下填充是必需的** —— "
          "合并单元格让锚点列出现空值。")

    # 负面测试：故意用不够的锚点，必须报错而不是静默退化
    print()
    print("负面测试：pbc_28772 只用 `Model` 当锚点（G401 重复 8 次）")
    path = outdir / "pbc_28772" / "p03_table_1.csv"
    if path.exists():
        rows = list(csv.reader(path.open(newline="")))
        keys, _ = anchors_of(rows, ["Model"], False)
        dup = check_anchor_unique(keys)
        if dup:
            print(f"   ✓ 正确检出不唯一：{len(dup)} 组重复，最多的 {dup[0]}")
        else:
            print("   ✗ 没检出重复 —— 唯一性校验失效")
            bad += 1
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate-anchors", type=Path, metavar="OUTDIR",
                    help="拿真实抽取产物复验 KNOWN_ANCHORS 里的锚点")
    ap.add_argument("--pdf-dir", type=Path, help="（保留）源 PDF 目录")
    ap.add_argument("--check", type=Path, metavar="GOLD", help="加载并校验一个 gold 文件")
    ap.add_argument("--score", type=Path, nargs="?", metavar="GOLD_DIR",
                    const=Path(os.environ.get("PTE_GOLD_DIR", "/Users/funanhe/pte_gold")),
                    help="对 gold 目录打分（默认 $PTE_GOLD_DIR 或 /Users/funanhe/pte_gold）")
    ap.add_argument("--outdir", type=Path, default=Path("1.output"), help="抽取产物根目录")
    ap.add_argument("--source", default="human",
                    help="计分的 source 档，逗号分隔。默认只算 human；"
                         "用 human,agreed 可看「LLM 与工具一致但没人看过」那部分的规模")
    ap.add_argument("--map", default="",
                    help="产物目录名映射，`PDF名=目录名` 逗号分隔。"
                         "回归用 —— `regression.sh` 的目录名是 `--prefix`（如 `gold_28772`），"
                         "不是 PDF 的 stem，打分器默认按 stem 找会全部 miss")
    ap.add_argument("--suggest-anchors", action="store_true",
                    help="给每张 gold 找「在我们产物里也还唯一」的锚点列组合（只报计数，不报值）")
    ap.add_argument("--baseline", type=Path, metavar="CSV",
                    help="与基线快照对账并决定退出码。缺少基线文件时只打分、不判定")
    ap.add_argument("--tier", default="full", choices=("fast", "full"),
                    help="只对账基线里 tier 不高于本档的条目（fast ⊂ full）")
    args = ap.parse_args()

    if args.validate_anchors:
        return validate_anchors(args.validate_anchors)
    if args.check:
        g = load(args.check)
        print(f"{g.path.name}: pdf={g.pdf} label={g.label} pages={g.pages} "
              f"anchor={'+'.join(g.anchor_cols)} check={g.check_cols} 条目 {len(g.values)}")
        from collections import Counter
        print("  source 分布:", dict(Counter(g.sources.values())))
        return 0
    if args.score:
        pmap = dict(kv.split("=", 1) for kv in args.map.split(",") if "=" in kv)
        if args.suggest_anchors:
            return cmd_suggest_anchors(args.score, args.outdir, pmap)
        return cmd_score(args.score, args.outdir, set(args.source.split(",")),
                         prefix_map=pmap, baseline=args.baseline, tier=args.tier)
    ap.print_help()
    return 0


def cmd_suggest_anchors(gold_dir: Path, outdir: Path,
                        prefix_map: dict[str, str] | None = None) -> int:
    """给每张 gold 找「在**我们的产物**里也还唯一」的锚点列组合。

    ═══ 先把问题看清，否则会改错东西 ═══

    打分器报的 `锚点不唯一 9 组: [('ALL-17', 4), ...]` 是发生在**我们抽出来的表**里，
    **不是 gold 里** —— gold 的锚点唯一性在 `load()` 里当场校验过，不唯一根本加载不了。
    真实原因是 OCR 把不同的瘤系名读成了同一串（`ALL-17` 出现 4 次）。

    所以「换锚点列」要找的是：**即使我们这边读糊了，仍然能把行区分开**的列组合。
    单列不行就试两列拼（瘤系名 + 组织学），像 `pbc_28772` 的 `Model + Agent` 那样。

    ═══ 非对称契约 ═══

    只打印**列序号和唯一性计数**，不打印任何格子的值 —— gold 和产物的都不打印。
    所以它不会泄露答案，也不会变成"照着答案挑锚点"。
    """
    prefix_map = prefix_map or {}
    for gp in sorted(gold_dir.glob("*.gold")):
        try:
            g = load(gp)
        except GoldError as e:
            print(f"{gp.stem}: gold 加载失败 {e}")
            continue
        stem = Path(g.pdf).stem
        csv_path = outdir / prefix_map.get(stem, stem) / g.csv_name
        if not csv_path.exists():
            print(f"{gp.stem}: 找不到产物 {csv_path}")
            continue
        rows = list(csv.reader(csv_path.open(newline="")))
        if len(rows) < 2:
            print(f"{gp.stem}: 产物不足 2 行")
            continue
        _, data, _ = split_rows(rows, g.section_rows)
        n = len(rows[0])
        cols = {i: forward_fill([r[i] if i < len(r) else "" for r in data])
                for i in range(n)}
        cur = "+".join(g.anchor_cols)
        print(f"\n══ {gp.stem}   产物 {len(data)} 数据行 x {n} 列   现用锚点: {cur}")
        singles = []
        for i in range(n):
            u = len(set(cols[i]))
            singles.append((u, i))
            mark = "✔唯一" if u == len(data) else f"{u}/{len(data)}"
            print(f"     #{i}: {mark}")
        # 只在单列都不唯一时才试两列 —— 锚点越短越稳，能单列就别拼
        if not any(u == len(data) for u, _ in singles):
            print("     单列都不唯一，试两列拼：")
            best = []
            for i in range(n):
                for j in range(i + 1, n):
                    u = len({f"{a}|{b}" for a, b in zip(cols[i], cols[j])})
                    best.append((u, i, j))
            best.sort(key=lambda x: -x[0])
            for u, i, j in best[:6]:
                mark = "✔唯一" if u == len(data) else f"{u}/{len(data)}"
                print(f"       #{i}+#{j}: {mark}")
    print("\n只打印列序号与唯一性计数，不打印任何格子的值（gold 与产物都不打）。")
    return 0


def load_baseline(path: Path) -> dict[str, dict]:
    """读基线快照。列：gold,tier,scored,match,missing,extra"""
    out: dict[str, dict] = {}
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("gold", "").strip() or r["gold"].lstrip().startswith("#"):
                continue
            out[r["gold"].strip()] = {
                "tier": r["tier"].strip(),
                **{k: int(r[k]) for k in ("scored", "match", "missing", "extra",
                                          "rows_full", "rows_total")},
            }
    return out


# 基线判定：**只有"变差"才 exit 1**。
#
# 这套三档跟 `holdout/report.py` 一致（✓ 通过 / ▲ 变好待更新基线 / ✗ 变差）。
# 为什么不用"完全相等"：expected.csv 那边用的是精确快照，因为那些是行列数、
# 整数、确定性的。而这里的 `match` 依赖 OCR 结果，一旦模型或 PIL 版本动一下就漂 ——
# 精确相等会天天亮红灯，然后大家就对红色脱敏了（`AGENTS.md` 里写过这一课）。
# 但"变好了也要吭一声"仍然必要，否则改进会静默淹掉、基线永远停在旧值。
WORSE_KEYS = ("match", "rows_full")      # 越大越好
WORSE_KEYS_INV = ("missing", "extra")  # 越小越好


def cmd_score(gold_dir: Path, outdir: Path, sources: set[str],
              prefix_map: dict[str, str] | None = None,
              baseline: Path | None = None, tier: str = "full") -> int:
    """对 gold_dir 下每份 gold 打分。**输出永不包含 gold 的值。**

    gold 放仓库外（默认 `/Users/funanhe/pte_gold/`，可用 `PTE_GOLD_DIR` 覆盖）。
    详见 `score_against` 的非对称契约。

    给了 `baseline` 就同时与基线快照对账，变差则返回 1 —— 这是为了接进
    `eval/regression.sh`。接它的直接动机是一次真事故（`log.md` §37.1）：
    `1.output/` 比产品代码旧了 3 天，打分报「漏行 5」，
    而那 5 行**早就修好了**，纯粹是产物过期。打分不在回归里 ⇒ 没人重跑产物 ⇒
    拿过期产物下结论。
    """
    if not gold_dir.exists():
        print(f"gold 目录不存在: {gold_dir}")
        print("这是预期情况 —— 打分器**刻意在 gold 数据之前写好**，避免照着答案拟合。")
        return 0
    golds = sorted(gold_dir.glob("*.gold"))
    if not golds:
        print(f"{gold_dir} 下没有 *.gold 文件")
        return 0
    prefix_map = prefix_map or {}
    base = load_baseline(baseline) if (baseline and baseline.exists()) else {}
    if baseline and not base:
        print(f"（基线文件 {baseline} 不存在或为空 —— 只打分，不判定）")
    got: dict[str, dict] = {}

    print(f"打分档: source ∈ {sorted(sources)}    （gold 的值不会出现在下面任何一行）")
    print("=" * 100)
    print("%-26s %7s %7s %8s %10s %8s %8s   %s" %
          ("gold", "计分", "命中", "命中率", "未标", "漏行", "多出行", "整行全对"))
    print("-" * 100)
    bad = 0
    for gp in golds:
        try:
            g = load(gp)
        except GoldError as e:
            print(f"{gp.name:26s} gold 文件本身有问题: {e}")
            bad += 1
            continue
        stem = Path(g.pdf).stem
        csv_path = outdir / prefix_map.get(stem, stem) / g.csv_name
        if not csv_path.exists():
            # 基线里没这一条、或它属于更高档 ⇒ 本档本来就不该产出，不算失败
            want = base.get(gp.stem)
            if base and (want is None or (tier == "fast" and want["tier"] == "full")):
                continue
            print(f"{gp.name:26s} 找不到产物 {csv_path}")
            bad += 1
            continue
        rows = list(csv.reader(csv_path.open(newline="")))
        res = score_against(g, rows, sources)
        if res.fatal:
            print(f"{gp.name:26s} {res.fatal}")
            bad += 1
            continue
        t = res.totals
        got[gp.stem] = {"scored": t["scored"], "match": t["match"],
                        "missing": len(res.missing_rows), "extra": len(res.extra_rows),
                        "rows_full": res.rows_full, "rows_total": res.rows_total}
        rate = f"{t['match'] / t['scored']:.1%}" if t["scored"] else "—"
        rrate = f"{res.rows_full / res.rows_total:.0%}" if res.rows_total else "—"
        print("%-26s %7d %7d %8s %10d %8d %8d   %s" %
              (gp.stem[:26], t["scored"], t["match"], rate,
               t["unlabeled"], len(res.missing_rows), len(res.extra_rows),
               f"{res.rows_full}/{res.rows_total}={rrate}"))
        if res.rows_partial or res.rows_missing:
            print(f"      整行: 全对 {res.rows_full} / 部分对 {res.rows_partial} / "
                  f"**整行漏抽 {res.rows_missing}** （分母 {res.rows_total}，漏行算错）")
        for line in res.shape:
            print(f"      形状: {line}")
        if res.anchor_not_unique:
            print(f"      ⚠ 锚点不唯一 {len(res.anchor_not_unique)} 组: "
                  f"{res.anchor_not_unique[:3]} —— 该表的逐格分不可信，先换锚点列")
        for col, c in res.per_col.items():
            if c["mismatch"]:
                print(f"      列 {col!r}: 错 {c['mismatch']}/{c['scored']}")
        for m in res.mismatches[:8]:
            print(f"        · {m.anchor}  {m.col}  我们抽到 {m.ours!r}")
        if len(res.mismatches) > 8:
            print(f"        · …… 另有 {len(res.mismatches) - 8} 处")
        for a in res.missing_rows[:5]:
            print(f"        · 漏行 {a}")
    print("-" * 100)
    print("对分歧只打印**我们这边**的值。要看 gold 期望值，请自己打开 gold 文件 ——")
    print("这条不对称是刻意的，防的是「照着答案调阈值」把 gold 变成第二个开发集。")

    if not base:
        return 1 if bad else 0

    print()
    print(f"########## 与基线 {baseline} 对账（档: {tier}）##########")
    worse = better = 0
    for name, want in sorted(base.items()):
        if tier == "fast" and want["tier"] == "full":
            continue
        have = got.get(name)
        if have is None:
            print(f"  ✗ {name:30s} 基线里有、本次没打出分（产物缺失或 gold 加载失败）")
            worse += 1
            continue
        deltas = []
        w = b = False
        if have["rows_total"] != want["rows_total"]:
            # 分母变了 ⇒ gold 的标注量改了 ⇒ rows_full 的比较**没有意义**，先让人确认
            print(f"  ✗ {name:30s} **分母变了**: rows_total "
                  f"{want['rows_total']}→{have['rows_total']}（gold 标注量改了？先确认再比 rows_full）")
            worse += 1
            continue
        for k in WORSE_KEYS:
            d = have[k] - want[k]
            if d:
                deltas.append(f"{k} {want[k]}→{have[k]}")
                w |= d < 0
                b |= d > 0
        for k in WORSE_KEYS_INV:
            d = have[k] - want[k]
            if d:
                deltas.append(f"{k} {want[k]}→{have[k]}")
                w |= d > 0
                b |= d < 0
        if w:
            print(f"  ✗ {name:30s} **变差**: {', '.join(deltas)}")
            worse += 1
        elif b:
            print(f"  ▲ {name:30s} 变好: {', '.join(deltas)}  ← 确认后请更新基线")
            better += 1
        else:
            print(f"  ✓ {name:30s} scored={have['scored']} match={have['match']} "
                  f"missing={have['missing']} extra={have['extra']} "
                  f"整行全对={have['rows_full']}/{have['rows_total']}")
    print(f"  —— 变差 {worse} / 变好 {better}")
    if worse:
        print("  ✗ gold 分变差 ⇒ 回归失败。若是有意为之，逐条确认后更新 "
              f"{baseline} 并在 log.md 记下原因。")
        return 1
    if better:
        print("  ▲ 只变好、不算失败。但**别忘了更新基线**，否则下次改坏会被旧基线放过。")
    return 1 if bad else 0
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
