"""检查产物。两个子命令，都不需要人工 gold。

    conda run -n gemini python eval/checks.py detect     1.output --pdf-dir 0.pdf_input
    conda run -n gemini python eval/checks.py regression /tmp/pte_regression eval/expected.csv

- `detect` —— **零标注判据**：拿 PDF 自己当参照，找出抽错的表。
- `regression` —— **现状快照对账**：拿 `eval/expected.csv` 比 `eval/regression.sh` 的产物。

两者都是"检查产物"，只是参照物不同（一个是 PDF 文字层，一个是上次跑出来的快照）。

═══ detect：为什么能零标注 ═══

文字表的格子是从 PDF 文字层**切**出来的、不是**认**出来的，所以每个格子的字符串必然能在
文字层里逐字找到。文字层就在 PDF 里，不用人抄。图片表不适用（那是 OCR 认的），所以文字层
判据只对 `source_type=text` 生效。

**要求 `<outdir>/<prefix>/` 的 prefix 等于 PDF 文件名 stem**（`eval/matrix.sh` 与
`holdout/run.sh` 都满足；`eval/regression.sh` 用的是 `resp_xxx` 这类自定义 prefix，不适用）。

三条判据，实测依据见 `eval/findings.md`：

1. **区域文字层逐字比**（只对文字表）
   作用域必须是「表所在的页」，不能是全文档 —— 全文档会让 OCR 垃圾"验证通过"：
   `pbc_21078 fig_3` 按全文档 92.6% 命中、按本页只有 56.1%，而它的标签在文字层里一个都不存在
   （矢量轮廓）。表头行有一条豁免（多级表头被 docling 用 `.` 拼平），见 `_header_rescue`。
2. **数值列并行检测**（docling 把两个数据行挤进一格）
   某列 >=70% 的格子是单个数值 ⇒ 数值列；某行有 >=2 个数值格装了 >=2 个数 ⇒ 疑似并行。
   实测 15 张表约 580 行，命中 1 处（`pbc_21296` 行 41，真 bug）、零误报。
3. **全空行**
   `pbc_24724 fig_1` 末尾 8 行、`pbc_21078 fig_3` 1 行（F-005）。

**阈值诚实分档**：`TEXT_LAYER_MIN_HIT=0.90` 全语料两侧是 74.3% ← 90% → 95.5%（21.2 个百分点，
**负例只有 2 个数据点**，算偏紧）；`NUM_COL_RATIO=0.7` 与 `MIN_MULTI_CELLS=2` 是**单点拟合**
（正例 1 个、负例 579 行）。

═══ regression：为什么断言的是「行为没变」而不是「行为正确」═══

`eval/expected.csv` 是**现状快照**（见 `eval/known_issues.md`）—— 里面至少有一条记的是已知的
错误行为（`resp_21296/p04_table_i.csv` 44 行，真值应为 45）。冻结期这样才对：断言器的职责是
"改代码别改坏"，不是"判定对错"。修 bug 时把对应条目从 known_issues 挪进 expected 并改数字。

为什么需要它（而不是像原来那样 printf 一屏行数让人眼比对）：**人眼对「不出现」最不敏感**。
一张 CSV 整个消失，摘要里只是少一行；而你在核对"有的这些行数对不对"的时候不会注意到少了
一行。本项目历史上三个由用户人眼发现的 bug 全是这个形状（`log.md` §5）。

单测在 `tests/test_eval.py`（纯逻辑，不碰 PDF）。
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# —— 判据 1 ——
TEXT_LAYER_MIN_HIT = 0.90  # 文字表 98.4-100% vs 图片表 24.6-56.1%
# —— 判据 2 ——
NUM_COL_RATIO = 0.7   # 一列里至少这么多格子是单个数值，才算数值列
NUM_COL_MIN_CELLS = 4  # 太少的列不判
MIN_MULTI_CELLS = 2   # 一行里至少这么多数值格装了多个数，才报并行

_NUM = re.compile(r"[<>~≥≤]?\s*-?\d+(?:[.,]\d+)?%?")
_NUM_FULL = re.compile(r"[<>~≥≤]?\s*-?\d+(?:[.,]\d+)?%?\Z")
_PM = re.compile(r"±\s*-?\d+(?:[.,]\d+)?")          # `2.814 ± 0.516` 是一个值，不是两个
_FOOTNOTE_TAIL = re.compile(r"\s+[a-z]\Z")           # 表头尾部的上标脚注标记：`Median final RTV d`


def norm(s: str) -> str:
    """折叠空白、拆连字、统一各种横线、丢掉控制字符与不可见符号。

    三条都是实测逼出来的，少一条就有假报警：

    - **NFKC 拆连字**：`pbc_21078` p6 的文字层里是 `ﬁnal RTV`（U+FB01 连字），而 docling 抽出的
      格子是 `final RTV`（普通 f+i）。不拆连字，`Median final RTV` 在 3 张表上都对不上。
    - **去控制字符**：`pbc_21078 table_ii/table_iii` 各有 14 个 `\\x01`（符号字体的对钩类字形），
      文字层里不以该码位出现。
    - **统一横线**：`−`(U+2212) / `–` / `—` 与 ASCII `-` 混用。
    """
    s = unicodedata.normalize("NFKC", s)
    for a, b in (("−", "-"), ("–", "-"), ("—", "-"), ("‐", "-")):
        s = s.replace(a, b)
    s = "".join(ch for ch in s if ch.isspace() or unicodedata.category(ch)[0] not in ("C", "Z"))
    return re.sub(r"\s+", "", s).lower()


def checkable(cell: str) -> bool:
    """值得拿去比对的格子：至少含一个字母或数字。纯符号格（对钩、破折号）不算。"""
    return bool(re.search(r"[a-z0-9]", norm(cell)))


@dataclass
class Finding:
    kind: str
    csv_path: str
    detail: str


@dataclass
class TableCheck:
    csv_path: Path
    prefix: str
    pages: list[int]
    source_type: str
    confidence: str
    notes: str
    n_data_rows: int = 0
    text_layer_hit: float | None = None
    findings: list[Finding] = field(default_factory=list)


# ————————————————————— 判据 1：区域文字层逐字比 —————————————————————

def _header_rescue(cell: str, page_text: str) -> bool:
    """多级表头被 docling 用 `.` 拼平了 —— 拆开后每段都在文字层里就算命中。

    实测（留出集 `Bao et al. 2025`，开发集 12 篇一个这样的样本都没有）：
    原表是两层表头 `Training` / `Cancer` + `n =2,254`，docling 拼成
    `Training.Cancer.n =2,254`。整串当然不在文字层里，但三段都在。

    **单层表头也会被加上分隔符**：留出集 `Bruhm 2025` 的横向表 Table 3 表头是
    `Sponsor.` / `Cancertype.` / `Studyname.` —— 父层为空、只剩一个尾点。所以要求 `>=1` 段
    而不是 `>=2`，否则这类整行 16 个表头格全部误报（实测把该表压到 83.9%、跌破硬阈值）。

    **只对表头行（第 0 行）用这条豁免。** 数据行不能用 —— `pbc_21296` 那个真 bug
    （`< 0.001 < 0.001`）按 `.` 拆开后每段也可能碰巧命中，用了就把真错放过了。
    """
    parts = [p for p in cell.split(".") if norm(p)]
    return bool(parts) and all(norm(p) in page_text for p in parts)


def check_text_layer(rows: list[list[str]], page_text: str) -> tuple[float, list[str]]:
    """返回 (命中率, 未命中的格子)。只对文字表调用。"""
    if not rows:
        return 1.0, []
    total = 0
    miss: list[str] = []
    for ri, row in enumerate(rows):
        for c in row:
            if not c.strip() or not checkable(c):
                continue
            total += 1
            n = norm(c)
            if n in page_text:
                continue
            # 表头尾部的上标脚注标记：`Median final RTV d` 里的 d 是上标，跟正文分开
            stripped = norm(_FOOTNOTE_TAIL.sub("", c))
            if stripped and stripped in page_text:
                continue
            if ri == 0 and _header_rescue(c, page_text):
                continue
            miss.append(c)
    if not total:
        return 1.0, []
    return (total - len(miss)) / total, miss


# ————————————————————— 判据 2：数值列并行检测 —————————————————————

def _n_values(cell: str) -> int:
    return len(_NUM.findall(_PM.sub("", cell)))


def _is_num_col(vals: list[str]) -> bool:
    v = [x.strip() for x in vals if x.strip()]
    if len(v) < NUM_COL_MIN_CELLS:
        return False
    return sum(1 for x in v if _NUM_FULL.match(x.replace(" ", ""))) >= len(v) * NUM_COL_RATIO


def check_merged_rows(rows: list[list[str]]) -> list[tuple[int, list[tuple[str, str]]]]:
    """某行有 >=MIN_MULTI_CELLS 个数值格装了多个数 ⇒ 疑似两个数据行被压成一格。

    为什么用「数值列」而不是所有列：文本列里出现两个词是正常的（`ALL B-precursor`），
    而数值列里出现两个数几乎只可能是并行。这条不含任何领域知识。
    """
    if len(rows) < 4:
        return []
    header, data = rows[0], rows[1:]
    ncol = len(header)
    num_cols = [
        c for c in range(ncol)
        if _is_num_col([r[c] for r in data if c < len(r)])
    ]
    out = []
    for i, r in enumerate(data):
        hits = [
            (header[c][:18], r[c])
            for c in num_cols
            if c < len(r) and _n_values(r[c]) >= 2
        ]
        if len(hits) >= MIN_MULTI_CELLS:
            out.append((i, hits))
    return out


# ————————————————————— 判据 3：全空行 —————————————————————

def check_blank_rows(rows: list[list[str]]) -> list[int]:
    return [i for i, r in enumerate(rows[1:]) if not any(c.strip() for c in r)]


# ————————————————————— 装配 —————————————————————

_SPANS = re.compile(r"spans_pages=(\d+)-(\d+)")


def load_manifest(prefix_dir: Path) -> dict[str, dict]:
    man = prefix_dir / "manifest.csv"
    if not man.exists():
        return {}
    out: dict[str, dict] = {}
    with man.open(newline="") as fh:
        for row in csv.DictReader(fh):
            name = Path(row.get("csv_path", "") or "").name
            if name and name not in out:
                out[name] = row
    return out


def page_text_of(pdf: Path, pages: list[int]) -> str:
    import fitz  # 惰性 import：纯逻辑函数的单测不该拖进 PyMuPDF

    doc = fitz.open(pdf)
    try:
        return norm("".join(doc[p - 1].get_text() for p in pages if 0 < p <= len(doc)))
    finally:
        doc.close()


def check_one(csv_path: Path, meta: dict, pdf: Path) -> TableCheck:
    rows = list(csv.reader(csv_path.open(newline="")))
    pages = [int(meta.get("page") or csv_path.name[1:3])]
    m = _SPANS.search(meta.get("notes", ""))
    if m:
        pages = list(range(int(m.group(1)), int(m.group(2)) + 1))

    tc = TableCheck(
        csv_path=csv_path,
        prefix=csv_path.parent.name,
        pages=pages,
        source_type=meta.get("source_type", ""),
        confidence=meta.get("confidence", ""),
        notes=meta.get("notes", ""),
        n_data_rows=max(len(rows) - 1, 0),
    )

    if tc.source_type == "text":
        hit, miss = check_text_layer(rows, page_text_of(pdf, pages))
        tc.text_layer_hit = hit
        if hit < TEXT_LAYER_MIN_HIT:
            tc.findings.append(Finding(
                "text_layer", str(csv_path),
                f"只有 {hit:.1%} 的格子能在 p{pages} 文字层里找到（阈值 {TEXT_LAYER_MIN_HIT:.0%}）",
            ))
        elif miss:
            tc.findings.append(Finding(
                "text_layer_partial", str(csv_path),
                f"{len(miss)} 个格子不在文字层里: {[m[:26] for m in miss[:4]]}",
            ))

    for i, hits in check_merged_rows(rows):
        tc.findings.append(Finding(
            "merged_row", str(csv_path),
            f"数据行 {i} 疑似两行压一格 —— {len(hits)} 个数值格装了多个数: "
            + "; ".join(f"{h}={v[:18]!r}" for h, v in hits[:4]),
        ))

    blanks = check_blank_rows(rows)
    if blanks:
        tc.findings.append(Finding(
            "blank_rows", str(csv_path), f"{len(blanks)} 个数据行全空: 行 {blanks[:8]}",
        ))

    return tc



# ————————————————————— 快照对账 —————————————————————

KIND_TABLE = "table"        # 该 CSV 必须存在，且 rows/matched_on/confidence 相符
KIND_NO_TABLES = "no_tables"  # 该 prefix 下不得有任何表 CSV（manifest/captions 不算）
KIND_EXIT = "exit_code"     # 该 prefix 的命令必须以指定退出码结束（regression.sh 写进 $OUT/.exit）


def load_expected(path: Path, tier: str = "full") -> list[dict]:
    """tier=fast 只取快档（不跑 OCR 的那些，约 2 分钟）；tier=full 取全部。

    为什么分档：全量 21 条命令要 **424 秒**，其中 12 条在跑 OCR —— 而 OCR 路径的产出
    本来就是 `low`、本来就要人核，花 5 分钟钉住一批"反正不可信"的数字收益接近零。
    快档 10 条 **121 秒**，覆盖了文字表的全部路径：跨页拼接、横向转正、合并单元格、
    出版社 OCR 文字层、两个拒跑退出码。
    """
    with path.open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("kind", "").strip()]
    if tier == "fast":
        rows = [r for r in rows if r.get("tier", "full").strip() == "fast"]
    return rows


def actual_tables(out: Path) -> dict[tuple[str, str], Path]:
    """扫出实际产出的表 CSV，键是 (prefix, 文件名)。manifest/captions 不算表。"""
    found: dict[tuple[str, str], Path] = {}
    for p in sorted(out.rglob("*.csv")):
        if p.name in ("manifest.csv", "captions.csv"):
            continue
        found[(p.parent.name, p.name)] = p
    return found


def manifest_row(out: Path, prefix: str, csv_name: str) -> dict | None:
    """从 manifest 里找到写出这个 CSV 的那一行，用来核 matched_on / confidence。

    manifest 是累加式的（同一张表可能被多个 query 命中，各一行），所以可能有多行指向同一个
    CSV。这里返回第一行 —— 回归脚本里每个 prefix 只跑一条命令，不会出现多 query。
    """
    man = out / prefix / "manifest.csv"
    if not man.exists():
        return None
    with man.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if Path(row.get("csv_path", "")).name == csv_name:
                return row
    return None


def exit_codes(out: Path) -> dict[str, int]:
    """regression.sh 把负面用例的退出码写在 $OUT/.exit，一行 `prefix code`。"""
    f = out / ".exit"
    if not f.exists():
        return {}
    codes: dict[str, int] = {}
    for line in f.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("-").isdigit():
            codes[parts[0]] = int(parts[1])
    return codes



def run_regression(out: Path, exp_path: Path, tier: str = "full") -> int:
    expected = load_expected(exp_path, tier)
    found = actual_tables(out)
    codes = exit_codes(out)
    bad: list[str] = []
    accounted: set[tuple[str, str]] = set()

    for e in expected:
        kind, prefix = e["kind"].strip(), e["prefix"].strip()

        # ① 退出码
        if kind == KIND_EXIT:
            want = int(e["rows"])
            got = codes.get(prefix)
            if got is None:
                bad.append(f"{prefix:16s} 期望退出码 {want}，但 {out}/.exit 里没有记录")
            elif got != want:
                bad.append(f"{prefix:16s} 退出码 {want} → {got}")
            continue

        # ② 该 prefix 不该产出任何表
        if kind == KIND_NO_TABLES:
            extra = sorted(n for (p, n) in found if p == prefix)
            for n in extra:
                accounted.add((prefix, n))
            if extra:
                bad.append(f"{prefix:16s} 期望 0 张表，实际产出 {len(extra)} 张: {extra}")
            continue

        # ③ 该 CSV 必须存在且形状/标记相符
        name = e["csv"].strip()
        key = (prefix, name)
        accounted.add(key)
        path = found.get(key)
        if path is None:
            bad.append(f"{prefix}/{name} 期望存在，实际**缺失**（漏表）")
            continue

        want_rows = int(e["rows"])
        got_rows = sum(1 for _ in path.open(newline=""))
        if got_rows != want_rows:
            bad.append(f"{prefix}/{name} 行数 {want_rows} → {got_rows}")

        want_on, want_conf = e.get("matched_on", "").strip(), e.get("confidence", "").strip()
        if want_on or want_conf:
            row = manifest_row(out, prefix, name)
            if row is None:
                bad.append(f"{prefix}/{name} manifest 里找不到对应行")
            else:
                if want_on and row.get("matched_on", "") != want_on:
                    bad.append(
                        f"{prefix}/{name} matched_on {want_on!r} → {row.get('matched_on')!r}"
                    )
                if want_conf and row.get("confidence", "") != want_conf:
                    bad.append(
                        f"{prefix}/{name} confidence {want_conf!r} → {row.get('confidence')!r}"
                    )

    # ④ 期望之外多出来的表（误报）
    # 快档只跑了一部分命令，所以只在**快档涉及的 prefix 内**查多余项 ——
    # 否则上一次全量跑剩下的产物会被当成"多出来的"。
    fast_prefixes = {e["prefix"].strip() for e in expected}
    for (prefix, name) in sorted(found):
        if (prefix, name) in accounted:
            continue
        if tier == "fast" and prefix not in fast_prefixes:
            continue
        bad.append(f"{prefix}/{name} **不在期望清单里**（误报或新增，需确认后写进 expected.csv）")

    n_tab = sum(1 for e in expected if e["kind"].strip() == KIND_TABLE)
    tier_note = "（快档）" if tier == "fast" else "（全量）"
    if bad:
        print(f"✗ 回归不符 {len(bad)} 处{tier_note}（期望 {n_tab} 张表 + {len(expected) - n_tab} 条其他约束）")
        for b in bad:
            print(f"    {b}")
        print("\n  行为变了。若是有意为之，改 eval/expected.csv；若不是，就是回归。")
        return 1

    print(f"✓ 回归全绿{tier_note}：{n_tab} 张表 + {len(expected) - n_tab} 条其他约束，行为与快照一致")
    print("  注意：这断言的是「行为没变」，不是「行为正确」—— 已知错误见 eval/known_issues.md")
    return 0



# ————————————————————— CLI —————————————————————

def cmd_detect(args: argparse.Namespace) -> int:
    # 用包自己的清理函数反查 prefix，不要猜规则 —— 实测 `Bao et al. - 2025 - ...` 会被清成
    # `Bao_et_al._-_2025_-_...`（保留 `.` 和 `-`），跟"非字母数字全换下划线"的猜测对不上。
    from pdf_table_extract.emit import sanitize_name

    stems: dict[str, Path] = {}
    for p in args.pdf_dir.glob("*.pdf"):
        stems.setdefault(p.stem, p)
        stems.setdefault(sanitize_name(p.stem), p)

    checks: list[TableCheck] = []
    unmatched: list[str] = []
    for prefix_dir in sorted(d for d in args.outdir.iterdir() if d.is_dir()):
        pdf = stems.get(prefix_dir.name)
        if pdf is None:
            unmatched.append(prefix_dir.name)
            continue
        manifest = load_manifest(prefix_dir)
        for csv_path in sorted(prefix_dir.glob("*.csv")):
            if csv_path.name in ("manifest.csv", "captions.csv"):
                continue
            checks.append(check_one(csv_path, manifest.get(csv_path.name, {}), pdf))

    hdr = "%-38s %-5s %-7s %6s %8s  %s"
    print(hdr % ("CSV", "类型", "置信度", "数据行", "文字层", "判据命中"))
    print("-" * 118)
    for tc in sorted(checks, key=lambda t: str(t.csv_path)):
        if args.quiet and not tc.findings:
            continue
        hit = "—" if tc.text_layer_hit is None else f"{tc.text_layer_hit:.1%}"
        kinds = ",".join(sorted({f.kind for f in tc.findings})) or "-"
        rel = str(tc.csv_path.relative_to(args.outdir))
        print(hdr % (rel[:38], tc.source_type or "?", tc.confidence or "?",
                     tc.n_data_rows, hit, kinds))

    flagged = [t for t in checks if t.findings]
    print("-" * 118)
    print(f"共 {len(checks)} 张表，{len(flagged)} 张命中判据")
    if unmatched:
        print(f"⚠ {len(unmatched)} 个 prefix 找不到对应 PDF（跳过）: {unmatched[:6]}")

    if flagged:
        print("\n明细：")
        for tc in flagged:
            print(f"\n  {tc.csv_path.relative_to(args.outdir)}"
                  f"  [{tc.source_type}/{tc.confidence}]  notes={tc.notes or '(空)'}")
            for f in tc.findings:
                print(f"      · {f.kind}: {f.detail}")

    # 最危险的一档：标了 high、notes 干净、却被判据命中 —— 没举手的错
    silent = [t for t in flagged
              if t.confidence == "high" and not t.notes
              and any(f.kind in ("text_layer", "merged_row") for f in t.findings)]
    if silent:
        print(f"\n⚠⚠ {len(silent)} 张表 confidence=high 且 notes 为空，却被判据命中 —— "
              f"**没举手的错**，危险度最高：")
        for t in silent:
            print(f"      {t.csv_path.relative_to(args.outdir)}")
    return 0


def cmd_regression(args: argparse.Namespace) -> int:
    return run_regression(args.outdir, args.expected, args.tier)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="零标注判据：拿 PDF 文字层当参照找抽错的表")
    d.add_argument("outdir", type=Path, help="输出根目录（prefix 须等于 PDF stem）")
    d.add_argument("--pdf-dir", type=Path, required=True)
    d.add_argument("--quiet", action="store_true", help="只打有问题的表")
    d.set_defaults(func=cmd_detect)

    r = sub.add_parser("regression", help="现状快照对账，不符 exit 1")
    r.add_argument("outdir", type=Path)
    r.add_argument("expected", type=Path, nargs="?", default=Path("eval/expected.csv"))
    r.add_argument("--tier", choices=("fast", "full"), default="full",
                   help="fast=只对账不跑 OCR 的那批（约 2 分钟）；full=全部（约 7 分钟）")
    r.set_defaults(func=cmd_regression)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
