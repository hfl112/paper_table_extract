#!/usr/bin/env python3
"""把 `01_paper_extract` 的 pptp collection 摊平成一个可用的语料目录。

为什么需要这一步：collection 里**每个 PDF 都叫 `article.pdf`**（真身在
`articles/<article_id>/article.pdf`）。而本项目的输出目录名来自
`emit.sanitize_name(pdf.stem)` —— 97 篇会全部变成 `article`，互相覆盖。
`eval/checks.py detect` 和 `eval/llm.py` 也都靠 stem 反查 PDF。
所以必须先摊平改名。

短名规则：**article_id 去掉 `doi_10_<注册号>_` 前缀**（即去掉前 3 个下划线段）。
这条规则是确定性的、不需要手工映射，且产出的名字与开发集现有命名同风格
（`blood_2014_12_618900` ↔ 开发集 `blood_2014_12_518900`）。

产物全部落在**仓库外** `~/Downloads/pte_pptp_gen/`：抽取产物含已发表文献的表格
原文，而本仓库现在**有 git 了**（`AGENTS.md` 里"本项目没有 git"那句已过期），
误提交一次就永久留在历史里。语料放仓库外沿用 `spike/build_threshold_corpus.py`
→ `~/Downloads/pte_threshold_corpus` 的先例。

    python gen/stage.py --dry-run    # 只自检不落盘
    python gen/stage.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from pathlib import Path

SRC = Path(
    "/Users/funanhe/00_MyCode/idea_generator/01_paper_extract/data/collections/pptp"
)
REPO = Path(__file__).resolve().parent.parent
DEV_DIR = REPO / "0.pdf_input"
BASE = Path.home() / "Downloads" / "pte_pptp_gen"

# ── 与开发集重合的 8 篇，必须排除 ──────────────────────────────────────────
# 逐篇核对过 DOI，不是靠文件名猜的。`blood_2014_12_618900` 对应开发集
# `blood_2014_12_518900.pdf` —— 开发集那个文件名里的 `518900` 是**笔误**，
# 该篇真实 DOI 是 `10.1182/blood-2014-12-618900`。只是文件名错，不影响行为，
# 且 `eval/expected.csv` 等多处引用它，**不要改**。
DEV_OVERLAP = {
    "doi_10_1002_pbc_21078": "pbc_21078.pdf",
    "doi_10_1002_pbc_21296": "pbc_21296.pdf",
    "doi_10_1002_pbc_24724": "pbc_24724.pdf",
    "doi_10_1002_pbc_28772": "pbc_28772.pdf",
    "doi_10_1002_pbc_29304": "pbc_29304.pdf",
    "doi_10_1158_1078_0432_ccr_18_2728": "1078-0432.CCR-18-2728.pdf",
    "doi_10_1101_566455": "566455.pdf",
    "doi_10_1182_blood_2014_12_618900": "blood_2014_12_518900.pdf",
}

# ── 深挖的 10 篇（用户确认：定向挑选，不随机）+ 1 个阳性对照 ────────────────
# `why` 是选它的理由，会写进 corpus.csv，跑完对照着看。
DEEP10: dict[str, str] = {
    "pbc_26825": "基线：PBC 2017+ 新模板，TABLE 1-4 含 (Continued)",
    "pbc_22188": "横向转正 547 竖排行；Xenograft 在首列表头不在 caption",
    "pbc_22576": "横向转正 501 竖排行，2 张表",
    "pbc_22741": "横向转正 435 竖排行，2 张表",
    "s00280_011_1618_8": "Springer + 新续表写法 'Table 2 continued'（无括号）",
    "0008_5472_can_16_0122": "Cancer Res + 第三种续表写法 \"(Cont'd )\"",
    "1078_0432_ccr_18_2675": "表最多（7 张）+ 临床试验型表格（患者特征/毒性）",
    "blood_2016_03_707414": "Blood 模板 + 色块编码表（F-020 风险）",
    "pbc_22921": "同篇内精确性对照：TABLE II/III 该中，I/IV 不该中；Dose–Response 用 U+2013",
    "j_pharmthera_2024_108742": "最大压力：68 页综述、9 张汇总大表、竖排 668",
}

# 半独立的第 11 篇：它是开发集 preprint `566455` 的正式发表版（标题逐字相同）。
# 不计入 10 篇统计，只用来看"preprint 拒跑 / 正式版正常抽"这组配对。
SEMI = {"j_celrep_2019_09_071": "566455 的正式发表版；Cell Press 版式；只有 Table S1 标号"}

# 阳性对照：开发集成员，正确答案已钉在 `eval/expected.csv`
# （`xeno_29304 p03_table_1.csv` 42 行 = 41 数据行，`matched_on=caption`，`high`）。
# 走**完全相同**的摊平路径。理由见 log.md §35.1：那次 `timeout` 在 macOS 上不存在，
# 25 个输出目录全空，而"这批料没有图片表"这个结论看起来完全合理。
# 任何"0 命中"的说法都必须有同批阳性对照垫底。
POSCTL = {"posctl_pbc_29304": "阳性对照：开发集 pbc_29304，正确值已钉在 expected.csv"}


def short_name(article_id: str) -> str:
    """`doi_10_1002_pbc_26825` → `pbc_26825`（去掉 `doi_10_<注册号>_`）。"""
    parts = article_id.split("_")
    if len(parts) > 3 and parts[0] == "doi":
        return "_".join(parts[3:])
    return article_id


def md5(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def die(msg: str) -> None:
    print(f"✗ {msg}", file=sys.stderr)
    sys.exit(1)


def link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="只自检，不落盘")
    ap.add_argument("--copy", action="store_true", help="复制而非 symlink（引擎拒收 symlink 时用）")
    args = ap.parse_args()

    if not SRC.exists():
        die(f"语料目录不存在: {SRC}")
    rows = list(csv.DictReader((SRC / "articles.csv").open()))

    # ── 自检 1：PDF 都在 ───────────────────────────────────────────────
    have = {}
    for r in rows:
        p = SRC / "articles" / r["article_id"] / "article.pdf"
        if p.exists():
            have[r["article_id"]] = (p, r)
    print(f"articles.csv {len(rows)} 行，其中有 PDF 的 {len(have)} 篇")
    if len(have) != 105:
        die(f"预期 105 篇有 PDF，实际 {len(have)} —— 语料变了，先查清再跑")

    # ── 自检 2：8 篇重合的必须与开发集逐字节相同 ───────────────────────
    # 这一条防的是"语料被重新抓过、内容已经不是当初那份"。
    for aid, devname in DEV_OVERLAP.items():
        if aid not in have:
            die(f"重合清单里的 {aid} 不在语料里 —— 清单或语料有一个错了")
        dev = DEV_DIR / devname
        if not dev.exists():
            print(f"  ▲ 开发集缺 {devname}，跳过 md5 比对")
            continue
        a, b = md5(have[aid][0]), md5(dev)
        flag = "✓" if a == b else "✗ 不同"
        print(f"  {flag} {aid:38} ↔ {devname}")
        if a != b:
            die(f"{aid} 与开发集 {devname} 内容不同 —— 不能假定是同一篇")

    # ── 自检 3：短名唯一、不与开发集撞 ─────────────────────────────────
    cand = {a: v for a, v in have.items() if a not in DEV_OVERLAP}
    print(f"排除重合后候选 {len(cand)} 篇")
    if len(cand) != 97:
        die(f"预期 97 篇候选，实际 {len(cand)}")

    sys.path.insert(0, str(REPO))
    from pdf_table_extract.emit import sanitize_name

    seen: dict[str, str] = {}
    for aid in cand:
        s = sanitize_name(short_name(aid))
        if s in seen:
            die(f"短名冲突: {aid} 与 {seen[s]} 都变成 {s}")
        seen[s] = aid

    dev_stems = {sanitize_name(p.stem) for p in DEV_DIR.glob("*.pdf")}
    for name in list(DEEP10) + list(SEMI):
        s = sanitize_name(name)
        if s not in seen:
            die(f"选中的 {name} 不在候选里（短名规则或清单错了）")
        if s in dev_stems:
            die(f"选中的 {name} 的输出目录名与开发集撞了")

    # ── 自检 4：语料内没有重复 PDF ─────────────────────────────────────
    sums: dict[str, str] = {}
    dupes = []
    for aid, (p, _) in cand.items():
        m = md5(p)
        if m in sums:
            dupes.append((aid, sums[m]))
        sums[m] = aid
    if dupes:
        for a, b in dupes:
            print(f"  ▲ 内容重复: {a} == {b}")
    print(f"语料内重复 PDF: {len(dupes)} 组")

    # ── 自检 5：fitz 打得开 ────────────────────────────────────────────
    import warnings

    warnings.filterwarnings("ignore")
    import fitz

    pages: dict[str, int] = {}
    for aid, (p, _) in cand.items():
        try:
            d = fitz.open(p)
            pages[aid] = len(d)
            d.close()
        except Exception as e:  # noqa: BLE001
            die(f"fitz 打不开 {aid}: {e}")
    print(f"fitz 全部可打开，页数 {min(pages.values())}–{max(pages.values())}")

    if args.dry_run:
        print("\n--dry-run：自检全过，未落盘")
        return 0

    # ── 落盘 ───────────────────────────────────────────────────────────
    sweep = BASE / "sweep97"
    deep = BASE / "deep10"
    for d in (sweep, deep, BASE / "list97", BASE / "logs", BASE / "png"):
        d.mkdir(parents=True, exist_ok=True)

    import shutil

    put = (lambda s, d: shutil.copy2(s, d)) if args.copy else link

    for aid, (p, _) in cand.items():
        put(p, sweep / f"{aid}.pdf")
    for name in list(DEEP10) + list(SEMI):
        put(have[seen[sanitize_name(name)]][0], deep / f"{name}.pdf")
    # 阳性对照单独接：它来自重合的那 8 篇
    put(have["doi_10_1002_pbc_29304"][0], deep / "posctl_pbc_29304.pdf")

    with (BASE / "corpus.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            "article_id staged_name tier doi journal pub_year md5 n_pages why title".split()
        )
        for aid, (p, r) in sorted(cand.items()):
            s = sanitize_name(short_name(aid))
            tier = "deep10" if s in DEEP10 else ("semi" if s in SEMI else "sweep97")
            why = DEEP10.get(s) or SEMI.get(s) or ""
            w.writerow(
                [aid, short_name(aid), tier, r["doi"], r["journal"], r["pub_year"],
                 md5(p), pages[aid], why, r["title"]]
            )
        pc = have["doi_10_1002_pbc_29304"]
        w.writerow(["doi_10_1002_pbc_29304", "posctl_pbc_29304", "posctl",
                    pc[1]["doi"], pc[1]["journal"], pc[1]["pub_year"],
                    md5(pc[0]), 0, POSCTL["posctl_pbc_29304"], pc[1]["title"]])

    n_sweep = len(list(sweep.glob("*.pdf")))
    n_deep = len(list(deep.glob("*.pdf")))
    print(f"\n落盘完成（{'复制' if args.copy else 'symlink'}）")
    print(f"  {sweep}  {n_sweep} 篇")
    print(f"  {deep}  {n_deep} 篇（10 深挖 + 1 半独立 + 1 阳性对照）")
    print(f"  {BASE / 'corpus.csv'}")
    if n_sweep != 97 or n_deep != 12:
        die(f"落盘数量不对：sweep97={n_sweep}(期望97) deep10={n_deep}(期望12)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
