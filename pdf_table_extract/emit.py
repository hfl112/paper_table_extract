"""落盘。**csv / pandas 的唯一入口。**

铁律 #2：不静默丢表 —— 抽失败的表在 manifest 里也要有一行，csv_path 留空、notes 写原因。
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from .models import ExtractedTable, Label, ManifestRow

_SAFE = re.compile(r"[^a-z0-9]+")
_UNSAFE_NAME = re.compile(r"[^\w.\-]+", re.UNICODE)


def sanitize_name(name: str) -> str:
    """把 PDF 文件名清理成好用的目录名。

    实测需要处理的：`EAP in advanced gastric cancer..pdf` 的 stem 是
    `EAP in advanced gastric cancer.`（带空格、带尾点）——
    直接当目录名会很难在命令行里输入。
    """
    s = _UNSAFE_NAME.sub("_", name.strip())
    s = re.sub(r"_{2,}", "_", s).strip("._-")
    return s or "output"


def resolve_outdir(outdir: Path | None, prefix: str | None, pdf: Path) -> Path:
    """一个 PDF 一个文件夹：`<outdir>/<prefix>/`。

    outdir 缺省为当前目录，prefix 缺省为 PDF 文件名（已清理）。
    """
    root = outdir if outdir is not None else Path(".")
    return root / sanitize_name(prefix if prefix else pdf.stem)


def table_id(label: Label | None, page: int, seq: int) -> str:
    """CSV 文件名用的标识。有标号用标号，没有就用页码+序号。"""
    if label is not None:
        return _SAFE.sub("_", label.key).strip("_")
    return f"p{page:02d}_t{seq}"


def csv_name(table_id_: str, page: int) -> str:
    return f"p{page:02d}_{table_id_}.csv"


def write_table(outdir: Path, table: ExtractedTable, name: str) -> str:
    """写一张表的 CSV，返回相对 outdir 的路径。"""
    path = outdir / name
    with path.open("w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows(table.rows)
    return name


def to_manifest_row(
    table: ExtractedTable, *, table_id_: str, csv_path: str, notes_extra: list[str] | None = None
) -> ManifestRow:
    lab = table.candidate.label
    notes = list(table.notes) + list(notes_extra or [])
    if table.continued_from:
        notes.append(f"spans_pages(continued_into={table.continued_from})")
    return ManifestRow(
        table_id=table_id_,
        label=lab.raw if lab else "",
        page=table.candidate.page,
        caption=(lab.text[:200] if lab else ""),
        csv_path=csv_path,
        extractor=table.extractor,
        source_type=table.candidate.source_type,
        matched_on=table.matched_on,
        matched_keywords=";".join(table.matched_keywords),
        n_rows=table.n_rows,
        n_cols=table.n_cols,
        coverage=("" if table.coverage is None else f"{table.coverage:.2f}"),
        plumber_cols=("" if table.plumber_cols is None else str(table.plumber_cols)),
        confidence=table.confidence,
        grid_status=table.grid_status,
        notes="; ".join(notes),
    )


def failure_row(
    *, table_id_: str, label: Label | None, page: int, reason: str, source_type: str = ""
) -> ManifestRow:
    """抽取失败或被跳过的表也必须有一行（铁律 #2）。"""
    return ManifestRow(
        table_id=table_id_,
        label=label.raw if label else "",
        page=page,
        caption=(label.text[:200] if label else ""),
        csv_path="",
        extractor="",
        source_type=source_type,
        matched_on="",
        matched_keywords="",
        n_rows=0,
        n_cols=0,
        coverage="",
        plumber_cols="",
        confidence="",
        grid_status="",
        notes=reason,
    )


def describe_query(keywords: list[str], *, match_all: bool, export_all: bool) -> str:
    """把这次运行的检索条件压成一个稳定字符串 —— manifest 的 query 列兼去重键。"""
    if export_all:
        return "--all"
    joined = ";".join(sorted(keywords))
    return f"{joined} (AND)" if match_all else joined


def write_manifest(outdir: Path, rows: list[ManifestRow], *, query: str) -> Path:
    """**累加式**写 manifest：保留其他 query 的历史行，只替换本次 query 的行。

    为什么累加而不是覆盖：同一篇 PDF 常要用不同关键词反复搜，覆盖式会把上次的
    记录冲掉，逼得用户为了留档去手工起 --prefix —— 那是把工具的问题转嫁给用户。
    累加后"用 a 搜到 table1、用 b 也搜到 table1"自然就是两行，各带自己的 query。

    去重键是 `query` 而不是 `matched_keywords`：前者标识**一次运行**，
    所以同一条命令重跑是更新而非堆积；后者只是那张表实际命中了哪几个词。
    """
    path = outdir / "manifest.csv"
    kept: list[list[str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for old in reader:
                if old.get("query", "") == query:
                    continue  # 同一次检索条件重跑 → 用新结果替换旧的
                kept.append([old.get(c, "") for c in ManifestRow.COLUMNS])

    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(ManifestRow.COLUMNS)
        w.writerows(kept)
        for r in rows:
            w.writerow([getattr(r, c) for c in ManifestRow.COLUMNS])
    return path


def print_summary(
    rows: list[ManifestRow],
    outdir: Path,
    *,
    manifest: Path | None = None,
    query: str = "",
) -> None:
    """人可读的摘要。低置信度和红旗要显眼 —— 铁律 #5：降级必须可见。"""
    ok = [r for r in rows if r.csv_path]
    bad = [r for r in rows if not r.csv_path]
    print(f"\n输出目录: {outdir}")
    if query:
        print(f"本次检索: {query}")
    print(f"导出 {len(ok)} 张表" + (f"，另有 {len(bad)} 张未能导出（见 manifest）" if bad else ""))
    if ok:
        print(f"\n{'表':22s} {'页':>3s} {'形状':>9s} {'置信':7s} {'来源':6s} {'命中处':11s} 备注")
        print("-" * 104)
        for r in sorted(ok, key=lambda x: x.page):
            shape = f"{r.n_rows}x{r.n_cols}"
            print(
                f"{r.table_id[:22]:22s} {r.page:3d} {shape:>9s} {r.confidence:7s} "
                f"{r.source_type:6s} {r.matched_on:11s} {r.notes[:34]}"
            )
    # 闸门挡掉的"图里没有表格网格"通常一篇好几张（多面板图每个 panel 一张位图），
    # 逐条打会把摘要刷满。它们仍然逐行写进 manifest.csv（铁律 #2：不静默丢表），
    # 这里只做聚合，并保留有标号的那些逐条显示。
    gated = [r for r in bad if "未检出表格网格" in r.notes]
    gated_named = [r for r in gated if r.label]
    gated_anon = [r for r in gated if not r.label]
    for r in bad:
        if r in gated:
            continue
        print(f"  [未导出] {r.label or r.table_id} p{r.page}: {r.notes}")
    for r in gated_named:
        print(f"  [未导出] {r.label} p{r.page}: 图内未检出表格网格，判定为纯图表，跳过 OCR")
    if gated_anon:
        pages = sorted({r.page for r in gated_anon})
        print(
            f"  [未导出] 另有 {len(gated_anon)} 张无标号位图未检出表格网格"
            f"（p{','.join(map(str, pages))}），判定为纯图表 —— 明细见 manifest"
        )

    low = [r for r in ok if r.confidence == "low"]
    if low:
        print(
            f"\n注意: {len(low)} 张为 low 置信度（OCR 派生或位于 OCR 文字层页），"
            f"建议人工核对。用 --dump-images 可导出原图对照。"
        )
    if manifest is not None:
        print(
            f"\nmanifest: {manifest}"
            f"\n  （累加式：换关键词再跑同一篇会**追加**行、不覆盖历史。"
            f"每行的 query 列记录那次搜的是什么，matched_on 列记录靠 caption 还是表内命中。）"
        )


def print_zero_hit_hint(labels: list[Label], query: str) -> None:
    """0 命中时把本篇**实际有什么**列出来。

    为什么必须做（用户实测点出的可用性问题）：
    `pbc_26870` 的 Table 1 其实是 PDX 表，但它的 caption **就是 `Table 1` 三个字**、
    表内用词是 `tumor line`；`pbc_30017` 的 Table 1 用的是 `osteosarcoma models`。
    搜 `-k xenograft -k PDX` 都是 0 张 —— 那是**词表不全**，不是本篇没有相关表。
    若只回一句"导出 0 张表"，用户会误以为这篇没东西。
    """
    tabs = [l for l in labels if l.kind == "table"]
    figs = [l for l in labels if l.kind == "figure"]
    if not tabs and not figs:
        return
    print(
        f"\n注意: 本篇其实有 {len(tabs)} 张表、{len(figs)} 张图，"
        f"只是都没命中 [{query}] —— 很可能是**用词不同**，不是没有相关内容："
    )
    for l in sorted(tabs + figs, key=lambda x: (x.kind != "table", x.page)):
        head = l.text[:74] if l.text.strip() != l.raw else f"{l.raw}（caption 无描述文字）"
        print(f"    {l.raw:22s} p{l.page:<3} {head}")
    print(
        "  下一步: 用 --list 看完整 caption/legend 挑词；"
        "或把同义词写进一个文件用 --keywords-file 一次带上。"
    )


def write_captions(outdir: Path, caps: list[Label], refs: list[Label]) -> Path:
    """--list 的机读输出。一行一个标号，含完整 caption/legend 原文。

    给下游脚本用：挑关键词、做词频、喂给别的工具。
    """
    path = outdir / "captions.csv"
    cols = ("label", "kind", "page", "present_in_pdf", "n_chars", "caption_legend")
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for l in sorted(caps, key=lambda x: (x.page, x.key)):
            w.writerow([l.raw, l.kind, l.page, "yes", len(l.text), l.text])
        for l in sorted(refs, key=lambda x: x.key):
            w.writerow([l.raw, l.kind, l.page, "no", 0, ""])
    return path
