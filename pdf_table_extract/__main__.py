"""CLI 入口与流程编排。业务规则都在 rules.py，这里只负责串。

退出码：
  0  正常
  2  参数错误
  3  判定为 preprint / peer-review 版，拒跑（铁律 #3）
  4  全文扫不到任何标号，拒跑（铁律 #3：不静默返回空）
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path

from . import quiet

quiet.silence()  # 必须在任何引擎 import 之前

from . import emit, engine_docling, engine_plumber, pdfio, rules  # noqa: E402
from .models import DocInfo, ExtractedTable, ImageInfo, Label, ManifestRow, Rect, TableCandidate

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_PREPRINT = 3
EXIT_NO_LABELS = 4


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pdf-table-extract",
        description="给一篇英文文献 PDF 和一组关键词，把命中的表格（含图片形式的表格）导出成 CSV。",
    )
    p.add_argument("pdf", type=Path, help="输入 PDF")
    p.add_argument(
        "-k", "--keyword", action="append", default=[], metavar="WORD", help="关键词，可重复；默认 OR"
    )
    p.add_argument("--keywords-file", type=Path, help="从文件读关键词，一行一个")
    p.add_argument("--match-all", action="store_true", help="改成 AND：必须全部命中")
    p.add_argument("--all", action="store_true", help="忽略关键词导出全部；也是标号惯例不被识别时的逃生舱")
    p.add_argument("--list", action="store_true", help="只列出全部标号+caption+页码，不抽内容（几乎免费）")
    p.add_argument(
        "-o", "--outdir", type=Path, help="输出根目录，默认当前目录。实际产物在 <outdir>/<prefix>/"
    )
    p.add_argument("--prefix", help="本篇的子文件夹名，默认取 PDF 文件名（会清理空格等字符）")
    p.add_argument("--pages", help="限定页范围，如 3-8,12")
    p.add_argument("--no-ocr", action="store_true", help="跳过一切 OCR，只处理文字表")
    p.add_argument("--dpi", type=int, default=200, help="兜底渲染分辨率（优先抠原生图）")
    p.add_argument("--dump-images", action="store_true", help="同时导出表格原图 PNG，便于人工核对")
    p.add_argument("--label-pattern", help="覆盖默认标号正则（核心部分，不含锚点）")
    p.add_argument("--hint-words-file", type=Path, help="用户自备的领域词表（OCR 闸门放行用），默认空")
    p.add_argument("--allow-preprint", action="store_true", help="覆盖 preprint 拒跑")
    p.add_argument(
        "--substring", action="store_true",
        help="改回纯子串匹配（默认是全词+屈折变体）。短关键词会命中英文单词内部，慎用",
    )
    return p


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------


def load_keywords(args: argparse.Namespace) -> list[str]:
    words = list(args.keyword)
    if args.keywords_file:
        words += _read_words(args.keywords_file)
    return words


def _read_words(path: Path) -> list[str]:
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]


def parse_pages(spec: str | None, n_pages: int) -> set[int] | None:
    if not spec:
        return None
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, _, b = part.partition("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return {p for p in out if 1 <= p <= n_pages}


# ---------------------------------------------------------------------------
# --list
# ---------------------------------------------------------------------------


def _print_labels(title: str, labels: list[Label], ocr_pages: set[int], width: int) -> None:
    if not labels:
        return
    print()
    print(f"── {title}（{len(labels)} 个）" + "─" * max(0, width - len(title) - 12))
    for l in sorted(labels, key=lambda x: (x.page, x.key)):
        mark = " [OCR文字层页]" if l.page in ocr_pages else ""
        print(f"\n  {l.raw}   p{l.page}{mark}   ({len(l.text)} 字)")
        for line in textwrap.wrap(l.text, width=width - 6) or [""]:
            print(f"      {line}")


def cmd_list(args: argparse.Namespace) -> int:
    """--list：列出全部标号 + 完整 caption/legend。不抽内容、不跑 ML，几乎免费。"""
    info = pdfio.read_doc(args.pdf)
    labels = pdfio.scan_labels(args.pdf, args.label_pattern)
    caps = pdfio.caption_labels(labels)
    refs = pdfio.referenced_only(labels)

    ocr_pages = {p.page for p in info.pages if p.ocr_text_layer}
    rot_pages = [(p.page, p.needs_rotation) for p in info.pages if p.needs_rotation]
    width = 96

    print(f"文件: {args.pdf}")
    print(f"页数: {info.n_pages}   文字层总字符: {info.total_chars}")
    if info.preprint_markers:
        total = sum(info.preprint_markers.values())
        detail = ", ".join(f"{k}×{v}" for k, v in sorted(info.preprint_markers.items()))
        flag = "  ← 判定为 preprint，正式抽取会拒跑（--allow-preprint 覆盖）" if pdfio.is_preprint(info) else ""
        print(f"preprint 特征: {total} 次 ({detail}){flag}")
    if ocr_pages:
        print(f"OCR 文字层页（位图全页覆盖 ⇒ 抽出的表无条件 confidence=low）: {sorted(ocr_pages)}")
        if rules.metadata_suggests_ocr(info.metadata):
            prod = info.metadata.get("producer") or info.metadata.get("creator") or ""
            print(f"  元数据佐证: {prod[:70]!r}")
    if rot_pages:
        print(f"需转正的页（横向表）: {', '.join(f'p{p}→{r}°' for p, r in rot_pages)}")

    if not caps:
        print("\n未找到任何图表标号。")
    else:
        _print_labels("表 TABLE", [l for l in caps if l.kind == "table"], ocr_pages, width)
        _print_labels("图 FIGURE", [l for l in caps if l.kind == "figure"], ocr_pages, width)

    if refs:
        print()
        print(f"── 只被正文引用、本 PDF 里没有实体（{len(refs)} 个）" + "─" * 40)
        print("   多半在单独的补充材料文件里，需另行下载：")
        for l in sorted(refs, key=lambda x: x.key):
            print(f"     {l.raw:28s} 首次出现 p{l.page}")

    if args.outdir or args.prefix:
        outdir = emit.resolve_outdir(args.outdir, args.prefix, args.pdf)
        outdir.mkdir(parents=True, exist_ok=True)
        path = emit.write_captions(outdir, caps, refs)
        print(f"\n已写出机读清单: {path}")

    print()
    print("提示: --list 不抽内容、不跑版面分析，所以不给行列数。")
    print("      挑好了用 -k 关键词抽（匹配的就是上面这些 caption/legend 文本），或 --all 全导出。")
    return EXIT_OK


# ---------------------------------------------------------------------------
# 候选构建与分流
# ---------------------------------------------------------------------------


def _spans_cache(pdf: Path) -> dict[int, list[tuple[Rect, str]]]:
    cache: dict[int, list[tuple[Rect, str]]] = {}

    class _Lazy(dict):
        def __missing__(self, page_no: int):
            v = pdfio.page_spans(pdf, page_no)
            self[page_no] = v
            return v

    return _Lazy(cache)


def build_candidates(
    norm_pdf: Path,
    info: DocInfo,
    dl: engine_docling.DoclingResult,
    spans,
    pages_filter: set[int] | None,
) -> tuple[list[TableCandidate], list[TableCandidate]]:
    """返回 (文字路径候选, OCR 路径候选)。

    文字候选来自 docling 的 TableItem；
    OCR 候选来自 PyMuPDF 列出的**嵌入位图**（而不是 docling 的 PictureItem）——
    因为只有位图有 xref、能按原生分辨率抠出来（PDF 事实 #4），而矢量绘制的"图"
    其文字本来就在文字层，会被分流判据自动送去文字路径，不需要走 OCR。
    """
    text_cands: list[TableCandidate] = []
    for t in dl.tables:
        if pages_filter and t.page not in pages_filter:
            continue
        n = rules.region_chars_of(t.rect, spans[t.page])
        text_cands.append(
            TableCandidate(
                page=t.page,
                rect=t.rect,
                source_type=rules.dispatch_source_type(n),
                chars_in_rect=n,
                origin="docling_table",
            )
        )

    # 图片路径的候选来源是「**图区域**」，不是「嵌入位图」——
    # 这是前身项目 Channel B 的做法（45 张 figure_ocr 实测有效）。
    # 必须这样的原因：实测 pbc_21078 的 Fig. 3（p10）是**矢量绘制**的热图，
    # 该页一张嵌入位图都没有（矢量指令 105 条），按位图找候选会**静默漏掉整张图**；
    # 而 docling 又把它归成 PictureItem（无网格），文字路径同样看不见它。
    # 图区域取自 docling 的 PictureItem；位图有就抠原生分辨率（PDF 事实 #4），
    # 没有就渲染（矢量无损放大）。
    image_cands: list[TableCandidate] = []
    ocr_layer_pages = {p.page for p in info.pages if p.ocr_text_layer}
    by_page_images: dict[int, list[ImageInfo]] = {p.page: p.images for p in info.pages}

    def _covering_image(page_no: int, r: Rect) -> ImageInfo | None:
        """找出覆盖该区域大部分面积的嵌入位图（有则用原生分辨率，画质更好）。"""
        best = None
        for im in by_page_images.get(page_no, []):
            ox = min(im.rect.x1, r.x1) - max(im.rect.x0, r.x0)
            oy = min(im.rect.y1, r.y1) - max(im.rect.y0, r.y0)
            if ox <= 0 or oy <= 0:
                continue
            frac = (ox * oy) / max(r.width * r.height, 1e-6)
            if frac >= 0.6 and (best is None or im.px_width > best.px_width):
                best = im
        return best

    seen_regions: list[tuple[int, Rect]] = []
    for pic in dl.pictures:
        if pages_filter and pic.page not in pages_filter:
            continue
        if pic.page in ocr_layer_pages:
            # 整页被位图覆盖 ⇒ 文字层是 OCR 派生但**存在**，走文字路径（强制 low）。
            continue
        if rules.is_decorative_image(pic.rect.height / max(info.pages[pic.page - 1].height, 1)):
            continue
        n = rules.region_chars_of(pic.rect, spans[pic.page])
        im = _covering_image(pic.page, pic.rect)
        if im is not None and rules.is_decorative_image(im.cover_h):
            continue
        seen_regions.append((pic.page, pic.rect))
        image_cands.append(
            TableCandidate(
                page=pic.page,
                rect=pic.rect,
                source_type="image",
                chars_in_rect=n,
                origin="docling_picture" if im is None else "raster_image",
                image=im,
            )
        )

    # 兜底：docling 没标成 Picture 但确实存在的大位图（它可能漏标）
    for pinfo in info.pages:
        if pages_filter and pinfo.page not in pages_filter:
            continue
        if pinfo.ocr_text_layer:
            continue
        for im in pinfo.images:
            if rules.is_decorative_image(im.cover_h):
                continue
            if any(
                pg == pinfo.page
                and im.rect.x0 < r.x1 and r.x0 < im.rect.x1
                and im.rect.y0 < r.y1 and r.y0 < im.rect.y1
                for pg, r in seen_regions
            ):
                continue
            n = rules.region_chars_of(im.rect, spans[pinfo.page])
            if rules.dispatch_source_type(n) != "image":
                continue
            image_cands.append(
                TableCandidate(
                    page=pinfo.page, rect=im.rect, source_type="image",
                    chars_in_rect=n, origin="raster_image", image=im,
                )
            )

    return text_cands, image_cands


def attach_labels(cands: list[TableCandidate], caps: list[Label]) -> None:
    """把真 caption 配到候选区域（惯例：表标题在表上方，图标题在图下方）。"""
    used: set[int] = set()
    for lab in sorted(caps, key=lambda l: (l.page, l.rect.y0 if l.rect else 0)):
        pool = [c for i, c in enumerate(cands) if i not in used and c.page == lab.page]
        if not pool:
            continue
        pick = rules.pair_label_to_candidate(lab, pool)
        if pick is None:
            continue
        pick.label = lab
        used.add(cands.index(pick))


# ---------------------------------------------------------------------------
# 两条路径
# ---------------------------------------------------------------------------


def run_text_path(
    norm_pdf: Path,
    cands: list[TableCandidate],
    dl: engine_docling.DoclingResult,
    info: DocInfo,
    all_captions: list[tuple[str, str]],
) -> list[ExtractedTable]:
    by_key = {(t.page, round(t.rect.y0, 1)): t for t in dl.tables}
    ocr_pages = {p.page for p in info.pages if p.ocr_text_layer}
    # 整篇词表，供 rules.rejoin_hyphen_breaks 用。**懒算** —— 要读全文，
    # 而多数文档一处断行都没有（三语料 456+55 张表里只有个位数命中）。
    word_counts: dict[str, int] | None = None
    out: list[ExtractedTable] = []
    for c in cands:
        dt = by_key.get((c.page, round(c.rect.y0, 1)))
        if dt is None or not dt.rows:
            continue

        # ——— F-017 接回断行连字符 ———
        # **必须在造 ExtractedTable 之前**：下面的 caption 归属校验也要用修正后的 rows，
        # 两边不能一份新一份旧。`rejoin_hyphen_breaks` 返回新 list，不会污染 dl.tables。
        rows, n_join = dt.rows, 0
        if rules.has_hyphen_break(rows):
            if word_counts is None:
                word_counts = pdfio.doc_word_counts(norm_pdf)
            rows, n_join = rules.rejoin_hyphen_breaks(rows, word_counts)

        table = ExtractedTable(candidate=c, rows=rows, extractor="docling")
        if n_join:
            table.notes.append(f"hyphen_rejoined={n_join}")
        if c.source_type == "image":
            table.notes.append("region_has_no_text_layer")

        caption = c.label.text if c.label else dt.caption
        # 只拿**同类** caption 当竞争者 —— 图注天然会列举表格的行标识（实测 pbc_21078 的
        # Fig.1 图注列了 EW5/SK-NEP-1/Rh28/KT-13，正是 TABLE I 的首列），混进来必然误报。
        kind = c.label.kind if c.label else "table"
        others = [t for k, t in all_captions if k == kind and t != caption]
        if caption and not rules.verify_caption_belongs(caption, rows, other_captions=others):
            table.grid_status = "grid_mismatch"
            table.notes.append("caption 实词与表头/首列对不上，归属可疑")

        table.plumber_cols = engine_plumber.columns_in_bbox(norm_pdf, c.page, c.rect)
        rules.audit(table, region_chars=c.chars_in_rect, ocr_page=c.page in ocr_pages)
        if c.page in ocr_pages:
            table.notes.append("ocr_text_layer_page")
        out.append(table)
    return out


def run_image_path(
    norm_pdf: Path,
    cands: list[TableCandidate],
    keywords: list[str],
    args: argparse.Namespace,
    workdir: Path,
) -> tuple[list[ExtractedTable], list[ManifestRow]]:
    """三级成本梯度：caption 命中 → 单元格闸门 → 粗 OCR 筛 → 结构还原。"""
    from . import engine_paddle

    hint_words = _read_words(args.hint_words_file) if args.hint_words_file else []
    tables: list[ExtractedTable] = []
    skipped: list[ManifestRow] = []

    for seq, c in enumerate(cands, 1):
        tid = emit.table_id(c.label, c.page, seq)
        png = workdir / f"p{c.page:02d}_img{seq}.png"
        try:
            if c.image is not None:
                pdfio.extract_native_image(norm_pdf, c.image, png, region=c.rect)
            else:
                # 矢量图：没有原生位图可抠，渲染该区域（前身项目 Channel B 的做法）
                pdfio.render_region(norm_pdf, c.page, c.rect, png, dpi=max(args.dpi, 300))
        except Exception as exc:
            skipped.append(
                emit.failure_row(
                    table_id_=tid, label=c.label, page=c.page,
                    reason=f"抠原生图失败: {type(exc).__name__}", source_type="image",
                )
            )
            continue

        # ① caption / legend 命中 → 直接放行
        cap_text = c.label.text if c.label else ""
        cap_hits = rules.keyword_hits(cap_text, keywords, substring=args.substring)
        if args.match_all and keywords and len(cap_hits) != len(keywords):
            cap_hits = []
        matched_on = "caption" if cap_hits else ""

        # 低分辨率图先放大再走后续步骤（裁剪坐标也随之放大，故在闸门之前做）
        eff = c.image.effective_dpi if c.image else 0.0
        factor = engine_paddle.upscale_factor(eff) if eff else 1
        if factor > 1:
            up = workdir / f"p{c.page:02d}_img{seq}_x{factor}.png"
            engine_paddle.upscale(png, up, factor)
            png = up

        # ② 单元格闸门（0.4-0.7s，比全套 OCR 便宜 15-70 倍；判别力 300 vs 0-6）
        grid = engine_paddle.cells_summary(png)
        n_cells, box = grid.n_cells, grid.bbox
        if n_cells < rules.TABLE_CELLS_THRESHOLD:
            skipped.append(
                emit.failure_row(
                    table_id_=tid, label=c.label, page=c.page,
                    reason=f"图内未检出表格网格（单元格 {n_cells} 个 < {rules.TABLE_CELLS_THRESHOLD}），"
                           f"判定为纯图表，跳过 OCR",
                    source_type="image",
                )
            )
            continue

        # ③ caption 未命中时，用粗 OCR 看关键词是否只写在图内
        img_hits: list[str] = []
        if not cap_hits and not args.all and keywords:
            hint_hit = not hint_words or bool(
                rules.keyword_hits(cap_text, hint_words, substring=args.substring)
            )
            if not hint_hit:
                skipped.append(
                    emit.failure_row(
                        table_id_=tid, label=c.label, page=c.page,
                        reason="legend 未命中关键词也未命中 --hint-words-file，跳过粗 OCR",
                        source_type="image",
                    )
                )
                continue
            blob = engine_paddle.plain_text(png)
            img_hits = rules.keyword_hits(blob, keywords, substring=args.substring)
            if args.match_all and len(img_hits) != len(keywords):
                img_hits = []
            if not img_hits:
                skipped.append(
                    emit.failure_row(
                        table_id_=tid, label=c.label, page=c.page,
                        reason="图内含表格但粗 OCR 未命中关键词，跳过", source_type="image",
                    )
                )
                continue
            matched_on = "image_text"

        # 裁到单元格外接框再还原 —— 期刊常把"表+图"拼一张，不裁会被整体判成 chart
        target = png
        if box is not None:
            cropped = workdir / f"p{c.page:02d}_img{seq}_crop.png"
            engine_paddle.crop(png, box, cropped)
            target = cropped

        rows = engine_paddle.recognize_table(target)
        if not rows:
            skipped.append(
                emit.failure_row(
                    table_id_=tid, label=c.label, page=c.page,
                    reason=f"检出 {n_cells} 个单元格但结构还原失败", source_type="image",
                )
            )
            continue

        table = ExtractedTable(candidate=c, rows=rows, extractor="ppstructure")
        table.matched_keywords = cap_hits or img_hits
        table.matched_on = matched_on or "caption"
        table.notes.append(
            f"cells_detected={n_cells}(≈{grid.n_row_bands}行×{grid.n_col_bands}列)"
        )
        # 免费的第二意见：检测器的列带/行带数 vs 还原结果的行列数
        mismatch = rules.grid_shape_mismatch(
            grid.n_row_bands, grid.n_col_bands, len(rows), max(len(r) for r in rows)
        )
        if mismatch:
            table.notes.append(mismatch)
            table.grid_status = "shape_disagree"
        if eff and eff < 200:
            table.notes.append(f"low_res({eff:.0f}dpi,放大{factor}x)")
        rules.audit(table, region_chars=0, ocr_page=True)
        if args.dump_images:
            table.notes.append(f"image={target.name}")
        tables.append(table)
    return tables, skipped


# ---------------------------------------------------------------------------
# 续表拼接
# ---------------------------------------------------------------------------


def stitch(tables: list[ExtractedTable]) -> list[ExtractedTable]:
    """把续表并进前一张。两条判据见 rules.is_continuation（缺一会漏）。"""
    tables = sorted(tables, key=lambda t: (t.candidate.page, t.candidate.rect.y0))
    out: list[ExtractedTable] = []
    for t in tables:
        if out:
            why = rules.is_continuation(out[-1], t)
            if why:
                prev = out[-1]
                body = t.rows[1:] if rules.same_column_names(
                    prev.rows[0] if prev.rows else [], t.rows[0] if t.rows else []
                ) else t.rows
                prev.rows.extend(body)
                prev.notes.append(
                    f"spans_pages={prev.candidate.page}-{t.candidate.page}({why})"
                )
                continue
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def cmd_extract(args: argparse.Namespace, keywords: list[str]) -> int:
    info_orig = pdfio.read_doc(args.pdf)

    # 铁律 #3：版本闸门必须在关键词匹配**之前**生效
    if pdfio.is_preprint(info_orig) and not args.allow_preprint:
        detail = ", ".join(f"{k}×{v}" for k, v in sorted(info_orig.preprint_markers.items()))
        print(
            f"错误: 这看起来是 preprint / peer-review 版（{detail}）。\n"
            f"      这类版本的表格与正式发表版可能不一致。确认要处理请加 --allow-preprint。",
            file=sys.stderr,
        )
        return EXIT_PREPRINT

    outdir = emit.resolve_outdir(args.outdir, args.prefix, args.pdf)
    outdir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="pte_") as tmp:
        tmpdir = Path(tmp)
        norm_pdf = tmpdir / "normalized.pdf"
        rotated = pdfio.write_normalized(args.pdf, norm_pdf, info_orig)
        if rotated:
            print(f"已转正横向表页: {rotated}")

        labels = pdfio.scan_labels(norm_pdf, args.label_pattern)
        caps = pdfio.caption_labels(labels)
        if not caps and not args.all:
            _print_no_labels(info_orig)
            return EXIT_NO_LABELS

        info = pdfio.read_doc(norm_pdf)
        pages_filter = parse_pages(args.pages, info.n_pages)
        spans = _spans_cache(norm_pdf)

        print("正在跑版面分析与表格还原（docling）…")
        dl = engine_docling.convert(norm_pdf)
        text_cands, image_cands = build_candidates(norm_pdf, info, dl, spans, pages_filter)
        # 配对必须按 kind 分开、且 table 优先。
        # 实测 bug：blood p4 上同时有 `Figure 1` 和 `Table 2` 两个 caption，
        # 不分 kind 时图注被配给了 docling 的 TableItem，导致 Table 2 被标成 fig_1。
        # 表格候选优先吃 table 类标号；剩下没配上的才允许吃 figure 类
        # （矢量绘制在"图"里的表确实存在，其文字在文字层，会走文字路径）。
        table_caps = [l for l in caps if l.kind == "table"]
        figure_caps = [l for l in caps if l.kind == "figure"]
        attach_labels(text_cands, table_caps)
        attach_labels([c for c in text_cands if c.label is None], figure_caps)
        attach_labels(image_cands, figure_caps)

        # 文字路径
        text_tables = run_text_path(
            norm_pdf, text_cands, dl, info, [(l.kind, l.text) for l in caps]
        )
        text_tables = stitch(text_tables)

        rows: list[ManifestRow] = []
        kept: list[ExtractedTable] = []
        for t in text_tables:
            if keywords:
                # 文字表：caption+legend **和**表格内容一起匹配 —— 内容此刻已在内存里，
                # 免费。实测 pbc_28772 的金标准表 caption 无 `response`，
                # 但第 11 列表头是 `Obj. Response`，只匹配 caption 会漏。
                hits, where = rules.match_table(
                    t, keywords, match_all=args.match_all, substring=args.substring
                )
                if not hits and not args.all:
                    continue
                t.matched_keywords = hits
                t.matched_on = where
            kept.append(t)

        # OCR 路径
        img_tables: list[ExtractedTable] = []
        img_skipped: list[ManifestRow] = []
        if image_cands and not args.no_ocr:
            print(f"图片候选 {len(image_cands)} 个，过单元格闸门…")
            img_tables, img_skipped = run_image_path(
                norm_pdf, image_cands, keywords, args, tmpdir
            )
        elif image_cands:
            print(f"--no-ocr：跳过 {len(image_cands)} 个图片候选")

        # 落盘
        for seq, t in enumerate(kept + img_tables, 1):
            tid = emit.table_id(t.candidate.label, t.candidate.page, seq)
            name = emit.csv_name(tid, t.candidate.page)
            emit.write_table(outdir, t, name)
            rows.append(emit.to_manifest_row(t, table_id_=tid, csv_path=name))
            if args.dump_images and t.candidate.source_type == "image":
                src = tmpdir / f"p{t.candidate.page:02d}_img{seq}.png"
                if src.exists():
                    (outdir / src.name).write_bytes(src.read_bytes())
        rows.extend(img_skipped)

        # 铁律 #2 的兜底：caption 命中了关键词、但一张表都没产出 → 必须留失败行。
        # 实测教训：pbc_21078 的 Fig. 3 legend 含 `response`，但它是矢量热图 ——
        # docling 归成 Picture（无网格）、该页又没有位图（OCR 路径无从下手），
        # 于是两条路都看不见它，**静默漏掉**。这条兜底让"漏了"至少是可见的。
        exported_keys = {
            (t.candidate.label.key if t.candidate.label else None)
            for t in kept + img_tables
        }
        if keywords:
            for lab in caps:
                if lab.key in exported_keys:
                    continue
                hits = rules.keyword_hits(lab.text, keywords, substring=args.substring)
                if args.match_all and len(hits) != len(keywords):
                    hits = []
                if not hits:
                    continue
                rows.append(
                    emit.failure_row(
                        table_id_=emit.table_id(lab, lab.page, 0),
                        label=lab,
                        page=lab.page,
                        reason=(
                            f"caption/legend 命中关键词 {';'.join(hits)}，"
                            f"但未能在该位置定位或抽出表格 —— 可能它本来就不是表格（纯图表），"
                            f"也可能是矢量绘制/版面识别漏了。请人工看一眼该页"
                        ),
                    )
                )

        # 只被引用、实体不在本 PDF 的标号
        for lab in pdfio.referenced_only(labels):
            hits = rules.cooccurs_in_sentence(lab.text, lab.raw, keywords) if keywords else []
            if hits or args.all:
                rows.append(
                    emit.failure_row(
                        table_id_=emit.table_id(lab, lab.page, 0),
                        label=lab,
                        page=lab.page,
                        reason=(
                            "label_referenced_but_absent: 正文引用了它"
                            + (f"（与关键词 {';'.join(hits)} 同句）" if hits else "")
                            + "，但本 PDF 里没有它的正文，多半在单独的补充材料文件里，请另行下载"
                        ),
                    )
                )

        query = emit.describe_query(keywords, match_all=args.match_all, export_all=args.all)
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for r in rows:
            r.query = query
            r.run_at = stamp
        path = emit.write_manifest(outdir, rows, query=query)
        emit.print_summary(rows, outdir, manifest=path, query=query)
        if not (kept or img_tables) and keywords:
            emit.print_zero_hit_hint(caps, query)
    return EXIT_OK


def _print_no_labels(info: DocInfo) -> None:
    ocr_hint = "（本文档已检出 OCR 文字层页）" if any(p.ocr_text_layer for p in info.pages) else ""
    print(
        f"错误: 全文扫不到任何图表标号。可能原因：\n"
        f"      1) 这份文档确实没有表格；\n"
        f"      2) 它的标号惯例不被默认正则识别 —— 用 --label-pattern 指定；\n"
        f"      3) 文字层是 OCR 派生的、标号被识别错了{ocr_hint}。\n"
        f"      逃生舱：加 --all 跳过标号匹配，直接让版面分析去找表格。",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.pdf.exists():
        print(f"错误: 找不到文件 {args.pdf}", file=sys.stderr)
        return EXIT_USAGE

    if args.list:
        return cmd_list(args)

    keywords = load_keywords(args)
    if not keywords and not args.all:
        print(
            "错误: 需要 -k/--keyword，或用 --all 导出全部，或用 --list 先看有哪些表。",
            file=sys.stderr,
        )
        return EXIT_USAGE

    return cmd_extract(args, keywords)


if __name__ == "__main__":
    raise SystemExit(main())
