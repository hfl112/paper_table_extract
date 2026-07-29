"""PDF 物理层访问。**PyMuPDF 的唯一入口** —— fitz 的 API 不许漏到本文件外。

职责：读文字层、扫标号、页面结构统计、生成归一化 PDF、区域内字符计数、
抠原生分辨率位图、preprint 检测。
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz

from . import rules
from .models import DocInfo, ImageInfo, Label, PageInfo, Rect, Word

# preprint / peer-review 版检测。实测 566455 每页页脚有 bioRxiv 水印：
# bioRxiv×126 / preprint×189 / peer review×63。字符串级检测即可，比几何判定可靠得多。
PREPRINT_MARKERS = ("bioRxiv", "medRxiv", "preprint", "peer review", "peer-review")
# 判 preprint 要看**出现在多少比例的页上**，不是总次数 —— 真 preprint 的特征是
# 页脚水印盖**每一页**。这是留出集抓出来的误报：
#   566455（真 preprint）  54/64 页 = 84%（总次数 378）
#   Baslan 2022（正式版）   2/33 页 =  6%（总次数正好 5，撞上旧的总次数阈值）
#     —— 那 5 次全在正式发表版的标准栏目和参考文献里：
#        `peer review information; details of author contributions`（Nature 标准段落）
#        `Preprint at bioRxiv https://doi.org/10.1101/...`（引用了别人的 preprint）
#        `Peer review information Nature thanks the anonymous reviewers...`
# 阈值 0.4：84% 与 6% 两侧各有 2 倍以上余量。
PREPRINT_MIN_PAGE_RATIO = 0.4

# arXiv 型预印本：**只在第 1 页左边缘盖一条竖排水印**，剩下出现 arXiv 字样的页全是参考文献。
# 所以「含水印词的页数占比」这条对它结构上就不成立 —— 实测 4 篇 arXiv：
#     Perrone 2020（有 `A PREPRINT` 页眉模板）90% → 现有判据抓到
#     Li_2013 (BWA-MEM) 33% / McInnes_2020 (UMAP) 10% / 2607.00042v1 (天文) 0% → **全漏**
# 那篇天文论文 8/27 页含 `arXiv`，其中 **1 页是真水印、7 页是参考文献里的引用** ——
# 噪声是信号的 7 倍，把 `arXiv` 加进 PREPRINT_MARKERS 只会更糟（log.md §8 的 Baslan 坑）。
#
# 换成结构判据就干净：第 1 页最左 8% 内、竖排、含「平台名 + 编号」。
# 实测 90 篇（开发集 12 + 留出集 25 + 阈值验证集 53）**零误伤**，4 篇 arXiv 全中。
# 这是**与页数占比并列的第二条**，不是替换 —— 占比管「每页盖章」型（bioRxiv/medRxiv）。
P1_WATERMARK_MAX_X_RATIO = 0.08
P1_WATERMARK_RE = re.compile(
    r"(arXiv|bioRxiv|medRxiv|Research\s*Square|SSRN|ChemRxiv)\s*:?\s*\d{4}\.\d{4,5}", re.I
)


_VERT_DIRS = {(0, 1), (0, -1)}


def _p1_watermark(page) -> str:
    """第 1 页左边缘的竖排预印本水印，取不到返回空串。"""
    limit = page.rect.width * P1_WATERMARK_MAX_X_RATIO
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            if tuple(round(x) for x in line["dir"]) not in _VERT_DIRS:
                continue
            if line["bbox"][2] > limit:
                continue
            text = "".join(s["text"] for s in line["spans"]).strip()
            if P1_WATERMARK_RE.search(text):
                return text
    return ""


def _rect(bbox) -> Rect:
    return Rect(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


# ---------------------------------------------------------------------------
# 页面结构
# ---------------------------------------------------------------------------


def _line_dirs(page: fitz.Page, pat: re.Pattern[str] | None = None) -> tuple[int, int, tuple[int, int], bool]:
    """返回 (水平行数, 竖排行数, 竖排的主方向, 竖排里有没有 Table 标号)。

    最后那个是转正判据的第三条 —— 横向表的 caption 跟着表一起转过来了，
    而多面板图的轴标签、逐字符拆开的版权水印都没有标号。见 rules.rotation_for_page。
    """
    pat = pat or rules.compile_label_pattern()
    horiz = vert = 0
    dir_count: dict[tuple[int, int], int] = {}
    has_label = False
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            d = tuple(round(x) for x in line["dir"])
            if d == (1, 0):
                horiz += 1
            elif d in _VERT_DIRS:
                vert += 1
                dir_count[d] = dir_count.get(d, 0) + 1
                if not has_label:
                    text = "".join(s["text"] for s in line["spans"])
                    if rules.match_label_at_line_start(text, pat):
                        has_label = True
    dominant = max(dir_count, key=lambda k: dir_count[k]) if dir_count else (0, -1)
    return horiz, vert, dominant, has_label


def _images(page: fitz.Page) -> list[ImageInfo]:
    out = []
    pw, ph = page.rect.width, page.rect.height
    for info in page.get_image_info(xrefs=True):
        r = _rect(info["bbox"])
        out.append(
            ImageInfo(
                xref=int(info.get("xref") or 0),
                rect=r,
                px_width=int(info["width"]),
                px_height=int(info["height"]),
                cover_w=(r.width / pw) if pw else 0.0,
                cover_h=(r.height / ph) if ph else 0.0,
            )
        )
    return out


def read_doc(path: str | Path) -> DocInfo:
    """一次读完整篇的结构特征。不做任何 ML。"""
    path = str(path)
    doc = fitz.open(path)
    pages: list[PageInfo] = []
    marker_counts: dict[str, int] = {}
    preprint_pages = 0
    p1_watermark = ""
    label_pat = rules.compile_label_pattern()
    prev_rotated = False
    try:
        for pno, page in enumerate(doc, 1):
            text = page.get_text()
            page_hit = False
            for m in PREPRINT_MARKERS:
                n = len(re.findall(re.escape(m), text, re.I))
                if n:
                    marker_counts[m] = marker_counts.get(m, 0) + n
                    page_hit = True
            if page_hit:
                preprint_pages += 1
            if pno == 1:
                p1_watermark = _p1_watermark(page)
            horiz, vert, dominant, has_label = _line_dirs(page, label_pat)
            imgs = _images(page)
            try:
                n_vec = len(page.get_drawings())
            except Exception:
                n_vec = 0
            pages.append(
                PageInfo(
                    page=pno,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    n_chars=len(text.strip()),
                    vert_lines=vert,
                    horiz_lines=horiz,
                    n_vector_ops=n_vec,
                    images=imgs,
                    ocr_text_layer=False,  # 第二趟按整篇判（见下）
                    needs_rotation=rules.rotation_for_page(
                        vert, dominant, horiz,
                        has_table_label=has_label, prev_page_rotated=prev_rotated,
                    ),
                )
            )
            prev_rotated = bool(pages[-1].needs_rotation)

        # 第二趟：「是不是扫描件」是**整篇**的属性，不是单页的 —— 见 rules.is_scanned_document。
        # 必须等所有页都读完才知道「整页位图页占多大比例」。
        scanned = rules.is_scanned_document(
            n_full_page_images=sum(1 for p in pages if rules.has_full_page_image(p.images)),
            n_pages=len(pages),
            total_chars=sum(p.n_chars for p in pages),
        )
        for p in pages:
            p.ocr_text_layer = rules.is_ocr_text_layer(p.images, doc_is_scanned=scanned)

        meta = {k: (v or "") for k, v in (doc.metadata or {}).items()}
    finally:
        doc.close()
    return DocInfo(
        path=path,
        n_pages=len(pages),
        pages=pages,
        metadata=meta,
        preprint_markers=marker_counts,
        preprint_pages=preprint_pages,
        p1_watermark=p1_watermark,
    )


def is_preprint(info: DocInfo) -> bool:
    """铁律 #3：preprint / peer-review 版默认拒跑（--allow-preprint 覆盖）。

    **两条并列判据**（见 PREPRINT_MIN_PAGE_RATIO 处的实测依据）：
      1. 含水印词的**页数占比** >= 40% —— 管「每页盖章」型（bioRxiv / medRxiv、
         以及用 `A PREPRINT` 页眉模板的 arXiv）
      2. 第 1 页左缘有**竖排的「平台名 + 编号」水印** —— 管 arXiv 那种「只盖首页」型，
         对它占比判据结构上不成立（实测 4 篇里占比只抓到 1 篇）
    """
    if not info.n_pages:
        return False
    if info.p1_watermark:
        return True
    return info.preprint_pages / info.n_pages >= PREPRINT_MIN_PAGE_RATIO


# ---------------------------------------------------------------------------
# 标号扫描（rules 提供判据，这里只负责取文字）
# ---------------------------------------------------------------------------


# legend 的安全阀。正常 legend 实测最长 1186 字符（blood Figure 1，23 行），
# 留足余量；超过它多半是把正文吞进来了。
LEGEND_MAX_CHARS = 3000


def scan_labels(path: str | Path, pattern: str | None = None) -> list[Label]:
    """逐页逐行扫标号。行首命中 = 真 caption；其余出现 = 正文引用。

    caption+legend 的取法：从命中行起，取到**同 block 内下一个 caption 行**为止
    （没有下一个就到 block 结束）。

    两个都是踩出来的细节：
      - 不能设固定行数/字符上限：实测 15 个 caption 超过 600 字符、7 个超过 8 行，
        blood 的 Figure 1 有 23 行 / 1186 字符，硬截会把 legend 切断。
      - 不能简单"取到 block 结束"：PyMuPDF 会把相邻两个 caption 并进一个 block
        （实测 blood p6 的 `Figure 4.` 在 line0、`Table 3.` 在 line12），
        那样会把后一个 caption 连同它的 legend 一起吞进前一个。
    """
    pat = rules.compile_label_pattern(pattern)
    loose = rules.compile_loose_pattern(pattern)
    doc = fitz.open(str(path))
    labels: list[Label] = []
    try:
        for pno, page in enumerate(doc, 1):
            raw_blocks = [b for b in page.get_text("dict")["blocks"] if "lines" in b]
            # 按 y 排序，供"续抓下一个 block"用（原始顺序不保证按版面从上到下）
            ordered = sorted(raw_blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
            for block in raw_blocks:
                lines = block["lines"]
                texts = [
                    rules.collapse_ws("".join(s["text"] for s in ln["spans"])) for ln in lines
                ]
                # 先把本 block 里所有 caption 行找出来，才能知道每段 legend 到哪结束
                cap_at: dict[int, str] = {}
                for i, line_text in enumerate(texts):
                    raw = rules.match_label_at_line_start(line_text, pat)
                    if raw is None:
                        continue
                    # 二次确认：排除"引用恰好被折行折到行首"的假 caption
                    if not rules.caption_line_is_plausible(texts[i - 1] if i else None):
                        continue
                    cap_at[i] = raw
                caption_idx = set(cap_at)
                starts = sorted(cap_at)
                for n, i in enumerate(starts):
                    raw = cap_at[i]
                    end = starts[n + 1] if n + 1 < len(starts) else len(texts)
                    key = rules.normalize_label(raw)
                    body = rules.collapse_ws(" ".join(texts[i:end]))
                    # legend 常常跨 block —— 只有当本 caption 是 block 里最后一个
                    # caption 时，才去续抓紧跟其后的 prose block（见 rules 里的实测依据）
                    if end >= len(texts):
                        body = _absorb_legend_tail(block, ordered, body, pat)
                    body = body[:LEGEND_MAX_CHARS]
                    labels.append(
                        Label(
                            raw=rules.collapse_ws(raw),
                            key=key,
                            kind=rules.label_kind(key),
                            page=pno,
                            is_caption=True,
                            text=body,
                            rect=_rect(lines[i]["bbox"]),
                        )
                    )
                # 正文引用：整个 block 里所有标号，减去本 block 里已判为 caption 的那些
                block_text = rules.collapse_ws(" ".join(texts))
                cap_keys = {
                    rules.normalize_label(r)
                    for i in caption_idx
                    for r in [rules.match_label_at_line_start(texts[i], pat)]
                    if r
                }
                for raw in rules.find_labels_anywhere(block_text, loose):
                    key = rules.normalize_label(raw)
                    if key in cap_keys:
                        continue
                    if any(l.page == pno and l.key == key and l.is_caption for l in labels):
                        continue
                    labels.append(
                        Label(
                            raw=rules.collapse_ws(raw),
                            key=key,
                            kind=rules.label_kind(key),
                            page=pno,
                            is_caption=False,
                            text=block_text,
                            rect=None,
                        )
                    )
    finally:
        doc.close()
    return labels


def _absorb_legend_tail(block, ordered, body: str, pat) -> str:
    """把紧跟 caption block 之后的 legend 续段并进来。

    判据在 rules.is_legend_continuation()（间距 + 最长行长度，两者都必需）。
    遇到含 caption 行的 block 就停 —— 那是下一张图表的标题。
    """
    bottom = block["bbox"][3]
    bx0, bx1 = block["bbox"][0], block["bbox"][2]
    try:
        idx = ordered.index(block)
    except ValueError:
        return body
    for nxt in ordered[idx + 1 :]:
        nx0, ny0, nx1, ny1 = nxt["bbox"]
        if ny0 < bottom - 1:  # 不在下方
            continue
        if min(bx1, nx1) - max(bx0, nx0) <= 0:
            # 横向不重叠 —— **跳过而不是终止**。
            # 实测踩坑：pbc_30017 是 PMC Author Manuscript 版，左边距有一条
            # 竖排 `Author Manuscript` 水印 block（y=108–712，跨整页），
            # 在 y 序上恰好插在 caption(y0=78.7) 与 legend 续段(y0=111.4) 之间。
            # 遇到它就 break 会导致续段永远抓不到。
            continue
        nlines = [
            rules.collapse_ws("".join(sp["text"] for sp in ln["spans"])) for ln in nxt["lines"]
        ]
        if any(rules.match_label_at_line_start(t, pat) for t in nlines):
            break  # 下一张图表的标题
        if not rules.is_legend_continuation(ny0 - bottom, max((len(t) for t in nlines), default=0)):
            break
        body = rules.collapse_ws(body + " " + " ".join(nlines))
        bottom = ny1
    return body


def caption_labels(labels: list[Label]) -> list[Label]:
    return [l for l in labels if l.is_caption]


def referenced_only(labels: list[Label]) -> list[Label]:
    """只被引用、没有真 caption 的标号 ⇒ 实体不在本 PDF 里（PDF 事实 #2b）。

    实测 CCR-18-2728 引用了 8 个 supp 标号但真 caption 只有 Table 1 —— supplementary 是单独文件。
    """
    cap_keys = {l.key for l in labels if l.is_caption}
    seen: set[str] = set()
    out = []
    for l in labels:
        if l.is_caption or l.key in cap_keys or l.key in seen:
            continue
        seen.add(l.key)
        out.append(l)
    return out


# ---------------------------------------------------------------------------
# 区域内字符计数（dispatcher 的判据）
# ---------------------------------------------------------------------------


def page_spans(path: str | Path, page_no: int) -> list[tuple[Rect, str]]:
    """取某页所有文字 span 的 (矩形, 文本)，供 rules.region_chars_of 使用。"""
    doc = fitz.open(str(path))
    try:
        page = doc[page_no - 1]
        out = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    out.append((_rect(span["bbox"]), span["text"]))
        return out
    finally:
        doc.close()


def page_words(path: str | Path, page_no: int) -> list[Word]:
    """取某页所有词的 (矩形, 文本)。1-based 页码。给 `rules.detect_merged_rows` 用。

    ═══ 必须做旋转变换，这是最容易踩的坑（实测）═══

    在归一化 PDF 上（`page.set_rotation(90)` 之后）实测：`page.rotation` 报 90、
    `page.rect` 确实换成了 792x594，**但 `get_text` 返回的坐标一个字都没变** ——
    `pbc_21296` p4 的首词 `Pediatr` 在原始与归一化 PDF 上 bbox 完全相同。
    也就是说**词坐标停留在未旋转帧**。

    而 **docling 用的是旋转后的帧**：它报 p4 尺寸 792x594、表 bbox 的 x 一直到 730.6
    —— 超过了未旋转的页宽 594。

    两者不在同一坐标系。不做这个变换就拿 docling 的 cell bbox 去框词，
    **一个词都框不到**，压行检测在所有横向表上静默失效。
    变换之后横向表的竖排文字变成正常水平文字，下游不必分方向做分支。
    """
    doc = fitz.open(str(path))
    try:
        page = doc[page_no - 1]
        mat = page.rotation_matrix if page.rotation else None
        out: list[Word] = []
        for x0, y0, x1, y1, text, *_ in page.get_text("words"):
            r = fitz.Rect(x0, y0, x1, y1)
            if mat is not None:
                r = r * mat
            out.append(Word(r.x0, r.y0, r.x1, r.y1, text))
        return out
    finally:
        doc.close()


_WORDISH = re.compile(r"[a-z]+")


def doc_word_counts(path: str | Path) -> dict[str, int]:
    """整篇文字层里每个「纯字母词」出现了几次（小写）。供 rules.rejoin_hyphen_breaks 判据用。

    ═══ 三条都是实测逼出来的，改之前先看 ═══

    1. **作用域取整篇，不是单页。** 断行的词前后半可能落在不同页（跨页表），
       而判据要问的是"这个拼接结果是不是一个真词"，不是"这一页有没有它"。
       注意这**不违反** F-004 那条「文字层比对必须按页」—— 那条针对的是**逐格验证网格**，
       用全文档作用域会让 OCR 垃圾"验证通过"；这里只做词表查询，性质不同。

    2. **不许先把连字符接掉。** 第一版建词表时做了 `re.sub(r"-\\s+", "")`，
       结果 `lineagedefining` 凭空进了词表 → **判据自己证明自己**，
       `lineage- defining` 照样被误接。按 `[a-z]+` 切词即可：
       文字层里 `lineage-defining` 切出 `lineage` + `defining`，**不会**有 `lineagedefining`。

    3. **返回计数而不是集合。** 只要求"出现过 >=1 次"太松 —— 实测
       `de_Bruijne_2021` 里 `channelaware` 恰好出现 **1** 次（本身就是别处断行的产物），
       于是合法的 `Channel-aware` 被误接。见 rules.MIN_WORD_COUNT。
    """
    doc = fitz.open(str(path))
    try:
        text = " ".join(doc[i].get_text() for i in range(len(doc)))
    finally:
        doc.close()
    counts: dict[str, int] = {}
    for w in _WORDISH.findall(text.lower()):
        counts[w] = counts.get(w, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# 归一化 PDF（把横向表的页转正）
# ---------------------------------------------------------------------------


def write_normalized(src: str | Path, dst: str | Path, info: DocInfo) -> list[int]:
    """整篇复制，只把 needs_rotation != 0 的页旋转。返回被转的页码。

    实测（必验 #1）：整篇归一化后喂 docling 一次，结果与逐页转正**逐字一致**
    （pbc_29304 p3=17x15/p4=24x15、pbc_26870 p18=26x8/p19=18x8、pbc_21296 p4=43x10）。
    所以不必逐页调用 docling —— 那样 19 页的文章要跑 19 次，慢十几倍。
    """
    doc = fitz.open(str(src))
    rotated: list[int] = []
    try:
        for p in info.pages:
            if p.needs_rotation:
                doc[p.page - 1].set_rotation(p.needs_rotation)
                rotated.append(p.page)
        doc.save(str(dst))
    finally:
        doc.close()
    return rotated


# ---------------------------------------------------------------------------
# 抠原生分辨率位图
# ---------------------------------------------------------------------------


def extract_native_image(
    path: str | Path, image: ImageInfo, dst: str | Path, region: Rect | None = None
) -> tuple[int, int]:
    """按**原生分辨率**导出嵌入位图（AGENTS.md PDF 事实 #4）。

    不要渲染整页 —— 实测 pbc_24724 p7 的图原生 2004x2056、显示 481pt 宽（有效 300dpi），
    按 200dpi 渲染整页只得 1336px，白丢 33% 线性分辨率。

    `region` 给出时按该子区域裁剪（页面 pt 坐标 → 像素坐标）。
    **必须支持这个**：实测 pbc_21296 p3 的一张位图 `[127,77,463,301]` 被 docling 拆成了
    两个 PictureItem —— `[126,75,298,302]`（左半，就是表格）与 `[311,85,463,269]`（右半，条形图）。
    docling 已经替我们把表格区域分出来了；忽略子区域会让两个候选都拿到整张图，
    结果抽出两份重复的表，而且表格那半还得靠密集带启发式再找一遍。
    """
    doc = fitz.open(str(path))
    try:
        pix = fitz.Pixmap(doc, image.xref)
        if pix.n > 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        if region is not None and image.rect.width > 0 and image.rect.height > 0:
            sx = pix.width / image.rect.width
            sy = pix.height / image.rect.height
            clip = fitz.IRect(
                max(0, int((region.x0 - image.rect.x0) * sx)),
                max(0, int((region.y0 - image.rect.y0) * sy)),
                min(pix.width, int((region.x1 - image.rect.x0) * sx)),
                min(pix.height, int((region.y1 - image.rect.y0) * sy)),
            )
            if clip.width > 20 and clip.height > 20:
                pix = fitz.Pixmap(pix, pix.width, pix.height, clip)
        pix.save(str(dst))
        return pix.width, pix.height
    finally:
        doc.close()


# 矢量图没有"原生分辨率"这回事，渲染 dpi 由我们定。300 足够 OCR，且矢量放大无损。
VECTOR_RENDER_DPI = 300


def render_region(path: str | Path, page_no: int, rect: Rect, dst: str | Path,
                  dpi: int = VECTOR_RENDER_DPI) -> tuple[int, int]:
    """把某页的一块区域渲染成 PNG。**矢量绘制的图表唯一的取图手段。**

    为什么需要它（前身项目 Channel B 的做法，实测有效：45 张 figure_ocr）：
    图片路径的候选来源必须是「图区域」而不是「嵌入位图」——
    实测 pbc_21078 的 Fig. 3（p10）是矢量热图，该页**一张嵌入位图都没有**
    （矢量指令 105 条），若按位图找候选就会静默漏掉整张图。
    """
    doc = fitz.open(str(path))
    try:
        page = doc[page_no - 1]
        clip = fitz.Rect(rect.x0, rect.y0, rect.x1, rect.y1) & page.rect
        pix = page.get_pixmap(clip=clip, dpi=dpi)
        pix.save(str(dst))
        return pix.width, pix.height
    finally:
        doc.close()
