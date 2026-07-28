"""坐标判据的原语层 —— 用 PDF 里每个词的位置验证表格结构对不对。

═══ 为什么需要它 ═══

`eval/checks.py` 的文字层判据只问「格子里的字串在该页文字层里存在吗」。它能证明
**内容是真的**，证明不了**位置是对的** —— 开发集 12 张文字表全是 100% 命中、零告警，
判据已经饱和。而 F-001（`pbc_21296` p4）把两个数据行压进一格，字符一个都没丢，
所以覆盖率、文字层命中、列数第二意见三道防线全过，最终标 `confidence=high`、`notes` 空。

**所有内容类判据对位置错误天然免疫。** 这不是阈值调得不好，是维度不对。
本模块补上缺的那个维度：词在纸上的坐标。

═══ 实测依据（planning 期间量的，不要重新验证）═══

- `pbc_21296` p4（横向表，行沿 x 轴）按 x 聚类得**恰好 44 带** = 人核的 44 数据行，
  而 docling 只给 43。`ALL-7`/`ALL-8`/`ALL-16`/`ALL-17` 的 x 区间等宽等距互不重叠；
  被压进一格的 `8.1 0.2` 两数 x 为 468.5 / 477.9，**正好落在 ALL-8 与 ALL-16 各自的带**。
- `pbc_28772` p3（11 列）按 y 聚类：满带 **11-20 词**，稀带只有 **1、2、6、7 词**；
  `Cyclophosphamide` 的折行续段 `phamide` 落在 1-2 档。满带过滤后 41 带 − caption
  − 2 行表头 = **38 数据行**，与 `AGENTS.md` 记的 p3 38 行一致。
  → **判据要看「占了多少列」，不是原始词数。**
- 全页聚类**不可行**：`blood` p6 只有 6 个数据行、全页却有 61 个满带（整页是正文）。
  **必须先用表的 bbox 过滤。**

═══ 架构例外（进产品那轮要还的债）═══

本文件直接 `import fitz`。`AGENTS.md` 规定 fitz 只能从 `pdfio.py` 出去，而 `pdfio.py`
是产品代码、本轮不动（本轮只出证据）。
**判据进产品那一轮，`page_words` 搬进 `pdfio.py`、紧挨 `page_spans`（L366）。**
不写这句，后人会以为架构规则作废了。
"""

from __future__ import annotations

import statistics
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import fitz  # 架构例外，见模块 docstring

from pdf_table_extract import pdfio

# 竖排方向，与 pdfio._VERT_DIRS 同口径（round 后比较）。
# 不能用 `dir != (1,0)` 当竖排判据 —— findings.md F-002 记过这个坑：45° 的小提琴图
# 刻度 round 后是 (1,-1)，那样会被算成竖排，报出一个假误报。
_VERT_DIRS = {(0, 1), (0, -1)}
_HORIZ_DIR = (1, 0)


@dataclass(frozen=True)
class Word:
    """一个词 + 它的矩形 + 它所属文字行的方向。"""

    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    dir: tuple[int, int]

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    def lo_hi(self, axis: str) -> tuple[float, float]:
        """沿行轴的区间。axis='y' 表示行沿 y 排（普通表），'x' 表示沿 x 排（横向表）。"""
        return (self.y0, self.y1) if axis == "y" else (self.x0, self.x1)

    def cross_lo_hi(self, axis: str) -> tuple[float, float]:
        """垂直于行轴的区间 —— 也就是「列」方向。"""
        return (self.x0, self.x1) if axis == "y" else (self.y0, self.y1)


def page_words(path: str | Path, page_no: int, apply_rotation: bool = True) -> list[Word]:
    """取某页所有词的 (矩形, 文本, 所属行方向)。1-based 页码。

    `get_text("words")` 给坐标但不给方向；`get_text("dict")` 给方向但只到 span 粒度。
    所以两边都取，按**几何包含**把词配到行上 —— 不靠 words 的 block_no/line_no 索引，
    因为 dict 的 blocks 里混着图片块，两套编号未必对得齐。

    ═══ `apply_rotation` 为什么必须默认开（实测，这是最容易踩的坑）═══

    在归一化 PDF 上（`page.set_rotation(90)` 之后）实测：`page.rotation` 报 90、
    `page.rect` 确实换成了 792x594，**但 `get_text` 返回的坐标一个字都没变** ——
    `pbc_21296` p4 的首词 `Pediatr` 在原始与归一化 PDF 上 bbox 完全相同，
    445 个竖排行也照旧是竖排。也就是说**词坐标停留在未旋转帧**。

    而 **docling 用的是旋转后的帧**：它报 p4 尺寸 792x594，表 bbox 的 x 一直到 730.6
    —— 超过了未旋转的页宽 594。

    两者不在同一坐标系。不做这个变换就拿 docling 的 bbox 去框词，**一个词都框不到**。
    变换之后横向表的竖排文字变成正常水平文字，行轴统一是 y，下游逻辑不必分支。
    """
    doc = fitz.open(str(path))
    try:
        page = doc[page_no - 1]
        mat = page.rotation_matrix if (apply_rotation and page.rotation) else None

        lines: list[tuple[fitz.Rect, tuple[int, int]]] = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                d = tuple(round(c) for c in line["dir"])
                if mat is not None:
                    dx, dy = line["dir"]
                    # 只用矩阵的线性部分转方向向量，不带平移
                    d = (round(dx * mat.a + dy * mat.c), round(dx * mat.b + dy * mat.d))
                lines.append((fitz.Rect(line["bbox"]) * mat if mat else fitz.Rect(line["bbox"]), d))

        out: list[Word] = []
        for x0, y0, x1, y1, text, *_ in page.get_text("words"):
            r = fitz.Rect(x0, y0, x1, y1)
            if mat is not None:
                r = r * mat
            cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
            d = _HORIZ_DIR
            for rect, ld in lines:
                # 稍微放宽 —— 词的中心偶尔压在行框边界上
                if rect.x0 - 0.5 <= cx <= rect.x1 + 0.5 and rect.y0 - 0.5 <= cy <= rect.y1 + 0.5:
                    d = ld
                    break
            out.append(Word(r.x0, r.y0, r.x1, r.y1, text, d))
        return out
    finally:
        doc.close()


@contextmanager
def normalized_pdf(orig: str | Path) -> Iterator[tuple[Path, pdfio.DocInfo]]:
    """重建流水线用的那份归一化 PDF（横向表的页已转正）。

    走的是 `__main__.py` L519-521 **同两个函数**（`pdfio.read_doc` + `pdfio.write_normalized`），
    所以坐标帧与 docling 拿到的可证同构，不是"近似复现"。

    为什么必须重建：流水线跑在 `tempfile.TemporaryDirectory` 里的 `normalized.pdf` 上，
    **用完即毁，没有 `--keep-work`**。而 `eval/checks.py:224 page_text_of` 用的是原始 PDF
    —— 那对纯文本判据无害（旋转不改字符），坐标判据一旦沿用就会全盘错位。
    """
    orig = Path(orig)
    info = pdfio.read_doc(orig)
    with tempfile.TemporaryDirectory(prefix="pte_geom_") as tmp:
        dst = Path(tmp) / "normalized.pdf"
        pdfio.write_normalized(orig, dst, info)
        yield dst, info


def row_axis_of(words: list[Word], min_share: float = 0.8) -> tuple[str | None, float]:
    """行沿哪个轴排？返回 ('y'|'x'|None, 众数占比)。

    - 众数 ≈ (1,0)  → 行沿 **y**（普通表，或已正确转正的横向表）
    - 众数 ∈ 竖排   → 行沿 **x**（该页本该转正却没转 —— 这本身就是一个发现）
    - 占比 < min_share → 返回 None，**不出数**。方向混杂的表这套判据不适用，
      给个错数字比不给数字更坏。
    """
    if not words:
        return None, 0.0
    horiz = sum(1 for w in words if w.dir == _HORIZ_DIR)
    vert = sum(1 for w in words if w.dir in _VERT_DIRS)
    total = len(words)
    if horiz >= vert:
        share = horiz / total
        return ("y" if share >= min_share else None), share
    share = vert / total
    return ("x" if share >= min_share else None), share


def words_in_rect(words: list[Word], rect, pad: float = -1.0) -> list[Word]:
    """只留中心落在矩形内的词。`pad` 为负表示内缩（默认内缩 1pt，挡压边的页眉）。

    这是三道污染过滤的第一道。实测污染源：`pbc_21296` p4 的期刊页眉竖排词
    （`Blood`/`Cancer`/`DOI`/`10.1002/pbc`）会落进已有的行带里 —— 它**不新增带**，
    所以对"数带"无害，但会污染占用矩阵。
    """
    x0, y0 = rect.x0 - pad, rect.y0 - pad
    x1, y1 = rect.x1 + pad, rect.y1 + pad
    return [w for w in words if x0 <= w.cx <= x1 and y0 <= w.cy <= y1]


def keep_axis_dir(words: list[Word], axis: str) -> list[Word]:
    """第二道污染过滤：只留方向与表一致的词。

    专治期刊页眉/侧边水印 —— 它们在表的坐标帧里跟表体正交。
    """
    want = _HORIZ_DIR if axis == "y" else None
    if want is not None:
        return [w for w in words if w.dir == want]
    return [w for w in words if w.dir in _VERT_DIRS]


# ————————————————————— 行带聚类：两个候选，用数据选 —————————————————————


def bands_by_center(words: list[Word], axis: str, gap: float) -> list[list[Word]]:
    """M1：按沿行轴的**中心**排序，相邻中心差 > gap 就切带。

    用户已在 `pbc_21296` p4 上验证可行（gap=4.0pt）。
    **已知风险**：上下标（脚注标记 `Median final RTV d`、`> EP g`）的中心偏离基线，
    可能切出伪带。开发集 `pbc_21078` 那三张表全带脚注上标，最容易在那儿翻车。
    """
    if not words:
        return []
    ordered = sorted(words, key=lambda w: (w.cy if axis == "y" else w.cx))
    out: list[list[Word]] = []
    cur = [ordered[0]]
    prev = ordered[0].cy if axis == "y" else ordered[0].cx
    for w in ordered[1:]:
        c = w.cy if axis == "y" else w.cx
        if c - prev > gap:
            out.append(cur)
            cur = []
        cur.append(w)
        prev = c
    out.append(cur)
    return out


def bands_by_overlap(words: list[Word], axis: str, gap: float) -> list[list[Word]]:
    """M2：按沿行轴的**区间**合并 —— 重叠或间距 < gap 的词归一带。

    对上下标天然免疫（上标的区间被正文区间包住），代价是行距很紧的表可能过度合并。
    """
    if not words:
        return []
    ordered = sorted(words, key=lambda w: w.lo_hi(axis)[0])
    out: list[list[Word]] = []
    cur = [ordered[0]]
    cur_hi = ordered[0].lo_hi(axis)[1]
    for w in ordered[1:]:
        lo, hi = w.lo_hi(axis)
        if lo - cur_hi > gap:
            out.append(cur)
            cur = []
            cur_hi = hi
        else:
            cur_hi = max(cur_hi, hi)
        cur.append(w)
    out.append(cur)
    return out


def gap_stats(words: list[Word], axis: str, bands: list[list[Word]]) -> dict:
    """阈值余量普查用：带内最大 gap vs 带间最小 gap。

    **不要拿单张表定阈值。** 期望是双峰分布（带内 ≈0-1pt / 带间 ≈8-11pt）；
    若某张表两侧交叠，说明固定阈值在它身上不成立，该表要标「gap 分不开」而不是给个错数。
    """
    within: list[float] = []
    for b in bands:
        cs = sorted((w.cy if axis == "y" else w.cx) for w in b)
        within += [b1 - b0 for b0, b1 in zip(cs, cs[1:])]
    between: list[float] = []
    for a, b in zip(bands, bands[1:]):
        a_hi = max((w.cy if axis == "y" else w.cx) for w in a)
        b_lo = min((w.cy if axis == "y" else w.cx) for w in b)
        between.append(b_lo - a_hi)
    return {
        "n_bands": len(bands),
        "within_max": max(within) if within else 0.0,
        "between_min": min(between) if between else float("inf"),
        "between_median": statistics.median(between) if between else 0.0,
    }


# ————————————————————— 列边界：投影剖面 —————————————————————


def column_spans(
    words: list[Word], axis: str, min_gap: float, step: float = 0.5
) -> list[tuple[float, float]]:
    """在垂直于行轴的方向上做墨迹投影，连续为空且宽 > min_gap 的区间就是列分隔符。

    为什么用投影而不是逐行数词间距：聚合 40+ 行之后噪声被压掉，比单行稳得多。
    **调用方要先把表头带排除** —— 跨列表头会把相邻两列桥接起来。
    """
    if not words:
        return []
    lo = min(w.cross_lo_hi(axis)[0] for w in words)
    hi = max(w.cross_lo_hi(axis)[1] for w in words)
    n = max(1, int((hi - lo) / step) + 1)
    ink = bytearray(n)
    for w in words:
        a, b = w.cross_lo_hi(axis)
        for i in range(max(0, int((a - lo) / step)), min(n, int((b - lo) / step) + 1)):
            ink[i] = 1

    spans: list[tuple[float, float]] = []
    start = None
    run = 0
    for i in range(n):
        if ink[i]:
            if start is None:
                start = i
            if run and run * step > min_gap and spans:
                pass
            run = 0
        else:
            if start is not None:
                run += 1
                if run * step > min_gap:
                    spans.append((lo + start * step, lo + (i - run + 1) * step))
                    start = None
                    run = 0
    if start is not None:
        spans.append((lo + start * step, hi))
    return spans


def occupancy(band: list[Word], axis: str, cols: list[tuple[float, float]]) -> list[int]:
    """这一带在每一列里有几个词。列外的词记到最近的列。"""
    counts = [0] * len(cols)
    if not cols:
        return counts
    for w in band:
        c = (w.cross_lo_hi(axis)[0] + w.cross_lo_hi(axis)[1]) / 2
        best = min(range(len(cols)), key=lambda i: abs(c - (cols[i][0] + cols[i][1]) / 2))
        counts[best] += 1
    return counts
