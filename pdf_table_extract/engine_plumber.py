"""pdfplumber 引擎。**pdfplumber 的唯一入口。**

它在本项目里**只做一件事：给定 bbox 报告列数，当第二意见。**

为什么只做这一件事（实测依据，见 AGENTS.md PDF 事实 #5）：
  - 当「发现器」失效：这些期刊用三线表（无竖线），`find_tables()` 在 pbc_28772 /
    pbc_29304 上返回 0 张表，pbc_21078 的真表在 p4/p5 却只在 p10/p11 命中。
  - 当「行数第二意见」不可用：给定 bbox 后 `both-text` 策略数的是**文字行**，
    docling 数的是**逻辑行**（一个单元格里 `Model\\nG401` 占两行文字算一行逻辑），
    实测比值 1.07-2.18 都属正常，没法据此判断有没有 merge。
  - 当「列数第二意见」可靠：实测 4 次里 2 次完全一致（pbc_24724 p3 6 vs 6、
    p4 10 vs 10），2 次差 1-2 列。

行数的第二意见改用 rules.coverage_ratio（内容覆盖率），它不依赖 bbox 精度。
"""

from __future__ import annotations

import warnings
from pathlib import Path

from .models import Rect

warnings.filterwarnings("ignore", module="pdfplumber")

# 三线表没有竖线，所以竖直方向必须靠文字对齐。约束在表格 bbox 内之后，
# 之前压倒信号的噪声（双栏正文、页眉页脚、参考文献）已被排除。
_SETTINGS = (
    {"horizontal_strategy": "text", "vertical_strategy": "text"},
    {"horizontal_strategy": "lines", "vertical_strategy": "text"},
)


def extract_in_bbox(pdf_path: str | Path, page_no: int, rect: Rect) -> list[list[str]]:
    """在给定 bbox 内抽出二维表。抽不到返回 []。

    两个用途：
      1. `columns_in_bbox()` 拿它的列数当**第二意见**（文字表，docling 已给出网格）。
      2. **矢量图里的表的唯一抽取手段** —— 实测 pbc_21078 的 Fig. 3（p10）是矢量绘制的
         热图，docling 把它归成 PictureItem 而非 TableItem，所以没有 docling 网格；
         而该页一张位图都没有，OCR 路径也无从下手。此时只能靠这个函数。
         那块区域实测有 1036 个文字层字符，pdfplumber 能抽出 8 列。
    """
    import pdfplumber

    try:
        with pdfplumber.open(str(pdf_path)) as doc:
            if page_no < 1 or page_no > len(doc.pages):
                return []
            page = doc.pages[page_no - 1]
            x0 = max(0.0, rect.x0)
            y0 = max(0.0, rect.y0)
            x1 = min(float(page.width), rect.x1)
            y1 = min(float(page.height), rect.y1)
            if x1 - x0 < 1 or y1 - y0 < 1:
                return []
            region = page.crop((x0, y0, x1, y1))
            for st in _SETTINGS:
                try:
                    tbl = region.extract_table(st)
                except Exception:
                    continue
                if tbl:
                    return [[("" if c is None else str(c)) for c in row] for row in tbl]
    except Exception:
        return []
    return []


def columns_in_bbox(pdf_path: str | Path, page_no: int, rect: Rect) -> int | None:
    """在给定 bbox 内抽一次表，只返回列数（第二意见）。失败返回 None。"""
    rows = extract_in_bbox(pdf_path, page_no, rect)
    if not rows:
        return None
    return max(len(r) for r in rows)
