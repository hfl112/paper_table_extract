"""全部业务规则与判据。零第三方依赖 —— 所有"会错"的判断都在这里，可纯单测。

每条规则都标注了它对应的 AGENTS.md「PDF 事实」编号和实测依据。
不要在这里引入任何学科词表（泛用性原则：领域知识由用户提供，不由工具内置）。
"""

from __future__ import annotations

import re

from .models import ExtractedTable, ImageInfo, Label, PageInfo, Rect, TableCandidate

# ---------------------------------------------------------------------------
# 标号判据（AGENTS.md PDF 事实 #1 #2）
# ---------------------------------------------------------------------------

# 默认标号「核心」正则 —— 只描述标号本身，不含锚点和结尾条件。
# 这是**排版惯例**而非领域知识，任何英文期刊都适用；可用 --label-pattern 覆盖。
#
# 复数形式（Tables / Figures / Figs）只出现在正文引用里（"Tables IV and V"），
# 归一化时会折回单数。注意：枚举式引用只能取到第一个标号（"Tables IV and V" → table iv），
# 这是已知限制。
# 前缀变体是**期刊惯例**，不是领域知识。只加语料里实测到的：
#   `Extended Data Table/Fig`  Nature 家族标准写法 —— 留出集 6 篇、84 处行首命中，
#                              现行正则一个都认不出（F-009）
#   `Supplemental Table`       美式拼写 —— 开发集 blood 有 1 处
#   `Supplementary Data Table` / `Online Table`
# **刻意不加** `eTable`（JAMA）与 `Additional file`（BMC），理由见 eval/findings.md F-009：
#   前者语料里 0 实例、加了无从验证；后者工具已经靠内嵌的 `Table SN` 正确处理成"只被引用"，
#   加了反而让同一实体出现两个标号。
DEFAULT_LABEL_CORE = (
    r"(?:Supplementary\s+Data\s+|Supplementary\s+|Supplemental\s+|Extended\s+Data\s+|Online\s+)?"
    r"(?:Tables?|TABLES?|Tabs?\.|Figures?|FIGURES?|Figs?\.?)\s+"
    r"(?:S?\d+|[IVXivx]{1,4})"
)

# 同一个核心要用两种结尾条件，混用会出错（本项目已经踩过）：
#
# caption 判据（逐「行」匹配，结尾允许行尾）：
#   1. 必须逐「行」匹配、不能逐 block —— PyMuPDF 会把相邻两个 caption 合并成一个 block。
#      实测 blood p6 把 `Figure 4.` 和 `Table 3.` 并进一个 block，`Table 3` 在 block 内偏移 931，
#      用 block 判据会漏掉整张 Table 3；而它在该 block 的第 12 行行首。
#   2. 标号后必须允许**行尾结束** —— 实测 pbc_28772 / pbc_29304 的 caption 行只有 `TABLE 1`
#      三个字，描述文字在下一行。要求后面必须跟 `.:` 或空格会漏掉它们。
_CAPTION_TAIL = r"(?:[.:\s]|$)"
#
# 宽松引用搜索（正文任意位置，结尾只要词边界）：
#   正文引用后面常跟逗号、括号、分号 —— `Table IV,` / `(Table IV)` / `Table IV;`。
#   若沿用 caption 的 `[.:\s]|$` 结尾，这些全都匹配不上，"只被引用"的名单会漏。
_LOOSE_TAIL = r"\b"

# 判断一个标号出现处是否为真 caption：它必须位于所在「行」的开头。
CAPTION_MAX_OFFSET = 0

_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# 关键词匹配语义
# ---------------------------------------------------------------------------

_VOWELS = "aeiou"
# 双写末尾辅音时排除这些（w/x/y 不双写；h/j/q 不构成这种形态）
_NO_DOUBLE = "wxyhjq"


def _morph_variants(word: str) -> list[str]:
    """给一个英文词生成常见屈折变体。纯英语构词法，**不含任何领域知识**。

    每条规则都有实测反例支撑（只加 s/es/d/ed/ing 会漏掉的）：
      y → ies      `toxicity`→`toxicities`、`activity`→`activities`、`study`→`studies`
      e → ing      `dose`→`dosing`（不是 doseing）
      末辅音双写    `control`→`controlled`/`controlling`（每张表都有 Control 组）

    含数字的词（`PD1`、`S2`）不做形态展开 —— 那是编号/代号，没有屈折。
    """
    w = word.lower()
    out = [w]
    if any(ch.isdigit() for ch in w):
        return out

    out += [w + "s", w + "es", w + "d", w + "ed", w + "ing"]
    if w.endswith("y") and len(w) > 2 and w[-2] not in _VOWELS:
        out.append(w[:-1] + "ies")
    if w.endswith("e"):
        out += [w[:-1] + "ing", w[:-1] + "ed"]
    if (
        len(w) >= 3
        and w[-1] not in _VOWELS
        and w[-1] not in _NO_DOUBLE
        and w[-2] in _VOWELS
        and w[-3] not in _VOWELS
    ):
        out += [w + w[-1] + "ed", w + w[-1] + "ing"]
    return sorted(set(out), key=len, reverse=True)


def keyword_regex(keyword: str, *, substring: bool = False) -> re.Pattern[str]:
    """把一个关键词编译成匹配用的正则。**默认全词 + 屈折变体，不区分大小写。**

    为什么默认不是子串（实测依据）：子串匹配下 2-3 字母的关键词全是噪声 ——
      `PR` 命中 130 处，全在英文单词内部（`rep`r`esentative` / `P`r`exasertib` / `P`r`otein`）
      `CR` 命中  41 处，大量来自 `vin`cr`istine`
    而 `CCR-18-2728` 那三张图（该篇没有响应码表）在子串模式下会被
    `response+(CR|PR|PD)` 全部误命中；全词模式下全部正确排除。

    反向的代价也实测过，所以必须带屈折变体 —— **纯全词会漏**：
      `Demographic` 全词 0 处（原文是 `Tumor Demographics`），带变体 2 处
      `xenograft`   全词 11 处，带变体 38 处（多数是 `xenografts`）

    多词短语（`overall response`）只对**最后一个词**做形态展开。

    **连字符按「词字符」处理** —— 即 `PR-104` 不满足 `-k PR`。
    这一条我先前选反了，是实测数据纠正的：语料里所有连字符 token 全是**标识符**，
    没有一个是"值得抓的复合词"：
      `PR-104` / `PR-104A`  药名（blood 那篇整篇讲这个药）→ 边界规则下 `-k PR` 会命中该篇每一处
      `NB-SD`               移植瘤系名 → `-k SD` 误命中
      `ALL-11` `KT-13` `CTG-0241` `JNJ-26481585` `ES-4` `BCP-ALL`  全是编号代号
    而"值得抓的连字符复合词"（如 `dose-response`）在语料里**一个都没有**
    —— 原文是空格写法，且那种曲线图本来就不是我们要的表。
    所以按词字符处理是零召回损失、消掉一整类误报。顺带 `PD-L1` 也不再满足 `-k PD`。
    """
    kw = collapse_ws(keyword)
    if substring:
        return re.compile(re.escape(kw), re.I)

    parts = kw.split(" ")
    head = r"\s+".join(re.escape(p) for p in parts[:-1])
    tail_alts = "|".join(re.escape(v) for v in _morph_variants(parts[-1]))
    body = (head + r"\s+" if head else "") + f"(?:{tail_alts})"
    return re.compile(rf"(?<![\w-]){body}(?![\w-])", re.I)


def keyword_hits(text: str, keywords: list[str], *, substring: bool = False) -> list[str]:
    """返回在 text 里命中的关键词（保持传入顺序）。"""
    return [k for k in keywords if keyword_regex(k, substring=substring).search(text)]


def collapse_ws(text: str) -> str:
    """折叠所有空白为单个空格。**匹配标号前必须先做这一步**（PDF 事实 #1）。

    实测同一个标号会出现 `Supplementary\\nTable S2` 与 `Supplementary Table\\nS2`
    两种断法，不折叠就会被当成两个不同标号。
    """
    return _WS.sub(" ", text).strip()


def normalize_label(raw: str) -> str:
    """把标号归一化成可比较的 key。

    `TABLE I` / `Table I` / `Tab. I` / `Tables I` → `table i`
    `Supplementary Fig. S4` / `Supplementary Figure S4` → `supp fig s4`
    复数形式折回单数 —— 复数只出现在正文引用里（"Tables IV and V"）。
    """
    s = collapse_ws(raw).lower()
    # `supplemental` 与 `supplementary` 是同一个意思，必须折成同一个 key ——
    # 否则 `Supplemental Table IV` 与 `Supplementary Table IV` 会被当成两张不同的表。
    s = s.replace("supplementary data ", "supp ").replace("supplementary ", "supp ")
    s = s.replace("supplemental ", "supp ")
    s = re.sub(r"\btabs?\.", "table", s)
    s = re.sub(r"\btables\b", "table", s)
    s = re.sub(r"\bfigures?\b", "fig", s)
    s = re.sub(r"\bfigs?\.", "fig", s)
    s = re.sub(r"\bfigs\b", "fig", s)
    s = re.sub(r"[.:,;]+$", "", s)
    s = collapse_ws(s)
    # 编号已带 S 前缀时，`supp ` 前缀是冗余的 —— 同一篇里作者常两种写法混用指同一张表。
    # 实测 CCR-18-2728 同时出现 `Supplementary Table S3` 与 `Tables S3`。
    # 但 `Supplementary Table 1`（编号无 S）要与 `Table 1` 保持区分，故只在带 S 时折叠。
    s = re.sub(r"^supp (table|fig) (s\d)", r"\1 \2", s)
    return s


# legend 常常**跨 block** —— 只取 caption 所在 block 会把描述段整段截断。
# 实测 pbc_30017 p7：`Table 1:` + 标题在 block2，而 `Tumor regression … All osteosarcoma
# models showed Progressive Disease 1 (PD1) as their objective response measure.`
# 整段在 block3，被完全漏掉（抓到的 legend 只有 60 字）。那段里有 `objective response
# measure`/`osteosarcoma models`/`PD1`，全是可搜的词。
#
# 续抓的判据要能区分「legend 续段」与「表体」，两个条件都必需 ——
# 各自单独用都会错一个实测案例：
#   pbc_30017 b3（legend 续段）  间距 8.7   最长行 107  → 吸收
#   pbc_30017 b4（表体 Cancer Type）间距 23.0  最长行 11  → 靠间距停
#   blood p6  b6（表体 Treatment） 间距 4.2 ← 很小 最长行 14 → **只看间距会误吸**，靠行长停
#   pbc_24724 p5 b4（页脚）        间距 291.8 最长行 36 → 靠间距停
LEGEND_MAX_BLOCK_GAP = 15.0  # pt
LEGEND_MIN_PROSE_LINE = 40  # 字符


def is_legend_continuation(gap: float, max_line_len: int) -> bool:
    """紧跟 caption 之后的 block 是不是 legend 的续段（而不是表体/页脚）。"""
    return gap < LEGEND_MAX_BLOCK_GAP and max_line_len >= LEGEND_MIN_PROSE_LINE


def label_kind(key: str) -> str:
    return "table" if "table" in key else "figure"


def compile_label_pattern(core: str | None = None) -> re.Pattern[str]:
    """caption 判据用的正则：行首锚定 + 允许行尾结束。"""
    return re.compile(rf"^(?P<label>{core or DEFAULT_LABEL_CORE}){_CAPTION_TAIL}")


def compile_loose_pattern(core: str | None = None) -> re.Pattern[str]:
    """引用搜索用的正则：不锚定、结尾只要词边界（正文引用后面常跟逗号/括号）。"""
    return re.compile(rf"(?P<label>{core or DEFAULT_LABEL_CORE}){_LOOSE_TAIL}")


def match_label_at_line_start(line: str, pat: re.Pattern[str]) -> str | None:
    """行首标号判据。返回标号原文，或 None。

    实测 6/6 篇与 docling 实际抽到的表完全吻合。基准反例（pbc_21078）：
      `Table I` 在 p2 偏移 94、p3 偏移 83  → 正文引用
      `TABLE I` 在 p4/p5 偏移 0            → 真 caption
    刻意**不依赖全大写** —— 那只是 Pediatr Blood Cancer 的期刊惯例，不通用。
    """
    m = pat.match(collapse_ws(line))
    if m and m.start() <= CAPTION_MAX_OFFSET:
        return m.group("label") if "label" in (m.groupdict() or {}) else m.group(0)
    return None


# 行首命中还不够 —— 正文引用可能**恰好被折行折到行首**，那是假 caption。
# 实测两例（CCR-18-2728）：
#   p7 line7  `Supplementary Fig. S4 and statistical analyses are summarized in…`
#             上一行结尾是 `M. Waterfall plots for each model are shown in`  ← 句子未完
#   p4 line3  `Supplementary Fig. S1. cCalculated from 72-hour time point…`（表格脚注）
#             上一行结尾是 `Max Inhibition ± SEM; complete curves shown in` ← 句子未完
# 而真 caption 要么在 block 第 0 行，要么前一行是**完整句**：
#   blood p6 line12 的真实 `Table 3.` 前一行结尾是 `…when it reached 25%.`
#
# 所以判据：位于 block 首行，或前一行以句末标点收尾。
# 不能改用"必须在 block 首行"—— 那会砍掉 blood p6 那个被并进 Figure 4 block 的真 caption。
_SENTENCE_END = re.compile(r"[.!?:][\"')\]]*$")


def caption_line_is_plausible(prev_line: str | None) -> bool:
    """给行首命中做二次确认：前一行必须不存在，或是完整句。"""
    if prev_line is None:
        return True
    prev = collapse_ws(prev_line)
    if not prev:
        return True
    return bool(_SENTENCE_END.search(prev))


def find_labels_anywhere(text: str, loose_pat: re.Pattern[str]) -> list[str]:
    """在一段文字里找出所有标号（不要求行首）—— 用于统计"只被引用"的标号。

    必须传 compile_loose_pattern() 的结果，不能传 caption 判据那个 —— 后者的结尾
    条件 `[.:\\s]|$` 会漏掉 `Table IV,` / `(Table IV)` 这类最常见的引用写法。
    """
    return [m.group("label") for m in loose_pat.finditer(collapse_ws(text))]


# ---------------------------------------------------------------------------
# 续表判据（AGENTS.md PDF 事实 #12）—— 两条都必须实现，只做一条会漏
# ---------------------------------------------------------------------------

# 实测出现过 `( Continued )` 这种括号内带空格的写法（pbc_21078 p5）。
_CONTINUED = re.compile(r"\(\s*continued\s*\)", re.I)


def caption_says_continued(caption: str) -> bool:
    """判据一：caption 里明写 (Continued)。

    实测：pbc_21078 p5 `TABLE I. ( Continued )`、pbc_28772 p4 与 pbc_29304 p4 `TABLE 1 (Continued)`。
    """
    return bool(_CONTINUED.search(collapse_ws(caption)))


def same_column_names(a: list[str], b: list[str]) -> bool:
    """判据二：列名逐字相同。

    实测 pbc_26870 的 Table 1 跨 p18–p19，**p19 的 caption 是空字符串**，
    唯一线索就是两页列名完全一致（8 列：Tumor Line | Treatment Group | ... | Response）。
    只做判据一会漏掉这一整张续表。
    """
    if not a or not b or len(a) != len(b):
        return False
    return [collapse_ws(x).lower() for x in a] == [collapse_ws(x).lower() for x in b]


def is_continuation(
    prev: ExtractedTable, cur: ExtractedTable, *, max_page_gap: int = 1
) -> str | None:
    """判断 cur 是不是 prev 的续表。返回命中的判据名，或 None。"""
    if cur.candidate.page - prev.candidate.page > max_page_gap:
        return None
    cur_caption = cur.candidate.label.text if cur.candidate.label else ""
    if caption_says_continued(cur_caption):
        return "caption_continued"
    prev_cols = prev.rows[0] if prev.rows else []
    cur_cols = cur.rows[0] if cur.rows else []
    if same_column_names(prev_cols, cur_cols):
        return "same_columns"
    return None


# ---------------------------------------------------------------------------
# 页面结构判据（AGENTS.md PDF 事实 #3 #8b）
# ---------------------------------------------------------------------------

# 竖排文字行数阈值。实测真横向表 134/150/198/208/264/352/446 行，Wiley 侧边水印**恒为 1 行**，
# 中间隔两个数量级。**必须用绝对行数，不能用占比** —— pbc_21078 p7 是图片页、水平文字仅 7 行，
# 那 1 行水印的占比就冲到 12.5%，而真横向表 p5 的占比是 68.7%，占比判据会误判 p7。
VERTICAL_LINES_THRESHOLD = 30

# 单张位图覆盖页面的比例阈值。≥ 此值 ⇒ 该页文字层是 OCR 派生 ⇒ 该页所有表无条件 low。
# 实测 2/2 命中、10/10 无误报：两篇扫描件 100%×101% 与 100%×100%；
# 其余 10 篇最大仅 83%×63%（pbc_29304 p5），有 12 个百分点余量。
FULL_PAGE_IMAGE_COVER = 0.95

# 区域内文字层字符数阈值。< 此值 ⇒ 内容是像素 ⇒ 走 OCR 路径。
# 实测分离极干净：三张位图框内字符**全是 0**，而所在页有 1480–4202 字符。
TEXT_IN_REGION_THRESHOLD = 30

# OCR 闸门：单元格检测数阈值。实测含表的图 300 个，KM 曲线图 0–6 个，差两个数量级。
TABLE_CELLS_THRESHOLD = 20

# 图片路径的第二意见：把单元格检测器给出的**列带/行带数**与还原结果的行列数对账。
#
# 图片路径没有文字层，coverage 与 pdfplumber 都用不上 —— 但闸门那一步已经免费数过
# 每个格子的坐标，聚类即得期望的行列数，零额外成本。
#
# 为什么不比总格子数：实测还原结果曾是 46x6=276 格，正确形状是 45x7=315 格，
# 而检测器给 300 格 —— 相对 300 分别差 8% 与 5%，**任何阈值都分不开**。
# 比列数则一目了然（6 vs 7）。这是"别去调阈值迁就数据、换个更直接的信号"的一个例子。
# 行列容差刻意不同：
#   列差 1 = **整列数据丢失**，必须报（实测那次就是丢了 Heat Map 一列）。
#   行差 1 = 正常噪声 —— 还原结果的第 0 行是表头，检测器的行带可能把表头算作一带
#            也可能不算，±1 属预期。
GRID_COL_TOLERANCE = 0
GRID_ROW_TOLERANCE = 1


def grid_shape_mismatch(
    det_rows: int, det_cols: int, n_rows: int, n_cols: int
) -> str | None:
    """图片表的免费第二意见。返回红旗文字，或 None。"""
    if det_rows <= 0 or det_cols <= 0 or n_rows <= 0 or n_cols <= 0:
        return None
    bad = []
    if abs(det_cols - n_cols) > GRID_COL_TOLERANCE:
        bad.append(f"列 检测{det_cols}/还原{n_cols}")
    if abs(det_rows - n_rows) > GRID_ROW_TOLERANCE:
        bad.append(f"行 检测{det_rows}/还原{n_rows}")
    if not bad:
        return None
    return "grid_shape_mismatch(" + ", ".join(bad) + ")"

OCR_PRODUCER_RE = re.compile(
    r"ABBYY|Recognition|Tesseract|FineReader|OmniPage|ocrmypdf|Scansoft", re.I
)


# 竖排**占比**下限。绝对行数与占比**两个条件都必需**，各自单独用都会错 ——
# 这是留出集抓出来的：我原先只用绝对行数，因为当时只见过两档数据。
#   Wiley 侧边下载水印        竖排   1 行            → 绝对行数挡住
#   Becker 2022 多面板图轴标签 竖排 41-79 行、占比 17-25%  → **只用绝对行数会误判**，靠占比挡住
#                             （单细胞论文一页几十个小图，每个图的 Y 轴标签都是竖排的：
#                              `UMAP dimension 2` / `log2FC (36,374 peaks)` / `PC2 (7.0%)`）
#   真横向表                  竖排 134-446 行、占比 69-99% → 两个都过
#   pbc_21078 p7 图片页        竖排 1 行、占比 12.5%      → **只用占比会误判**，靠绝对行数挡住
VERTICAL_RATIO_THRESHOLD = 0.5


def rotation_for_page(
    vert_lines: int,
    dominant_dir: tuple[int, int],
    horiz_lines: int = 0,
    *,
    has_table_label: bool = False,
    prev_page_rotated: bool = False,
) -> int:
    """该页需要转多少度才能让横向表变正。**三个条件**。

    实测 4/4：竖排方向 (0,-1) 时 rotation=90 全对，0 与 270 全错。
    方向 (0,1) 时按对称性应为 270 —— 实测触发过 4 次，**4 次全是误报、正例仍然 0 个**
    （见 eval/findings.md F-012）。

    ═══ 为什么要第三条（2026-07-26 加，留出集 + 阈值验证集共 78 篇实测）═══

    原来是「行数 >=30 且 占比 >=50%」。`AGENTS.md` 事实 2c 说过单条不够、两条才够 ——
    **现在两条也不够了**：

      Li et al. 2025 p6         竖排 1555 行、占比 85%  → 多面板图的 Y 轴标签
      Williams(cells of) p15    竖排   90 行、占比 67%  → 染色体臂标签
      5D5JKFJM p17 / 5PMEJG7T p6,p9  竖排 71 行、占比 51-64%
                                → **`©2017...` 版权声明被 PDF 逐字符拆成 71 个"竖排行"**

    这些的占比（51-85%）全落在真横向表的区间（57-99%）里，两条判据的组合分不开。

    第三条：**竖排文字里含 `Table N` 标号** —— 横向表的 caption 是跟着表一起转过来的。
    实测 8 页真横向表全部有、6 页误报全部没有。

    `prev_page_rotated` 是必需的兜底：`pbc_26870` p19 是真横向表，但它是**续表页、
    caption 为空**（`AGENTS.md` 事实 #12 的那个样本），竖排里没有标号。
    **单靠标号判据会误杀它。**
    """
    if vert_lines < VERTICAL_LINES_THRESHOLD:
        return 0
    total = vert_lines + horiz_lines
    if total and vert_lines / total < VERTICAL_RATIO_THRESHOLD:
        return 0  # 多面板图的 Y 轴标签，不是横向表
    if not (has_table_label or prev_page_rotated):
        return 0  # 占比够高但没有标号、上一页也不是横向表 —— 轴标签 / 逐字符水印
    return 270 if dominant_dir == (0, 1) else 90


# 判"出版社 OCR 文字层"还需要该页文字层足够**丰富** —— 光看"整页被位图覆盖"不够。
#
# 这是**留出集**（25 篇从未参与调参的现代论文）抓出来的泛化 bug：
#   真扫描件      EAP gastric  整页位图 100%x101%，文字层 5255-6398 字/页
#                 rstl.1665    整页位图 100%x100%，文字层 2163-2462 字/页
#   整版大图页    Beddows 2023 p24-29 整页位图 100%x100%，文字层**仅 185 字/页**（只是页眉）
# 前者的文字层是图像内容的 OCR、可用；后者是全页 supplementary figure，文字层里没有内容。
# 原规则把两者混为一谈，而代码对 ocr_text_layer 的页**跳过图片候选** ——
# 于是 Beddows 那 6 页的整版大图永远不会被 OCR，**静默漏表**。
# 阈值 1000：真扫描件最低 2163（2 倍余量），整版大图页 185（5 倍余量）。
OCR_LAYER_MIN_CHARS = 1000


def has_full_page_image(images: list[ImageInfo]) -> bool:
    """该页是否被单张位图覆盖 ≥95% 宽和高。"""
    return any(
        im.cover_w >= FULL_PAGE_IMAGE_COVER and im.cover_h >= FULL_PAGE_IMAGE_COVER
        for im in images
    )


# 判据从「单页」改成「整篇」—— 2026-07-26，90 篇实测推翻了原来的逐页字数判据。
#
# 原判据（整页位图 且 该页文字层 >= 1000 字）**两类错同时存在，而且顺序是反的**：
#   真扫描页  Dobzhansky_1946 p12=602 / Rous_1911 p21=681 / Bell_1876 p3=832  → **漏判**
#   现代大图页 Chunduri_2022 p7=1008 / Drapkin_2018 p1=1122                    → **误报**
# 真扫描页的字比误报页**更少**，所以调高阈值加重漏判、调低加重误报 —— 字数分不开这两类。
#
# 换个维度就完美分离：**真扫描件是每一页都被位图覆盖，现代论文只有 1–2 页大图。**
#   真扫描件 21 篇（1876–1995）  整页位图页占比 97%–100%
#   现代论文含大图 5 篇          整页位图页占比  6%– 21%（含 log.md §8 的 Beddows 21%）
# 余量 76 个百分点，零误报零漏判。
SCANNED_DOC_PAGE_RATIO = 0.8


def is_scanned_document(n_full_page_images: int, n_pages: int, total_chars: int) -> bool:
    """整篇是不是「扫描件 + 出版社 OCR 文字层」。

    两个条件：
      1. **整页位图页占比 >= 80%** —— 主判据，余量 21% vs 97%
      2. **整篇平均每页 >= OCR_LAYER_MIN_CHARS 字** —— 挡住"纯图、根本没 OCR"那种。
         那种得走 OCR 路径，误判成"有 OCR 文字层"会让它**跳过图片候选、整篇丢光**。
         实测 21 篇真扫描件是 1128–9668 字/页（最低 `Bell_1876`），阈值 1000 有 13% 余量；
         **负例侧无样本** —— 手上没有"整本扫描但没做 OCR"的 PDF。
    """
    if not n_pages:
        return False
    if n_full_page_images / n_pages < SCANNED_DOC_PAGE_RATIO:
        return False
    return total_chars / n_pages >= OCR_LAYER_MIN_CHARS


def is_ocr_text_layer(images: list[ImageInfo], *, doc_is_scanned: bool) -> bool:
    """该页的文字层是不是"出版社 OCR 生成的、可用的"。

    **要整篇先判为扫描件**（见 `is_scanned_document`），再看这一页是不是被位图覆盖 ——
    扫描件里偶尔有非整页位图的页（如 `Triolo_1965` 32/33），那些页不算。
    """
    return doc_is_scanned and has_full_page_image(images)


def metadata_suggests_ocr(metadata: dict[str, str]) -> bool:
    """元数据里的 OCR 特征词。**只能辅助确认，不能反推** ——
    实测 rstl.1665.0039 的 producer 是 `ABBYY Recognition Server`（命中），
    但 EAP gastric cancer 的 creator/producer 全是空字符串（漏）。
    """
    blob = " ".join(v or "" for v in metadata.values())
    return bool(OCR_PRODUCER_RE.search(blob))


# 位图候选的最小页高占比 —— 用来滤掉期刊 logo / 出版社徽标。
# 实测依据：logo 的页高占比 pbc_28772 是 3.9% 与 1.9%、CCR-18-2728 是 3.5%；
# 而真实内容最小的是 blood p7 的多面板显微照片 13.2%，含表格的图是 39.5%。
# 阈值 8% 两侧各有 2-3 倍余量。用页高而非页宽，因为 logo 常是细长横条（12.8% 宽 × 1.9% 高）。
MIN_IMAGE_COVER_H = 0.08


def is_decorative_image(cover_h: float) -> bool:
    return cover_h < MIN_IMAGE_COVER_H


def dispatch_source_type(chars_in_rect: int) -> str:
    """分流判据：区域内文字层字符数决定走哪条路。

    这自动涵盖了矢量表 vs 位图表之别，无需专门区分 ——
    矢量表的线条是矢量指令但文字在文字层（pbc_28772 p3 有 181 条矢量线、框内有字）→ 文字路径；
    位图表框内 0 字 → OCR 路径。
    """
    return "text" if chars_in_rect >= TEXT_IN_REGION_THRESHOLD else "image"


# ---------------------------------------------------------------------------
# 审计（覆盖率 / 列数第二意见 / 列内字符一致性）
# ---------------------------------------------------------------------------


def coverage_ratio(cell_chars: int, region_chars: int) -> float | None:
    """内容覆盖率 = 抽出的单元格总字符 ÷ 区域文字层字符。

    这是区分「合法的合并单元格」与「识别器把两行挤成一格」的判据（PDF 事实 #15）——
    两者在 CSV 里都表现为"有空格子"，但前者不丢字符、后者丢。
    也能抓结构性崩塌：pbc_29304 p4 未转正时 docling 给 1x2（约 30 字符），
    而该页有 1308 字符 → 覆盖率 ~2%，一眼就是崩了。
    """
    if region_chars <= 0:
        return None
    return cell_chars / region_chars


COVERAGE_LOW = 0.35


def audit(table: ExtractedTable, *, region_chars: int, ocr_page: bool) -> ExtractedTable:
    """给一张抽取结果打 coverage / confidence / grid_status / notes。就地修改并返回。"""
    notes = table.notes

    if table.candidate.source_type == "text":
        table.coverage = coverage_ratio(table.cell_chars, region_chars)
        if table.coverage is not None and table.coverage < COVERAGE_LOW:
            notes.append(f"coverage_low={table.coverage:.0%}")

    if table.plumber_cols is not None and table.n_cols:
        if abs(table.plumber_cols - table.n_cols) > 1:
            notes.append(f"col_mismatch(docling={table.n_cols},plumber={table.plumber_cols})")

    for hint in suspect_lt_as_minus(table.rows):
        notes.append(hint)

    table.confidence = decide_confidence(table, ocr_page=ocr_page)
    return table


def decide_confidence(table: ExtractedTable, *, ocr_page: bool) -> str:
    """三档，不要自创第四档。

    low  —— 任何 OCR 派生（无条件）。包括「出版社 OCR 文字层」的页：
             实测 docling 能从那种文字层的字符坐标恢复出正确行列结构，但字符本身仍是 OCR 产物
             （`Cardia`→`Cordia`、`et al.`→`et al'`），所以结构对≠内容可信。
    high —— docling 抽到 + 覆盖率正常 + pdfplumber 列数一致(±1) + 无 grid_mismatch
    medium —— 其余
    """
    if table.candidate.source_type == "image" or ocr_page:
        return "low"
    if table.grid_status != "ok":
        return "medium"
    if table.coverage is None or table.coverage < COVERAGE_LOW:
        return "medium"
    if table.plumber_cols is None:
        return "medium"
    if abs(table.plumber_cols - table.n_cols) > 1:
        return "medium"
    return "high"


# docling 会把 PDF 里**未映射的字形名**原样写进单元格 —— 实测留出集 `Medina 2025`
# 的缩进用 em space，导出成 `/emspaceMean` / `/emspaceRange` / `/emspaceI`（27 处），
# `Chang_2023` 有 1 处 `/bowtie`。开发集 12 篇**零出现**，是留出集补上的一档。
#
# **只认已知的空白字形名、且保留后面的内容。** 第一版写成 `^/[a-zA-Z]{2,15}` 贪婪匹配，
# 把 `/emspaceMean` 整格吃成空串 —— 而留出集里有 24 格是 `/emspaceClear cell`、
# `/emspaceHigh-grade serous` 这类**卵巢癌组织学分型**，那样会丢数据。
_SPACE_GLYPHS = "em|en|thin|hair|figure|punctuation|third|quarter|sixth|zerowidth|nb|no-break"
_GLYPH_NAME = re.compile(rf"^/(?:{_SPACE_GLYPHS})space\s*", re.I)


def strip_glyph_names(rows: list[list[str]]) -> list[list[str]]:
    """就地清掉单元格开头漏出来的空白字形名。返回同一个 rows。"""
    for row in rows:
        for i, cell in enumerate(row):
            if cell.startswith("/"):
                row[i] = _GLYPH_NAME.sub("", cell)
    return rows


# ---------------------------------------------------------------------------
# 断行连字符（F-017）—— docling 把「一个词换行断开」的两截拼回同一格时，
# 留下了连字符和空格：`EPZ011989 + cyclophos- phamide`（真值 `…cyclophosphamide`）。
# ---------------------------------------------------------------------------
#
# **docling 判对了格子归属，只是拼接约定不同** —— 它认出 `cyclophos-` 和 `phamide`
# 属于同一格，只是拼的时候保留了连字符、还插了个空格。本判据只补这最后一步。
#
# ═══ 为什么不能只用正则（第一版就是这么错的）═══
#
# 两件完全不同的事产生**一模一样**的表面形式，字符串里不含区分它们的信息：
#     换行断词      `cyclophos-` + 换行 + `phamide`      → 该接
#     合法复合词    `lineage- defining` / `patient- derived` / `p-value` → 不该接
# 纯正则版在三个语料上会改坏 `lineage-defining`、`patient-derived`、`Log-rank p-value`，
# 还有 URL（`…childhood-cancer- statistics/`、`bcl2fastq- conversion-software`）。
#
# ═══ 判据：拼接结果必须在该篇文字层里作为完整词出现 >= MIN_WORD_COUNT 次 ═══
#
# 真断行拼出来是**一个真词**，合法复合词拼出来是个不存在的字符串。
#
# `MIN_WORD_COUNT = 2` 而不是 1 —— 三语料实测（`pdfio.doc_word_counts` 的计数）：
#     该接的 11 处：electrophoresis 2 / replacement 3 / frequency 3 / subpopulations 3 /
#                   nondisjunction 3 / transcarbamylase 6 / neoplasia 7 / cyclophosphamide 10 /
#                   objectness 16 / chromosome 27 / regression 153
#     不该接的 1 处：**channelaware 1**（`Channel-aware` 是标准术语；那 1 次本身就是别处断行的产物）
# ⚠ **余量是相邻整数（1 <- 2 -> 2），这是最紧的一种。** 失效方向是"不接"（安全侧）。
#
# 三语料记录：接对 12 处、正确不动 25 处、0 处改错。
_HYPHEN_BREAK = re.compile(r"([a-z])-\s+([a-z])")
_WORD_RUN = re.compile(r"[a-z]+")
MIN_WORD_COUNT = 2


def has_hyphen_break(rows: list[list[str]]) -> bool:
    """便宜的预筛：这张表里有没有可能需要接的地方。

    用来避免为每一篇文档都白算一次整篇词表（`pdfio.doc_word_counts` 要读全文）。
    """
    return any(_HYPHEN_BREAK.search(c) for row in rows for c in row)


def rejoin_hyphen_breaks(
    rows: list[list[str]], word_counts: dict[str, int]
) -> tuple[list[list[str]], int]:
    """接回断行连字符。返回 (**新的** rows, 接回处数)。

    **必须返回新 list** —— `__main__.run_text_path` 里 `ExtractedTable.rows` 与
    `DoclingResult.tables[i].rows` 是同一个对象（`stitch` 会就地 extend 它），
    就地改会污染 `dl`。
    """
    n = 0

    def fix(s: str) -> str:
        nonlocal n

        def repl(m: re.Match) -> str:
            nonlocal n
            left = (_WORD_RUN.findall(s[: m.start(1) + 1].lower()) or [""])[-1]
            right = (_WORD_RUN.findall(s[m.end(2) - 1 :].lower()) or [""])[0]
            joined = left + right
            if joined and word_counts.get(joined, 0) >= MIN_WORD_COUNT:
                n += 1
                return m.group(1) + m.group(2)
            return m.group(0)  # 查不到 → 一个字不动

        return _HYPHEN_BREAK.sub(repl, s)

    return [[fix(c) for c in row] for row in rows], n


_LEADING_LT = re.compile(r"^\s*<")
_LEADING_MINUS = re.compile(r"^\s*-\s*\d")


def suspect_lt_as_minus(rows: list[list[str]]) -> list[str]:
    """抓 OCR 把 `<` 读成 `-`：`<0.001` → `-0.001`（AGENTS.md PDF 事实 #11）。

    这是最危险的一类 OCR 错误：结果是一个**看起来完全合法的数值**，不触发任何格式告警。

    判据是**列内字符集一致性**，不含任何领域知识：一列里多数值以 `<` 开头、
    某一格以 `-` 开头 → 可疑。不需要知道那一列是 p 值还是别的东西。
    """
    if len(rows) < 3:
        return []
    n_cols = max(len(r) for r in rows)
    out: list[str] = []
    for c in range(n_cols):
        col = [(i, r[c]) for i, r in enumerate(rows) if c < len(r) and r[c]]
        if len(col) < 3:
            continue
        n_lt = sum(1 for _, v in col if _LEADING_LT.match(v))
        if n_lt < max(2, len(col) // 3):
            continue
        for i, v in col:
            if _LEADING_MINUS.match(v):
                out.append(f"suspect_lt_as_minus(R{i}C{c}={v.strip()!r})")
    return out


# ---------------------------------------------------------------------------
# caption 归属校验（铁律 #1）
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "table",
    "figure",
    "fig",
    "supplementary",
    "supp",
    "continued",
    "of",
    "the",
    "in",
    "for",
    "and",
    "a",
    "an",
    "to",
    "on",
    "with",
    "from",
    "by",
    "as",
    "at",
    "all",
    "summary",
    "results",
    "data",
}


def caption_words(caption: str) -> set[str]:
    """caption 里的实词，用于归属校验。停用词表是**排版/语法**层面的，不含学科词汇。"""
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", collapse_ws(caption).lower())
    return {w for w in words if w not in _STOPWORDS}


# 竞争 caption 至少要与网格重叠这么多个实词，才算"错配的正面证据"。
# 1 个词太松（会被偶然碰巧的单词命中），真正的错配会有大量重叠。
MIN_RIVAL_OVERLAP = 2


def _grid_haystack(rows: list[list[str]]) -> str:
    header = " ".join(rows[0]) if rows else ""
    first_col = " ".join(r[0] for r in rows if r)
    return collapse_ws(f"{header} {first_col}").lower()


def _overlap(caption: str, hay: str) -> int:
    return sum(1 for w in caption_words(caption) if w in hay)


def verify_caption_belongs(
    caption: str, rows: list[list[str]], *, other_captions: list[str] | None = None
) -> bool:
    """铁律 #1：一个网格在被信任高于原文之前，必须先验证它确实属于它所声称代表的那张表。

    前身项目最痛的 bug：上游把 Table S1 的网格错配给了 Table 2，而 prompt 无条件命令
    "网格是权威、绝不用原文覆盖网格"，于是忠实地抽出了一张错误的表。

    判据是**相对比较**，不是绝对阈值 —— 这是踩过的坑：
      早先要求"caption 实词必须与表头/首列有重叠"，结果误报了两处正常情况 ——
      `TABLE I. Tumor Demographics` 的表头是 `Xenograft line | Histology | ...`，
      `Table 1. Patient Characteristics` 的表头是 `'' | Preusser et al | Taal et al`。
      caption 描述的是**主题**，表头列的是**变量名**，本来就常常不重叠。
      **缺少证据不等于有反证。**

    所以只在有**正面反证**时才判错配：本 caption 与网格零重叠，而同篇**另一个** caption
    与该网格重叠达到 MIN_RIVAL_OVERLAP 个实词 —— 那才说明这个网格更像是别人的。

    `other_captions` 应只传**同类**（表 vs 表）的 caption。实测踩坑：pbc_21078 的
    `Fig. 1` 图注里列举了 `EW5` `SK-NEP-1` `Rh28` `KT-13` 这些移植瘤系名称，而它们正是
    `TABLE I` 首列的内容 —— 图注天然会列举表格的行标识，混进来必然误报。
    另外要求至少 MIN_RIVAL_OVERLAP 个词，是为了滤掉偶然的单词碰巧重合；真正的错配
    （前身项目那次）会让真主人的 caption 与网格**大量**重叠。
    """
    if not rows:
        return True
    hay = _grid_haystack(rows)
    if _overlap(caption, hay) > 0:
        return True
    best_other = max((_overlap(c, hay) for c in (other_captions or [])), default=0)
    return best_other < MIN_RIVAL_OVERLAP


# ---------------------------------------------------------------------------
# 同句共现（label_referenced_but_absent）
# ---------------------------------------------------------------------------

# 句子切分时要保护的缩写，否则 `Fig. 1` / `et al.` / `e.g.` / `No.` 会被误切成两句。
# 变长交替不能放进 look-behind（`look-behind requires fixed-width pattern`），
# 所以做法是：先找出所有候选切点，再逐个检查切点前是否为缩写、是则否决。
_ABBREVS = (
    "fig", "figs", "tab", "tabs", "al", "e.g", "i.e", "cf", "vs", "no", "nos",
    "ref", "refs", "approx", "ca", "dr", "mr", "ms", "st", "eq", "ch", "pp",
)
_SPLIT_CANDIDATE = re.compile(r"\.(?=\s+[A-Z(])")
_WORD_BEFORE = re.compile(r"([A-Za-z.]+)$")


def split_sentences(text: str) -> list[str]:
    """保守的句子切分：句点后跟空白 + 大写字母才算断句，且保护常见缩写与小数。"""
    flat = collapse_ws(text)
    flat = re.sub(r"(\d)\.(\d)", r"\1<DOT>\2", flat)  # 保护小数

    cuts: list[int] = []
    for m in _SPLIT_CANDIDATE.finditer(flat):
        before = _WORD_BEFORE.search(flat[: m.start()])
        if before and before.group(1).lower().rstrip(".") in _ABBREVS:
            continue  # 是缩写，不切
        cuts.append(m.start())

    parts, prev = [], 0
    for c in cuts + [len(flat)]:
        seg = flat[prev:c].strip()
        if seg:
            parts.append(seg.replace("<DOT>", "."))
        prev = c + 1
    return parts


def cooccurs_in_sentence(text: str, label_raw: str, keywords: list[str]) -> list[str]:
    """标号与关键词是否**同句**共现。返回命中的关键词。

    刻意限制到同句：一段里提了 5 个标号和 8 个关键词，按段落算会连出一堆假关联。
    """
    lab = collapse_ws(label_raw).lower()
    hits: set[str] = set()
    for sent in split_sentences(text):
        low = sent.lower()
        if lab not in low:
            continue
        hits.update(k for k in keywords if k.lower() in low)
    return sorted(hits)


# ---------------------------------------------------------------------------
# 标号 ↔ 候选区域 配对
# ---------------------------------------------------------------------------


def pair_label_to_candidate(
    label: Label, candidates: list[TableCandidate]
) -> TableCandidate | None:
    """把一个真 caption 配到同页的候选区域。

    惯例方向：**表标题在表的上方**，图标题在图的下方。所以对 table 类标号，
    优先取 caption 下方、纵向距离最近的候选；取不到再放宽到同页任意候选。
    """
    same_page = [c for c in candidates if c.page == label.page]
    if not same_page:
        return None
    if label.rect is None:
        return same_page[0]

    want_below = label.kind == "table"
    y = label.rect.y1 if want_below else label.rect.y0

    def key(c: TableCandidate) -> tuple[int, float]:
        if want_below:
            return (0 if c.rect.y0 >= y - 4 else 1, abs(c.rect.y0 - y))
        return (0 if c.rect.y1 <= y + 4 else 1, abs(y - c.rect.y1))

    return sorted(same_page, key=key)[0]


def page_needs_rotation(page: PageInfo) -> bool:
    return page.needs_rotation != 0


def region_chars_of(rect: Rect, spans: list[tuple[Rect, str]]) -> int:
    """数落在 rect 内的文字层字符（按 span 中心点判定）。"""
    total = 0
    for r, text in spans:
        cx = (r.x0 + r.x1) / 2
        cy = (r.y0 + r.y1) / 2
        if rect.contains_point(cx, cy):
            total += len(text.strip())
    return total


# ---------------------------------------------------------------------------
# 表 × 关键词
# ---------------------------------------------------------------------------


def match_table(
    table,
    keywords: list[str],
    *,
    match_all: bool = False,
    include_content: bool = True,
    substring: bool = False,
) -> tuple[list[str], str]:
    """关键词匹配一张表。返回 (命中的关键词, 靠哪儿命中的)。

    **文字表必须把表格内容也算进来，而且这不花钱。**
    实测依据：pbc_28772 的金标准表 caption 是
    `TABLE 1 EPZ011989 as a single agent and in combination for all PDX models`
    —— 不含 `response`；但它第 11 列的表头是 `Obj. Response`。
    只匹配 caption+legend 会漏掉这张明显该命中的表。

    为什么免费：docling 是**整篇跑一次**，过滤发生在抽取之后，
    所以过滤时单元格内容早就在内存里了。（早先把"匹配表内内容"整体判成昂贵操作，
    那个判断只对图片表成立 —— 那条路要 OCR 才知道图里写了什么，故保留三级梯度。）

    `include_content=False` 用于图片表：它的内容要付 OCR 代价，由调用方按梯度决定。
    """
    cap = table.caption_text
    body = table.content_text if include_content else ""
    cap_hits = keyword_hits(cap, keywords, substring=substring)
    body_hits = keyword_hits(body, keywords, substring=substring)
    union = [k for k in keywords if k in cap_hits or k in body_hits]

    if match_all and len(union) != len(keywords):
        return [], ""
    if not union:
        return [], ""
    if cap_hits and body_hits and set(cap_hits) != set(body_hits):
        return union, "caption+cell"
    return union, ("caption" if cap_hits else "cell")
