"""rules.py 的单测。零第三方依赖，跑得飞快。

每个 case 都来自 0.pdf_input/ 里真实文献的实测结果 —— 不是编造的例子。
注释里的篇名/页码是可以回查的基准。

    conda run -n gemini python -m pytest tests/ -q
"""

from __future__ import annotations

from pdf_table_extract import rules
from pdf_table_extract.models import Word as _WordModel


def _W(x0, y0, x1, y1, text):
    return _WordModel(x0, y0, x1, y1, text)
from pdf_table_extract.models import (
    ExtractedTable,
    ImageInfo,
    Rect,
    TableCandidate,
    TableCellBox,
)

# ---------------------------------------------------------------------------
# 空白折叠与标号归一化
# ---------------------------------------------------------------------------


def test_collapse_ws_fixes_wrapped_labels():
    """PDF 事实 #1：同一标号会被换行截断成两种写法，折叠后必须一致。

    实测 CCR-18-2728 同时出现 `Supplementary\\nTable S2` 与 `Supplementary Table\\nS2`。
    """
    assert rules.collapse_ws("Supplementary\nTable S2") == "Supplementary Table S2"
    assert rules.collapse_ws("Supplementary Table\nS2") == "Supplementary Table S2"


def test_normalize_label_case_and_variants():
    for raw in ("TABLE I", "Table I", "Tab. I", "table  i"):
        assert rules.normalize_label(raw) == "table i"
    assert rules.normalize_label("Figure 3") == "fig 3"
    assert rules.normalize_label("Fig. 3") == "fig 3"
    assert rules.normalize_label("FIGURE 3") == "fig 3"


def test_normalize_label_folds_plural():
    """复数只出现在正文引用里（"Tables IV and V"），要折回单数才能与 caption 对上。"""
    assert rules.normalize_label("Tables S3") == rules.normalize_label("Table S3")
    assert rules.normalize_label("Figs. 2") == rules.normalize_label("Fig. 2")


def test_normalize_label_folds_redundant_supp_prefix():
    """编号带 S 时 `Supplementary` 前缀冗余 —— 实测 CCR 同篇混用两种写法指同一张表。"""
    assert rules.normalize_label("Supplementary Table S3") == rules.normalize_label("Table S3")
    # 但编号不带 S 时必须保持区分
    assert rules.normalize_label("Supplementary Table 1") != rules.normalize_label("Table 1")


# ---------------------------------------------------------------------------
# caption 判据（PDF 事实 #2）
# ---------------------------------------------------------------------------


def test_caption_vs_citation_pbc_21078():
    """实测基准：pbc_21078 的 `Table I` 在 p2/p3 是引用，`TABLE I` 在 p4/p5 是真 caption。

    刻意不依赖全大写 —— 那只是 Pediatr Blood Cancer 的期刊惯例。
    """
    pat = rules.compile_label_pattern()
    # 真 caption（行首）
    assert rules.match_label_at_line_start("TABLE I. Tumor Demographics", pat) == "TABLE I"
    assert rules.match_label_at_line_start("TABLE I. (Continued)", pat) == "TABLE I"
    # 正文引用（非行首）
    assert (
        rules.match_label_at_line_start(
            "were coded as PD2. Response criteria for the solid tumor panel are in Table I", pat
        )
        is None
    )


def test_caption_allows_line_end_after_label():
    """实测 pbc_28772 / pbc_29304 的 caption 行**只有** `TABLE 1` 三个字，描述在下一行。

    早先要求标号后必须跟 `.:` 或空格，导致这两篇整张表被漏掉。
    """
    pat = rules.compile_label_pattern()
    assert rules.match_label_at_line_start("TABLE 1", pat) == "TABLE 1"


def test_loose_pattern_matches_citations_with_punctuation():
    """引用搜索的结尾条件必须是词边界 —— 正文引用后面常跟逗号/括号/分号。

    早先复用 caption 判据那个 `[.:\\s]|$` 结尾，导致 pbc_21078 的 `Table IV`/`Table V`
    在"只被引用"名单里被漏掉（实测名单应是 Table IV/V/VI 三个）。
    """
    loose = rules.compile_loose_pattern()
    found = rules.find_labels_anywhere(
        "as summarized in Table IV, and in (Table V); see Table VI;", loose
    )
    keys = {rules.normalize_label(x) for x in found}
    assert keys == {"table iv", "table v", "table vi"}


# ---------------------------------------------------------------------------
# 续表判据（PDF 事实 #12）—— 两条都必须有
# ---------------------------------------------------------------------------


def test_caption_says_continued_handles_inner_spaces():
    """实测 pbc_21078 p5 是 `TABLE I. ( Continued )`，括号内带空格。"""
    assert rules.caption_says_continued("TABLE I. ( Continued )")
    assert rules.caption_says_continued("TABLE 1 (Continued)")
    assert not rules.caption_says_continued("TABLE 1 Tumor Demographics")


def test_same_column_names_is_needed_when_caption_empty():
    """实测 pbc_26870 的 Table 1 跨 p18-p19，**p19 caption 是空字符串**，
    唯一线索是两页列名逐字相同（8 列）。只做 (Continued) 判据会漏掉这张续表。
    """
    cols = [
        "Tumor Line",
        "Treatment Group",
        "Median Time to Event",
        "P-value",
        "EFS T/C",
        "Tumor Volume T/C",
        "EFS Activity",
        "Response",
    ]
    assert rules.same_column_names(cols, list(cols))
    assert not rules.same_column_names(cols, cols[:-1])
    assert not rules.same_column_names([], [])


# ---------------------------------------------------------------------------
# 页面结构判据（PDF 事实 #3 #8b）
# ---------------------------------------------------------------------------


def test_rotation_threshold_uses_absolute_count_not_ratio():
    """实测：真横向表竖排 134/150/198/208/264/352/446 行；Wiley 侧边水印**恒为 1 行**。

    必须用绝对行数 —— pbc_21078 p7 是图片页、水平文字仅 7 行，那 1 行水印的占比
    就冲到 12.5%，占比判据会把它误判成横向表。
    """
    assert rules.rotation_for_page(1, (0, 1), has_table_label=True) == 0  # 水印
    # pbc_30017 p7 实测 4 行，正确不转
    assert rules.rotation_for_page(4, (0, -1), has_table_label=True) == 0
    # 真横向表：竖排占比也高（69-99%），且竖排里有 Table 标号，三个条件都过
    for n, h in [(134, 61), (150, 5), (198, 5), (208, 5), (264, 5), (352, 3), (446, 3)]:
        assert rules.rotation_for_page(n, (0, -1), h, has_table_label=True) == 90


def test_rotation_needs_a_table_label_in_the_vertical_text():
    """**第三条判据** —— 78 篇实测发现原来那两条组合起来也不够了（findings F-002/F-012）。

    这些误报的占比全落在真横向表的区间（57-99%）里，两条判据分不开：
      Li et al. 2025 p6        1555 行 / 85%  多面板图 Y 轴标签
      Williams(cells of) p15     90 行 / 67%  染色体臂标签
      5D5JKFJM p17               71 行 / 64%  `©2017...` 版权声明被逐字符拆成 71 行
    它们的共同点是**竖排里没有 Table 标号** —— 横向表的 caption 是跟着表一起转过来的。
    """
    # 占比够高但没标号 → 不转
    assert rules.rotation_for_page(1555, (0, 1), 270, has_table_label=False) == 0   # Li p6
    assert rules.rotation_for_page(90, (0, -1), 44, has_table_label=False) == 0     # Williams p15
    assert rules.rotation_for_page(71, (0, 1), 40, has_table_label=False) == 0      # ©逐字符
    # 有标号 → 转
    assert rules.rotation_for_page(82, (0, -1), 62, has_table_label=True) == 90     # Bruhm p9


def test_rotation_falls_back_to_previous_page_for_continuation_tables():
    """**必需的兜底** —— pbc_26870 p19 是真横向表，但它是续表页、**caption 为空**
    （AGENTS.md 事实 #12 的那个样本），竖排里没有标号。单靠标号判据会误杀它。
    """
    # p18 有标号 → 转
    assert rules.rotation_for_page(208, (0, -1), 5, has_table_label=True) == 90
    # p19 没标号，但上一页转了 → 仍然转
    assert rules.rotation_for_page(150, (0, -1), 5,
                                   has_table_label=False, prev_page_rotated=True) == 90
    # 两个兜底都没有 → 不转
    assert rules.rotation_for_page(150, (0, -1), 5) == 0


def test_rotation_direction_symmetry():
    """(0,-1) → 90 已实测 9/9；(0,1) → 270 **触发过 4 次、4 次全是误报，正例仍然 0 个**。

    见 eval/findings.md F-012 —— AGENTS.md 已知限制 #3 的状态从"未验证"更新为
    「有反例、无正例」。
    """
    assert rules.rotation_for_page(200, (0, -1), has_table_label=True) == 90
    assert rules.rotation_for_page(200, (0, 1), has_table_label=True) == 270


def _img(cover_w: float, cover_h: float) -> ImageInfo:
    return ImageInfo(xref=1, rect=Rect(0, 0, 100, 100), px_width=100, px_height=100,
                     cover_w=cover_w, cover_h=cover_h)


def test_full_page_image_threshold_has_margin():
    """实测两篇扫描件 100%x101% 与 100%x100%；其余 10 篇最大 83%x63%（pbc_29304 p5）。

    阈值 95% 两侧各有余量，不能收窄到 83% 以下否则会误伤正常论文的大图。
    """
    assert rules.has_full_page_image([_img(1.00, 1.01)])
    assert rules.has_full_page_image([_img(1.00, 1.00)])
    assert not rules.has_full_page_image([_img(0.83, 0.63)])  # pbc_29304 p5
    assert not rules.has_full_page_image([_img(0.81, 0.62)])  # pbc_24724 p7
    assert not rules.has_full_page_image([])


def test_metadata_suggests_ocr_is_advisory_only():
    """实测 rstl.1665 的 producer 有 ABBYY（命中）；EAP gastric 的元数据**全空**（漏）。

    所以元数据只能辅助确认，不能反推 —— 结构信号才是主判据。
    """
    assert rules.metadata_suggests_ocr(
        {"producer": "ABBYY Recognition Server; modified using iTextSharp"}
    )
    assert not rules.metadata_suggests_ocr({"producer": "", "creator": ""})
    assert not rules.metadata_suggests_ocr({"producer": "Acrobat Distiller 10.1.10 (Windows)"})


def test_dispatch_source_type():
    """实测三张位图框内文字层字符**全是 0**，而所在页有 1480-4202 字符。"""
    assert rules.dispatch_source_type(0) == "image"
    assert rules.dispatch_source_type(29) == "image"
    assert rules.dispatch_source_type(30) == "text"
    assert rules.dispatch_source_type(3190) == "text"  # pbc_28772 p3 矢量表


# ---------------------------------------------------------------------------
# 审计
# ---------------------------------------------------------------------------


def _table(rows: list[list[str]], source_type: str = "text") -> ExtractedTable:
    cand = TableCandidate(page=1, rect=Rect(0, 0, 100, 100), source_type=source_type)
    return ExtractedTable(candidate=cand, rows=rows, extractor="docling")


def test_coverage_catches_structural_collapse():
    """实测 pbc_29304 p4 未转正时 docling 给 1x2（约 30 字符），该页有 1308 字符 → ~2%。"""
    assert rules.coverage_ratio(30, 1308) < rules.COVERAGE_LOW
    assert rules.coverage_ratio(1200, 1308) > rules.COVERAGE_LOW
    assert rules.coverage_ratio(10, 0) is None


def test_ocr_page_forces_low_confidence():
    """PDF 事实 #8b：出版社 OCR 文字层页 —— 结构可能是对的，但字符是 OCR 产物。

    实测 EAP gastric p2：docling 从字符坐标正确恢复出 21x3 结构，
    但字符里有 `Cordia`（应为 Cardia）、`et al'`（句点→撇号）。
    """
    t = _table([["a", "b"], ["1", "2"], ["3", "4"]])
    t.coverage = 0.9
    t.plumber_cols = 2
    assert rules.decide_confidence(t, ocr_page=True) == "low"
    assert rules.decide_confidence(t, ocr_page=False) == "high"


def test_image_source_always_low():
    """铁律 #4：任何 OCR 派生结果无条件 low，不看质量。"""
    t = _table([["a", "b"]], source_type="image")
    t.coverage = 0.99
    t.plumber_cols = 2
    assert rules.decide_confidence(t, ocr_page=False) == "low"


def test_suspect_lt_as_minus():
    """PDF 事实 #11：OCR 把 `<0.001` 读成 `-0.001` —— 看起来完全合法的数值。

    判据是**列内字符集一致性**，不含任何领域知识（不需要知道那列是 p 值）。
    """
    rows = [
        ["Xenograft", "P-value"],
        ["BT-29", "<0.001"],
        ["KT-11", "<0.001"],
        ["KT-13", "-0.001"],  # ← OCR 误读
        ["EW5", "<0.001"],
    ]
    hints = rules.suspect_lt_as_minus(rows)
    assert len(hints) == 1
    assert "R3C1" in hints[0]


def test_suspect_lt_as_minus_no_false_positive_on_normal_negatives():
    """一列里本来就都是负数时不该报警。"""
    rows = [["m", "delta"], ["a", "-1.2"], ["b", "-3.4"], ["c", "-0.5"]]
    assert rules.suspect_lt_as_minus(rows) == []


# ---------------------------------------------------------------------------
# caption 归属校验（铁律 #1）
# ---------------------------------------------------------------------------


def test_verify_caption_belongs_catches_mismatch():
    """前身项目最痛的 bug：Table S1 的网格被错配给 Table 2 且被无条件信任。

    注意判据是相对比较 —— 单独一个"零重叠"的 caption 不算反证，
    必须有另一个 caption 与该网格重叠更多。详见 test_verify_caption_needs_positive_counter_evidence。
    """
    rows = [["Xenograft line", "Histology", "P-value"], ["BT-29", "Rhabdoid", "<0.001"]]
    # 自己有重叠 → 通过
    assert rules.verify_caption_belongs("TABLE I. Tumor Demographics Xenograft", rows)
    # 自己零重叠，且另一个 caption 明显更像这个网格 → 报错配
    assert not rules.verify_caption_belongs(
        "TABLE 5. Pharmacokinetic parameters plasma clearance",
        rows,
        other_captions=["TABLE I. Xenograft histology"],
    )


def test_verify_caption_belongs_tolerates_empty_caption():
    """续表 caption 常为空（实测 pbc_26870 p19），此时无从校验、不能因此判错。"""
    assert rules.verify_caption_belongs("", [["a"], ["b"]])


# ---------------------------------------------------------------------------
# 同句共现（label_referenced_but_absent）
# ---------------------------------------------------------------------------


def test_split_sentences_protects_abbreviations_and_decimals():
    text = "See Fig. 1 for details. Response was 0.05 in Table S2. Next sentence here."
    sents = rules.split_sentences(text)
    assert len(sents) == 3
    assert "Fig. 1" in sents[0]
    assert "0.05" in sents[1]


def test_cooccurs_in_sentence_is_sentence_scoped():
    """刻意限制到同句：整段共现会连出大量假关联。"""
    text = (
        "Complete response rates are summarized in Supplementary Table S2. "
        "Toxicity data appear in Supplementary Table S5."
    )
    assert rules.cooccurs_in_sentence(text, "Supplementary Table S2", ["response"]) == ["response"]
    assert rules.cooccurs_in_sentence(text, "Supplementary Table S5", ["response"]) == []


# ---------------------------------------------------------------------------
# 关键词匹配（models.ExtractedTable.matches）
# ---------------------------------------------------------------------------


def _labeled(caption: str) -> ExtractedTable:
    from pdf_table_extract.models import Label

    lab = Label(raw="TABLE II", key="table ii", kind="table", page=6, is_caption=True, text=caption)
    cand = TableCandidate(page=6, rect=Rect(0, 0, 1, 1), source_type="text", label=lab)
    return ExtractedTable(candidate=cand, rows=[["a"]], extractor="docling")


def test_matches_or_default_and_case_insensitive():
    """实测基准：`response` 应命中 pbc_21078 的 `TABLE II. Vincristine Response Data`。"""
    t = _labeled("TABLE II. Vincristine Response Data")
    assert rules.match_table(t, ["response"])[0] == ["response"]
    assert rules.match_table(t, ["RESPONSE"])[0] == ["RESPONSE"]
    assert rules.match_table(t, ["xenograft"])[0] == []
    assert rules.match_table(t, ["xenograft", "response"])[0] == ["response"]  # OR


def test_matches_all_requires_every_keyword():
    t = _labeled("TABLE II. Vincristine Response Data")
    assert rules.match_table(t, ["response", "xenograft"], match_all=True)[0] == []
    assert rules.match_table(t, ["response", "vincristine"], match_all=True)[0] == [
        "response", "vincristine"
    ]


# ---------------------------------------------------------------------------
# 图片路径的免费第二意见（单元格数对账）
# ---------------------------------------------------------------------------


def test_grid_shape_mismatch_catches_clipped_column():
    """实测教训：pbc_24724 Fig.1 还原成 46x6，而正确形状是 45x7 —— 少了 Heat Map 一列。

    **比总格子数分不开**：检测器给 300 格，276 与 315 相对 300 分别差 8% 与 5%，
    任何阈值都无法区分。所以改用检测器的列带/行带数直接比行列（6 vs 7 一目了然）。
    """
    assert rules.grid_shape_mismatch(45, 7, 46, 6) is not None  # 少一列 → 报警
    assert "列" in rules.grid_shape_mismatch(45, 7, 46, 6)
    assert rules.grid_shape_mismatch(45, 7, 45, 7) is None  # 完全一致
    assert rules.grid_shape_mismatch(45, 7, 46, 7) is None  # 差 1 在容差内
    assert rules.grid_shape_mismatch(0, 0, 45, 7) is None  # 没检出就不评判


def test_grid_shape_mismatch_catches_lost_rows():
    """也能抓丢行 —— 实测那次还原丢了第一行数据（BT-29 整行不见）。"""
    msg = rules.grid_shape_mismatch(45, 7, 30, 7)
    assert msg is not None and "行" in msg


def test_verify_caption_needs_positive_counter_evidence():
    """判据是**相对比较**，不是绝对阈值。

    实测误报：早先要求"caption 实词必须与表头/首列重叠"，把两处正常情况判成了错配 ——
      `TABLE I. Tumor Demographics`      表头是 `Xenograft line | Histology | ...`
      `Table 1. Patient Characteristics` 表头是 `'' | Preusser et al | Taal et al`
    caption 描述主题、表头列变量名，本来就常常不重叠。缺少证据不等于有反证。
    """
    rows = [["Xenograft line", "Histology", "P-value"], ["BT-29", "Rhabdoid", "<0.001"]]
    # 零重叠但没有别的 caption 更像 → 不该报警
    assert rules.verify_caption_belongs("TABLE I. Tumor Demographics", rows)
    # 有另一个 caption 与网格重叠更多 → 这才是正面反证
    assert not rules.verify_caption_belongs(
        "TABLE 5. Pharmacokinetic parameters",
        rows,
        other_captions=["TABLE 1. Xenograft histology panel"],
    )
    # 本 caption 自己就有重叠 → 直接通过
    assert rules.verify_caption_belongs(
        "TABLE 1. Xenograft lines by histology",
        rows,
        other_captions=["TABLE 5. Pharmacokinetics"],
    )


def test_decorative_image_threshold_from_measurements():
    """实测：logo 页高占比 3.9%/1.9%（pbc_28772）、3.5%（CCR）；
    真实内容最小是 blood p7 多面板显微照片 13.2%，含表格的图 39.5%。
    """
    assert rules.is_decorative_image(0.039)
    assert rules.is_decorative_image(0.019)
    assert rules.is_decorative_image(0.035)
    assert not rules.is_decorative_image(0.132)
    assert not rules.is_decorative_image(0.395)


def test_rival_overlap_needs_two_words():
    """竞争 caption 至少要重叠 2 个实词才算反证 —— 1 个词会被偶然碰巧命中。

    实测踩坑：pbc_21078 的 Fig.1 图注列举了 EW5/SK-NEP-1/Rh28/KT-13 这些移植瘤系名，
    正是 TABLE I 的首列内容。图注天然会列举表格的行标识，所以
    (a) 只能拿**同类** caption 当竞争者，(b) 还要求至少 2 个词。
    """
    rows = [["Xenograft", "Panel"], ["BT-29", "Kidney"], ["KT-13", "Wilms"]]
    # 只碰巧重叠 1 个词 → 不算反证
    assert rules.verify_caption_belongs(
        "TABLE I. Tumor Demographics", rows, other_captions=["TABLE 9. Outcomes by panel"]
    )
    # 重叠 2 个词 → 算反证
    assert not rules.verify_caption_belongs(
        "TABLE I. Tumor Demographics",
        rows,
        other_captions=["TABLE 9. Xenograft panel composition"],
    )


# ---------------------------------------------------------------------------
# 输出目录（一个 PDF 一个文件夹）
# ---------------------------------------------------------------------------


def test_sanitize_name_handles_real_filenames():
    """实测要处理的：`EAP in advanced gastric cancer..pdf` 的 stem 带空格、带尾点。"""
    from pdf_table_extract import emit

    assert emit.sanitize_name("EAP in advanced gastric cancer.") == "EAP_in_advanced_gastric_cancer"
    assert emit.sanitize_name("pbc_24724") == "pbc_24724"
    assert emit.sanitize_name("1078-0432.CCR-18-2728") == "1078-0432.CCR-18-2728"
    assert emit.sanitize_name("  ") == "output"


def test_resolve_outdir_one_folder_per_pdf():
    from pathlib import Path

    from pdf_table_extract import emit

    pdf = Path("0.pdf_input/pbc_24724.pdf")
    # 都不给 → ./<pdf名>/
    assert emit.resolve_outdir(None, None, pdf) == Path("pbc_24724")
    # 只给 outdir → <outdir>/<pdf名>/
    assert emit.resolve_outdir(Path("/tmp/out"), None, pdf) == Path("/tmp/out/pbc_24724")
    # 给 prefix → 用 prefix 当子文件夹名
    assert emit.resolve_outdir(Path("/tmp/out"), "run1", pdf) == Path("/tmp/out/run1")


def test_text_table_matches_content_not_only_caption():
    """实测基准：pbc_28772 的金标准表 caption 不含 `response`，但第 11 列表头是 `Obj. Response`。

    只匹配 caption+legend 会漏掉这张明显该命中的表。文字表的内容在过滤时
    **已经在内存里**（docling 整篇跑一次、过滤在抽取之后），所以算进来不花钱。
    """
    from pdf_table_extract.models import Label

    lab = Label(
        raw="TABLE 1", key="table 1", kind="table", page=3, is_caption=True,
        text="TABLE 1 EPZ011989 as a single agent and in combination for all PDX models",
    )
    cand = TableCandidate(page=3, rect=Rect(0, 0, 1, 1), source_type="text", label=lab)
    t = ExtractedTable(
        candidate=cand,
        rows=[["Model", "Agent", "KMmed (days)", "Obj. Response"], ["G401", "Control", "9.5", "PD"]],
        extractor="docling",
    )
    hits, where = rules.match_table(t, ["response"])
    assert hits == ["response"]
    assert where == "cell"          # caption 里没有，靠表内命中
    assert rules.match_table(t, ["EPZ011989"])[1] == "caption"
    # 图片表要显式关掉内容匹配（内容得付 OCR 代价）
    assert rules.match_table(t, ["response"], include_content=False)[0] == []


# ---------------------------------------------------------------------------
# 关键词匹配语义：全词 + 屈折变体（默认），子串（--substring）
# ---------------------------------------------------------------------------


def test_short_keywords_must_not_match_inside_words():
    """实测噪声：子串模式下 `PR` 命中 130 处、`CR` 41 处，全在英文单词内部。

    `CCR-18-2728` 那三张图（该篇没有响应码表）在子串模式下会被
    `response+(CR|PR|PD)` 全部误命中；全词模式下必须全部排除。
    """
    cases = [
        ("Prexasertib-mediated CHK1 inhibition", "PR"),
        ("with increasing concentrations", "CR"),
        ("response to vincristine", "CR"),
        ("cells transduced with AKR1C3", "SD"),
        ("(DSRCT) PDX models", "PD"),
        ("A representative image", "PR"),
        ("C, SJCRH30 alveolar", "CR"),
    ]
    for text, kw in cases:
        assert not rules.keyword_regex(kw).search(text), f"{kw!r} 不该命中 {text!r}"
        # 子串模式下这些**会**命中 —— 这正是默认不能用子串的原因
        assert rules.keyword_regex(kw, substring=True).search(text)


def test_inflection_variants_must_match():
    """纯全词会漏（实测）：`Demographic` 全词 0 处（原文是 `Tumor Demographics`）、
    `xenograft` 全词 11 处 vs 带变体 38 处。所以屈折变体是必需的，不是可选。
    """
    cases = [
        ("Tumor Demographics", "Demographic"),      # +s
        ("ALL xenografts", "xenograft"),            # +s
        ("Responses of rhabdomyosarcoma", "response"),
        ("Obj. Response", "response"),
        ("controlled release", "control"),          # 末辅音双写
        ("controlling for age", "control"),
        ("dosing schedule", "dose"),                # e → ing
        ("toxicities observed", "toxicity"),        # y → ies
        ("objective activities", "activity"),       # y → ies
        ("treated animals", "treat"),               # +ed
    ]
    for text, kw in cases:
        assert rules.keyword_regex(kw).search(text), f"{kw!r} 应命中 {text!r}"


def test_numeric_codes_get_no_morphology():
    """`PD1` / `S2` 是编号代号，没有屈折 —— 不该展开成 `PD1s`/`PD1ing`。"""
    assert rules._morph_variants("PD1") == ["pd1"]
    assert rules.keyword_regex("PD1").search("code PD1 here")
    assert not rules.keyword_regex("PD1").search("PD10")


def test_multiword_phrase_inflects_last_word_only():
    r = rules.keyword_regex("overall response")
    assert r.search("Overall Group Response")  is None  # 中间插了词，不该命中
    assert r.search("the overall responses were")        # 末词复数应命中
    assert r.search("Overall Response")


def test_hyphen_counts_as_word_char_not_boundary():
    """连字符按**词字符**处理 —— `PR-104` 不满足 `-k PR`。

    这一条最初选反了，由实测数据纠正：语料里所有连字符 token 全是标识符 ——
      `PR-104`/`PR-104A` 药名（blood 整篇讲这个药）、`NB-SD` 移植瘤系名、
      `ALL-11`/`KT-13`/`CTG-0241`/`JNJ-26481585` 编号代号。
    而"值得抓的连字符复合词"（`dose-response`）在语料里一个都没有（原文是空格写法，
    且曲线图本来就不是要的表）。所以这是零召回损失、消掉一整类误报。
    """
    assert not rules.keyword_regex("PR").search("PR-104A treatment")
    assert not rules.keyword_regex("SD").search("NB-SD xenograft")
    assert not rules.keyword_regex("PD").search("PD-L1 expression")
    assert not rules.keyword_regex("ALL").search("ALL-11 engrafted")
    # 空格分隔的正常写法照常命中
    assert rules.keyword_regex("response").search("Dose response curves")
    assert rules.keyword_regex("survival").search("Event-free survival")


# ---------------------------------------------------------------------------
# legend 跨 block 续抓（AGENTS.md PDF 事实：legend 常常不在 caption 那个 block 里）
# ---------------------------------------------------------------------------


def test_legend_continuation_needs_both_gap_and_line_length():
    """区分「legend 续段」与「表体」必须同时看间距和行长 —— 各自单独用都会错。

    实测的四个案例（数值可回查）：
      pbc_30017 b3  legend 续段    间距 8.7   最长行 107  → 吸收
      pbc_30017 b4  表体 Cancer Type 间距 23.0  最长行 11   → 靠间距停
      blood p6  b6  表体 Treatment  间距 4.2 ← **很小**  最长行 14 → **只看间距会误吸**，靠行长停
      pbc_24724 p5 b4 页脚          间距 291.8 最长行 36  → 靠间距停
    """
    assert rules.is_legend_continuation(8.7, 107)      # pbc_30017 legend 续段
    assert not rules.is_legend_continuation(23.0, 11)  # pbc_30017 表体
    assert not rules.is_legend_continuation(4.2, 14)   # blood 表体：间距小但行短
    assert not rules.is_legend_continuation(291.8, 36) # pbc_24724 页脚：行够长但太远


def test_upscale_factor_targets_300dpi():
    """低分辨率图在还原前放大到 ~300 有效 dpi（实测 pbc_21296 是 120dpi）。"""
    from pdf_table_extract import engine_paddle as ep

    assert ep.upscale_factor(120) == 3   # pbc_21296 Fig.1
    assert ep.upscale_factor(150) == 2   # CCR Figure 1
    assert ep.upscale_factor(300) == 1   # pbc_24724 Fig.1，本来就够，不放大
    assert ep.upscale_factor(600) == 1
    assert ep.upscale_factor(50) == ep.UPSCALE_MAX  # 封顶
    assert ep.upscale_factor(0) == 1


def test_scanned_document_is_a_whole_doc_property_not_per_page():
    """**判据从「单页字数」改成「整篇整页位图占比」** —— 90 篇实测推翻了原来的逐页判据。

    原判据（整页位图 且 该页 >= 1000 字）**两类错同时存在，而且顺序是反的**：
      真扫描页   Dobzhansky_1946 p12=602 / Rous_1911 p21=681 / Bell_1876 p3=832 → **漏判**
      现代大图页 Chunduri_2022 p7=1008 / Drapkin_2018 p1=1122                   → **误报**
    真扫描页的字比误报页更少，所以调高加重漏判、调低加重误报 —— 字数分不开这两类。

    换成「整篇有多大比例的页是整页位图」就完美分离：
      真扫描件 21 篇（1876-1995）  97%-100%
      现代论文含大图 5 篇            6%- 21%（含 log.md §8 的 Beddows 21%）
    """
    # 真扫描件：每页都是整页位图，且整篇字够多
    assert rules.is_scanned_document(n_full_page_images=3, n_pages=3, total_chars=3 * 5255)
    assert rules.is_scanned_document(n_full_page_images=32, n_pages=33, total_chars=33 * 2337)
    # Bell_1876：每页平均 1128 字，是真扫描件里最低的，必须过
    assert rules.is_scanned_document(n_full_page_images=3, n_pages=3, total_chars=3 * 1128)
    # 现代论文只有 1-2 页大图 —— Beddows 6/29=21%、Drapkin 1/16=6%
    assert not rules.is_scanned_document(n_full_page_images=6, n_pages=29, total_chars=29 * 4000)
    assert not rules.is_scanned_document(n_full_page_images=1, n_pages=16, total_chars=16 * 4000)
    # 整本扫描但**根本没做 OCR** —— 必须走 OCR 路径，误判会让它跳过图片候选、整篇丢光
    assert not rules.is_scanned_document(n_full_page_images=10, n_pages=10, total_chars=50)
    assert not rules.is_scanned_document(n_full_page_images=0, n_pages=0, total_chars=0)


def test_ocr_text_layer_needs_both_doc_scanned_and_page_covered():
    """逐页那一层只剩「这一页是不是被位图覆盖」—— 扫描件里偶有非整页位图的页。

    实测 `Triolo_1965` 是 32/33 页被覆盖，那 1 页不算 OCR 文字层。
    """
    full, partial = [_img(1.00, 1.00)], [_img(0.83, 0.63)]
    assert rules.is_ocr_text_layer(full, doc_is_scanned=True)
    assert not rules.is_ocr_text_layer(partial, doc_is_scanned=True)   # 扫描件里的非位图页
    assert not rules.is_ocr_text_layer(full, doc_is_scanned=False)     # 现代论文的整版大图页


def test_rotation_needs_both_absolute_count_and_ratio():
    """**留出集抓出的泛化 bug**：只用绝对行数会把多面板图的 Y 轴标签误判成横向表。

    三档实测数据（我原先只见过第一、三档，所以只用了绝对行数）：
      Wiley 侧边下载水印         竖排   1 行            → 绝对行数挡住
      Becker 2022 多面板图轴标签 竖排 41-79 行、占比 17-25% → **只用绝对行数会误判**，占比挡住
                                （`UMAP dimension 2` / `log2FC (36,374 peaks)` / `PC2 (7.0%)`）
      真横向表                   竖排 134-446 行、占比 69-99% → 两个都过
      pbc_21078 p7 图片页        竖排 1 行、占比 12.5%     → **只用占比会误判**，绝对行数挡住

    注：后来又加了第三条（竖排里含 Table 标号），见 test_rotation_needs_a_table_label...。
    本用例只测前两条，所以统一给 has_table_label=True 把第三条让开。
    """
    kw = dict(has_table_label=True)
    assert rules.rotation_for_page(79, (0, -1), 233, **kw) == 0   # Becker p4：占比 25%
    assert rules.rotation_for_page(41, (0, -1), 197, **kw) == 0   # Becker p6：占比 17%
    assert rules.rotation_for_page(134, (0, -1), 61, **kw) == 90  # 真横向表：占比 69%
    assert rules.rotation_for_page(352, (0, -1), 3, **kw) == 90   # 真横向表：占比 99%
    assert rules.rotation_for_page(1, (0, -1), 7, **kw) == 0      # 图片页水印：占比 12.5%


def test_preprint_two_parallel_criteria_page_ratio_and_p1_watermark():
    """两条**并列**判据，因为两类预印本的水印形态根本不同。

    实测 4 篇 arXiv 论文的「含水印词页数占比」：
      Perrone 2020（用 `A PREPRINT` 页眉模板）  90% → 占比判据抓到
      Li_2013 (BWA-MEM)                          33% → 漏
      McInnes_2020 (UMAP)                        10% → 漏
      2607.00042v1（天文，SKA）                    0% → 漏
    那篇天文论文 8/27 页含 `arXiv`，其中 **1 页是真水印、7 页是参考文献引用** ——
    噪声是信号的 7 倍，加词表只会更糟（log.md §8 的 Baslan 坑）。

    结构判据（第 1 页最左 8%、竖排、含「平台名+编号」）实测 90 篇零误伤、4 篇 arXiv 全中。
    """
    from pdf_table_extract import pdfio
    from pdf_table_extract.models import DocInfo

    def doc(pages: int, hit_pages: int, watermark: str = "") -> DocInfo:
        return DocInfo(path="x", n_pages=pages, pages=[], metadata={},
                       preprint_pages=hit_pages, p1_watermark=watermark)

    # ① 页数占比：566455 是 54/64=84%，Baslan 2022 正式版是 2/33=6%
    assert pdfio.is_preprint(doc(64, 54))
    assert not pdfio.is_preprint(doc(33, 2))
    # ② 首页竖排水印：占比再低也算 —— 天文那篇是 0/27
    assert pdfio.is_preprint(doc(27, 0, "arXiv:2607.00042v1  [astro-ph.HE]  29 Jun 2026"))
    assert pdfio.is_preprint(doc(3, 1, "arXiv:1303.3997v2  [q-bio.GN]  26 May 2013"))
    # 两条都不满足才放行
    assert not pdfio.is_preprint(doc(27, 7))       # 7/27=26%，且无首页水印
    assert not pdfio.is_preprint(doc(0, 0))


def test_p1_watermark_regex_needs_platform_and_number():
    """光有平台名不够 —— 参考文献里 `arXiv e-prints` 这种到处都是，必须带编号。"""
    from pdf_table_extract import pdfio

    ok = pdfio.P1_WATERMARK_RE
    assert ok.search("arXiv:2607.00042v1  [astro-ph.HE]  29 Jun 2026")
    assert ok.search("arXiv:1802.03426v3  [stat.ML]  18 Sep 2020")
    assert ok.search("arXiv:2006.04951v1  [cs.SI]  2 Jun 2020")
    assert not ok.search("arXiv e-prints, art. arXiv preprint")       # 参考文献里的引用
    assert not ok.search("Downloaded from http://aacrjournals.org")   # 出版社下载水印
    assert not ok.search("arXiv")                                     # 光有平台名不算


def test_strip_glyph_names_keeps_the_content():
    """docling 把 PDF 未映射的字形名原样写进单元格 —— 实测 `Medina 2025` 有 27 处 `/emspace*`。

    **第一版写成 `^/[a-zA-Z]{2,15}` 贪婪匹配，把 `/emspaceMean` 整格吃成空串。**
    而留出集里有 24 格是 `/emspaceClear cell`、`/emspaceHigh-grade serous` 这类
    卵巢癌组织学分型 —— 那样会**丢数据**。所以只认已知的空白字形名、保留后面的内容。
    """
    rows = [
        ["/emspaceMean", "/emspaceRange", "/emspaceI"],
        ["/emspaceClear cell", "/emspaceHigh-grade serous", "/emspaceCystadenoma/adenofibroma"],
        ["/bowtie", "Mean", ""],           # /bowtie 不是空白字形名，不动
    ]
    out = rules.strip_glyph_names([r[:] for r in rows])
    assert out[0] == ["Mean", "Range", "I"]
    assert out[1] == ["Clear cell", "High-grade serous", "Cystadenoma/adenofibroma"]
    assert out[2] == ["/bowtie", "Mean", ""]
    assert all(c.strip() for r in out[:2] for c in r), "一个格子都不许被清空"


# ---------------------------------------------------------------------------
# 断行连字符（F-017）—— rejoin_hyphen_breaks
# ---------------------------------------------------------------------------
#
# 全部 case 来自三个语料的实测（log.md §30.2 / §32）。
# 第一版只用正则，5 种命中里 3 种改坏；第二版词表建错（先去了连字符）导致循环论证；
# 第三版门槛设成"出现 >=1 次"又被 channelaware 骗过。下面每个 case 钉一种错法。


class TestRejoinHyphenBreaks:
    def test_joins_real_line_break(self):
        """pbc_28772 p3/p4 `Agent` 列：`EPZ011989 + cyclophos- phamide`。

        这 5 行的 `Agent` 是 gold 的锚点列，不接回就整整 5 行对不上（F-017）。
        """
        rows = [["Agent"], ["EPZ011989 + cyclophos- phamide"]]
        out, n = rules.rejoin_hyphen_breaks(rows, {"cyclophosphamide": 10})
        assert out[1][0] == "EPZ011989 + cyclophosphamide"
        assert n == 1

    def test_does_not_join_legit_compound_words(self):
        """留出集 Chen 2024 与阈值验证集实测的三个假阳性 —— 纯正则版全部会改坏。

        `lineage-defining` / `patient-derived` 是合法连字符复合词，
        `p-value` 的连字符也该保留。它们拼起来都不是真词，所以词表里查不到。
        """
        rows = [
            ["Controls enhancer acetylation of lineage- defining core regulatory"],
            ["Cell lines, patient- derived neurosphere model"],
            ["Log-rank p- vaiue"],
        ]
        out, n = rules.rejoin_hyphen_breaks(rows, {"lineage": 9, "defining": 4, "value": 7})
        assert n == 0
        assert out == rows, "查不到就该一个字不动"

    def test_does_not_break_urls(self):
        """阈值验证集实测：纯正则版会破坏 URL。

        `…childhood-cancer- statistics/`（Chen 2024 p17）、
        `bcl2fastq- conversion-software`（Abeshouse 2017 p19）。
        """
        rows = [["https://x.org/childhood-cancer- statistics/"],
                ["…/bcl2fastq- conversion-software/index.html"]]
        out, n = rules.rejoin_hyphen_breaks(rows, {"statistics": 5, "conversion": 8})
        assert n == 0

    def test_single_occurrence_is_not_enough(self):
        """de_Bruijne_2021：`Channel- aware` 被误接成 `Channelaware`。

        原因是 `channelaware` 在那本 800+ 页论文集里**恰好出现 1 次** ——
        而那 1 次本身就是别处同样断行的产物。所以门槛必须是 >=2，不能是 >=1。
        对照：该接的 11 处里最低的 `electrophoresis` 是 2 次。
        """
        rows = [["Channel- aware?"]]
        assert rules.rejoin_hyphen_breaks(rows, {"channelaware": 1})[1] == 0
        assert rules.rejoin_hyphen_breaks(rows, {"channelaware": 2})[1] == 1

    def test_returns_new_list_not_mutated(self):
        """**必须返回新 list。**

        `__main__.run_text_path` 里 `ExtractedTable.rows` 与 `DoclingResult.tables[i].rows`
        是同一个对象（`stitch` 会就地 extend 它），就地改会污染 dl。
        """
        rows = [["a", "cyclophos- phamide"]]
        out, _ = rules.rejoin_hyphen_breaks(rows, {"cyclophosphamide": 10})
        assert rows[0][1] == "cyclophos- phamide", "原 rows 不许被改"
        assert out is not rows and out[0] is not rows[0]

    def test_has_hyphen_break_prefilter(self):
        """预筛用来避免为每篇文档白算一次整篇词表。"""
        assert rules.has_hyphen_break([["cyclophos- phamide"]])
        assert not rules.has_hyphen_break([["Cyclophosphamide"], ["PR-104"], ["NB-SD"]])

    def test_identifier_hyphens_are_untouched(self):
        """AGENTS.md 已知限制 #5：本语料的连字符 token 全是标识符。

        判据靠的是**连字符后面有没有空白** —— `PR-104` / `NB-SD` / `ALL-11` 都没有，
        所以连正则都匹配不上，不依赖词表。
        """
        rows = [["PR-104"], ["NB-SD"], ["ALL-11"], ["KT-13"], ["PD-L1"]]
        out, n = rules.rejoin_hyphen_breaks(rows, {})
        assert n == 0 and out == rows


# ---------------------------------------------------------------------------
# 压行检测（F-001）—— 四层判据各自独立可测
# ---------------------------------------------------------------------------


def _cell(text, r0, c0, y0, y1, *, hdr=False, rspan=1, x0=0.0, x1=50.0):
    return TableCellBox(text=text, r0=r0, c0=c0, row_span=rspan, col_span=1,
                        column_header=hdr, bbox=Rect(x0, y0, x1, y1))


def _even_col(c0, n=8, h=10.0, tall_row=None, tall_h=25.0):
    """造一列高度整齐的格子；`tall_row` 那一行给 tall_h。"""
    out = []
    y = 0.0
    for r in range(1, n + 1):
        hh = tall_h if r == tall_row else h
        out.append(_cell(f"v{r}", r, c0, y, y + hh))
        y += hh
    return out


class TestSuspiciousTallRows:
    def test_two_tall_columns_in_even_table_is_flagged(self):
        cells = _even_col(0, tall_row=4) + _even_col(1, tall_row=4)
        assert 4 in rules.suspicious_tall_rows(cells)

    def test_one_tall_column_is_not_enough(self):
        """只有一列偏高 = 长标签折行，不是压行。这是 4 个已知误报的机制。"""
        cells = _even_col(0, tall_row=4) + _even_col(1)
        assert rules.suspicious_tall_rows(cells) == {}

    def test_uneven_column_is_not_evidence(self):
        """行高天然参差的列（综述表每格都是长句）里，'偏高'这个信号没有意义。

        不加这道闸门时留出集 27 张表命中 28 次、拆分 14 次**全错**。
        """
        cells = []
        for c0 in (0, 1):
            y = 0.0
            for r, hh in enumerate([10, 40, 12, 60, 11, 55, 13, 90], start=1):
                cells.append(_cell(f"v{r}", r, c0, y, y + hh))
                y += hh
        assert rules.suspicious_tall_rows(cells) == {}

    def test_header_rows_excluded_by_docling_flag(self):
        """多级表头的第 2/3 行会被 docling 拼平、行高偏高 —— 必须靠 column_header 排除。

        实测 90 篇里 9 处误报全是这个，且表头占 1/2/3 行的都有
        （`Ng 2017` / `Nowell 1976` 是 3 行）—— 所以不能写死跳过前 N 行。
        """
        cells = _even_col(0, tall_row=2) + _even_col(1, tall_row=2)
        assert 2 in rules.suspicious_tall_rows(cells), "先确认不加表头标记时会命中"
        flagged = [_cell(c.text, c.r0, c.c0, c.bbox.y0, c.bbox.y1, hdr=(c.r0 <= 2))
                   for c in cells]
        assert rules.suspicious_tall_rows(flagged) == {}

    def test_row_span_cells_ignored(self):
        """纵跨多行的格子当然更高，那是合法合并单元格（PDF 事实 #15），不是压行。"""
        cells = _even_col(0) + _even_col(1)
        cells += [_cell("span", 4, 2, 0.0, 90.0, rspan=3),
                  _cell("span", 4, 3, 0.0, 90.0, rspan=3)]
        assert rules.suspicious_tall_rows(cells) == {}

    def test_too_few_samples_gives_up(self):
        """同列样本太少 ⇒ 中位数不可信 ⇒ 放弃（保守：不报）。"""
        cells = _even_col(0, n=3, tall_row=2) + _even_col(1, n=3, tall_row=2)
        assert rules.suspicious_tall_rows(cells) == {}


class TestConfirmMergedRow:
    def test_wrapped_label_only_one_column_has_two_bands(self):
        """长标签折行：只有标签列有两带 ⇒ 不算压行。"""
        cells = [_cell("Image size (pixels)", 3, 0, 0, 24, x0=0, x1=50),
                 _cell("2455", 3, 1, 0, 24, x0=60, x1=100)]
        words = [_W(5, 2, 40, 10, "Image"), _W(5, 14, 40, 22, "size"),
                 _W(65, 6, 95, 16, "2455")]
        assert rules.confirm_merged_row(cells, 3, words) == 1

    def test_real_merged_row_has_many_columns_with_two_bands(self):
        cells = [_cell("ALL-8 ALL-16", 3, 0, 0, 24, x0=0, x1=50),
                 _cell("a b", 3, 1, 0, 24, x0=60, x1=100),
                 _cell("c d", 3, 2, 0, 24, x0=110, x1=150)]
        words = []
        for x0, x1 in ((5, 40), (65, 95), (115, 145)):
            words += [_W(x0, 2, x1, 10, "u"), _W(x0, 14, x1, 22, "v")]
        assert rules.confirm_merged_row(cells, 3, words) == 3


class TestMatchCellRowToOutputRow:
    def test_matches_by_content_not_index(self):
        """两套视图行号差 1（多级表头被 dataframe 拼平）—— 必须按内容配对。

        实测 184/427 张表两者行数不一致；按行号找会报到无辜的一行上。
        """
        cells = [_cell("ALL-8 ALL-16", 41, 0, 0, 24), _cell("> EP > EP", 41, 1, 0, 24)]
        rows = [["h1", "h2"]] + [[f"x{i}", f"y{i}"] for i in range(1, 40)] \
            + [["ALL-8 ALL-16", "> EP > EP"]]
        i = rules.match_cell_row_to_output_row(cells, 41, rows)
        assert i == 40, f"内容在第 40 行，不是格子的行号 41；实际 {i}"

    def test_no_overlap_returns_none(self):
        cells = [_cell("zzz", 5, 0, 0, 24)]
        assert rules.match_cell_row_to_output_row(cells, 5, [["a"], ["b"]]) is None
