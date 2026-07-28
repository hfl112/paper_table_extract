"""eval 层纯逻辑函数的单测。

`eval/checks.py` 与 `eval/gold.py` 现在是产出 `eval/findings.md` 的承重件 ——
改坏了清单会静默变样。这里只测纯函数（不碰 PDF、不联网），符合 tests/ 与 eval/ 的分工。

**每个 case 的值都来自真实文献的实测**，出处写在各自的 docstring 里。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "eval" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 判据住在 eval/checks.py（detect 与 regression 合并后的文件），gold 住在 eval/gold.py
detect = _load("checks")
gold_io = _load("gold")


# ————————————————————— detect.norm —————————————————————

class TestNorm:
    def test_nfkc_splits_ligature(self):
        """pbc_21078 p6 文字层里是 `ﬁnal RTV`（U+FB01），docling 抽出的是 `final RTV`。

        不拆连字，`Median final RTV` 在 3 张表上都对不上 —— 一度被误判成上标脚注问题。
        """
        assert detect.norm("ﬁnal RTV") == detect.norm("final RTV")

    def test_drops_control_chars(self):
        """pbc_21078 table_ii/table_iii 各有 14 个 `\\x01`（符号字体的对钩类字形）。

        文字层里不以该码位出现，不去掉就是 14 个假报警。
        """
        assert detect.norm("\x01") == ""
        assert detect.norm("PD\x012") == "pd2"

    def test_unifies_dashes(self):
        assert detect.norm("−1") == detect.norm("-1") == "-1"
        assert detect.norm("–1") == "-1"

    def test_folds_whitespace_and_case(self):
        assert detect.norm("  Obj.  Response \n") == "obj.response"

    def test_checkable_needs_alnum(self):
        """纯符号格（对钩、破折号）不参与比对 —— 它们在文字层里本来就不以该形式出现。"""
        assert detect.checkable("PD2")
        assert detect.checkable("< 0.001")
        assert not detect.checkable("\x01")
        assert not detect.checkable("—")


# ————————————————————— detect.check_merged_rows —————————————————————

PBC_21296_HEADER = [
    "Xenograft line", "Histology", "KM estimate of median", "P -value b",
    "EFS T/C c", "Median final RTV d",
]


def _rows(*data):
    return [PBC_21296_HEADER] + [list(r) for r in data]


class TestMergedRows:
    def test_detects_real_bug(self):
        """pbc_21296 TABLE I 第 41 行：两个数据行被压成一格。真值 44 行，CSV 只有 43。

        PDF p4 文字层里 ALL-4 / ALL-8 / ALL-16 是三个独立的 word span。
        manifest 记的是 confidence=high / grid_status=ok / notes='' —— 完全没举手。
        """
        rows = _rows(
            ("ALL-4", "ALL T-cell", "> EP", "< 0.001", "> 2.2", "0.2"),
            ("ALL-7", "ALL T-cell", "> EP", "< 0.001", "> 2.2", "0.3"),
            ("ALL-2", "ALL B-precursor", "> EP", "< 0.001", "> 2.2", "0.1"),
            ("ALL-8 ALL-16", "ALLT-cell ALLT-cell", "> EP > EP",
             "< 0.001 < 0.001", "> 2.2 > 2.2", "8.1 0.2"),
        )
        hits = detect.check_merged_rows(rows)
        assert [i for i, _ in hits] == [3]
        assert len(hits[0][1]) >= detect.MIN_MULTI_CELLS

    def test_pm_sd_is_one_value_not_two(self):
        """pbc_28772 的 `minRTV mean + SD` 列是 `2.814 ± 0.516` —— 一个值，不是两个。

        不特判 ± 的话，整张金标准表每一行都会被报成并行。
        """
        rows = [
            ["Model", "Agent", "minRTV mean + SD"],
            ["G401", "Control", "2.814 ± 0.516"],
            ["G401", "EPZ011989", "1.429 ± 0.586"],
            ["RBD1", "Control", "1.553 ± 0.366"],
            ["RBD1", "Irinotecan", "0.329 ± 0.422"],
        ]
        assert detect.check_merged_rows(rows) == []

    def test_text_column_with_two_words_is_not_flagged(self):
        """文本列里出现两个词是正常的（`ALL B-precursor`）—— 判据只看数值列。"""
        rows = [
            ["Line", "Histology", "P -value"],
            ["ALL-2", "ALL B-precursor", "0.01"],
            ["ALL-4", "ALL T-cell", "0.02"],
            ["ALL-7", "ALL T-cell", "0.03"],
            ["ALL-8", "ALL B-precursor", "0.04"],
        ]
        assert detect.check_merged_rows(rows) == []

    def test_single_multi_value_cell_is_not_enough(self):
        """一行里只有一个数值格装了两个数，不报 —— 阈值是 MIN_MULTI_CELLS=2。

        单点拟合的阈值，正例只有 1 个、负例 579 行（见 log.md §16）。
        """
        rows = [
            ["Line", "A", "B"],
            ["x1", "1.0", "2.0"],
            ["x2", "1.1", "2.1"],
            ["x3", "1.2", "2.2"],
            ["x4", "1.3 9.9", "2.3"],
        ]
        assert detect.check_merged_rows(rows) == []

    def test_too_few_rows_is_skipped(self):
        assert detect.check_merged_rows([["a", "b"], ["1", "2"]]) == []


class TestBlankRows:
    def test_finds_trailing_blank_rows(self):
        """pbc_24724 Fig.1 末尾有 8 个全空数据行（F-005）。"""
        rows = [["a", "b"], ["1", "2"], ["", ""], ["", "  "]]
        assert detect.check_blank_rows(rows) == [1, 2]


# ————————————————————— detect 的表头豁免 —————————————————————

class TestHeaderRescue:
    def test_multilevel_header_flattened_with_dots(self):
        """留出集 Bao et al. 2025：docling 把两层表头拼成 `Training.Cancer.n =2,254`。

        整串不在文字层里，但三段都在。开发集 12 篇一个这样的样本都没有（F-006）。
        """
        page = detect.norm("Training Cancer Noncancer n =2,254 n =2,553 Internal validation")
        assert detect._header_rescue("Training.Cancer.n =2,254", page)

    def test_single_level_header_with_trailing_dot(self):
        """留出集 Bruhm 2025 的横向表 Table 3：单层表头也被加了分隔符。

        表头是 `Sponsor.` / `Cancertype.` / `Studyname.` —— 父层为空、只剩一个尾点。
        豁免最初要求拆出 >=2 段，导致整行 16 个表头格全部误报、把该表压到 83.9%、
        跌破 90% 硬阈值。放宽到 >=1 段。
        """
        page = detect.norm("Sponsor Assay Cancer type Study name")
        assert detect._header_rescue("Sponsor.", page)
        assert detect._header_rescue("Cancertype.", page)

    def test_rescue_still_requires_every_part_present(self):
        """放宽到 >=1 段之后，"每一段都必须在文字层里"这条不能松 —— 否则豁免就等于关掉判据。"""
        page = detect.norm("Sponsor Assay")
        assert not detect._header_rescue("Sponsor.Nonexistent", page)
        assert not detect._header_rescue("Nonexistent", page)

    def test_rescue_is_header_only_in_check_text_layer(self):
        """豁免只对第 0 行生效 —— 数据行用了就会把 pbc_21296 那个真 bug 放过。"""
        page = detect.norm("A B 0.001 2.2 header one two")
        rows = [
            ["header.one.two"],       # 表头：拆开三段都在 → 命中
            ["< 0.001 < 0.001"],      # 数据行：不给豁免
        ]
        hit, miss = detect.check_text_layer(rows, page)
        assert "header.one.two" not in miss
        assert "< 0.001 < 0.001" in miss


# ————————————————————— gold_io 的行结构 —————————————————————

class TestForwardFill:
    def test_fills_merged_cells(self):
        """AGENTS.md 事实 #15：pbc_30017 的 `Model` 列，OS-2 纵跨两行，第二行该格为空。"""
        assert gold_io.forward_fill(["OS-2", "", "OS-9", ""]) == ["OS-2", "OS-2", "OS-9", "OS-9"]

    def test_leading_blank_stays_blank(self):
        assert gold_io.forward_fill(["", "A", ""]) == ["", "A", "A"]


class TestSectionRows:
    def test_recognises_section_header(self):
        """pbc_26870 的 `Stage 1` / `Stage 2` / `Stage 3` —— 整行只有第一格有值。"""
        assert gold_io.is_section_row(["Stage 1", "", "", ""])
        assert not gold_io.is_section_row(["ES-2", "Patritumab", "20.6", "0.525"])
        assert not gold_io.is_section_row(["", "", ""])

    def test_section_rows_excluded_from_data(self):
        rows = [
            ["Tumor Line", "Treatment Group", "P-value"],
            ["Stage 1", "", ""],
            ["ES-2", "Patritumab", "0.525"],
            ["Stage 2", "", ""],
            ["ES-2", "Cisplatin", "1.000"],
        ]
        header, data, sections = gold_io.split_rows(rows, section_rows=True)
        assert len(data) == 2
        assert sections == ["Stage 1", "Stage 2"]


class TestAnchors:
    def test_section_disambiguates_repeated_pair(self):
        """pbc_26870：`ES-4|Erlotinib` 在 Stage 1 和 Stage 3 各出现一次。

        不把 Stage 算进锚点必然撞车 —— 41 行只有 38 个唯一键。
        """
        rows = [
            ["Tumor Line", "Treatment Group", "P-value"],
            ["Stage 1", "", ""],
            ["ES-4", "Erlotinib", "0.112"],
            ["Stage 3", "", ""],
            ["ES-4", "Erlotinib", "0.900"],
        ]
        with_sec, _ = gold_io.anchors_of(rows, ["Tumor Line", "Treatment Group"], True)
        assert gold_io.check_anchor_unique(with_sec) == []

        without_sec, _ = gold_io.anchors_of(rows, ["Tumor Line", "Treatment Group"], False)
        assert gold_io.check_anchor_unique(without_sec) == [("ES-4|Erlotinib", 2)]

    def test_forward_fill_makes_anchor_unique(self):
        """pbc_28772：Model 列有 8 行为空，不填充时 44 行只有 43 个唯一键。"""
        rows = [
            ["Model", "Agent"],
            ["G401", "Control"],
            ["", "EPZ011989 + irinotecan"],
            ["RBD1", "Control"],
            ["", "EPZ011989 + irinotecan"],
        ]
        keys, _ = gold_io.anchors_of(rows, ["Model", "Agent"], False)
        assert gold_io.check_anchor_unique(keys) == []
        assert keys == ["G401|Control", "G401|EPZ011989 + irinotecan",
                        "RBD1|Control", "RBD1|EPZ011989 + irinotecan"]

    def test_missing_anchor_column_raises(self):
        """锚点列不存在必须当场抛，不能静默降级。"""
        rows = [["Model", "Agent"], ["G401", "Control"]]
        with pytest.raises(gold_io.GoldError):
            gold_io.anchors_of(rows, ["Tumor Line"], False)

    def test_insufficient_anchor_is_detected_not_silently_accepted(self):
        """pbc_28772 只用 `Model` 当锚点时 G401 重复 8 次。

        必须报出来 —— 静默按序号对齐的话，「漏 1 行」会被报成「错 40 处」。
        """
        rows = [["Model", "Agent"]] + [["G401", f"agent{i}"] for i in range(8)]
        keys, _ = gold_io.anchors_of(rows, ["Model"], False)
        assert gold_io.check_anchor_unique(keys) == [("G401", 8)]


class TestLLMTargetStemMatching:
    """`eval/llm.py collect_targets` 的 prefix 反查必须双键注册。

    只注册 `p.stem` 会让 `EAP in advanced gastric cancer..pdf` 整篇被静默跳过 ——
    它的输出目录叫 `EAP_in_advanced_gastric_cancer`（`emit.sanitize_name` 的产物）。
    `eval/checks.py:433-438` 早就踩过这一课，注释里写着"不要猜规则"。

    **这条只能靠单测钉住**：开发集里 EAP 那篇三个关键词都不命中、`matrix.sh` 也不跑
    `--all`，所以 `1.output/EAP_.../` 下一张表 CSV 都没有，跑流水线看不出行为差异。
    """

    def test_sanitized_stem_is_registered(self, tmp_path):
        from pdf_table_extract.emit import sanitize_name

        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        raw = "EAP in advanced gastric cancer."
        (pdf_dir / f"{raw}.pdf").write_bytes(b"%PDF-1.4\n")

        sanitized = sanitize_name(raw)
        assert sanitized != raw, "前提：这个文件名确实会被清理，否则本测试没意义"

        out = tmp_path / "out"
        (out / sanitized).mkdir(parents=True)
        (out / sanitized / "manifest.csv").write_text(
            "csv_path,page,label,source_type\np02_table_1.csv,2,Table 1,text\n"
        )
        (out / sanitized / "p02_table_1.csv").write_text("a,b\n1,2\n")

        from eval.llm import collect_targets

        targets = collect_targets(out, pdf_dir)
        assert len(targets) == 1, "清理过的目录名必须能反查到 PDF，否则整篇静默跳过"


# ————————————————————— gold 打分器 —————————————————————

def _write_gold(tmp_path, body, **head):
    h = {"pdf": "x.pdf", "csv": "p01_t.csv", "label": "T1", "pages": "1",
         "expect_data_rows": "2", "expect_cols": "3",
         "anchor_cols": "Model", "check_cols": "Resp"}
    h.update(head)
    txt = "\n".join(f"{k}: {v}" for k, v in h.items()) + "\n---\n" + body
    p = tmp_path / "t.gold"
    p.write_text(txt)
    return gold_io.load(p)


class TestScorerDoesNotLeakGold:
    """**这条单测才是 gold 隔离的真正保证。**

    `.claude/settings.local.json` 的 deny 规则只挡顺手读（Read / cat / grep），
    挡不住 `open()`。真正让「照着答案调阈值」这件事做不成的，是打分器在**代码层面**
    就不把 gold 的期望值放进任何输出 —— `CellVerdict` 只有 `ours` 字段。
    改 score_against 时若不小心加回 `expected`，这条会立刻变红。
    """

    def test_expected_value_never_appears_in_result(self, tmp_path):
        g = _write_gold(tmp_path, "anchor,Resp,source\nA,ZZZSECRET,human\nB,ZZZSECRET,human\n")
        rows = [["Model", "Resp", "X"], ["A", "PD", "1"], ["B", "CR", "2"]]
        res = gold_io.score_against(g, rows)
        assert "ZZZSECRET" not in repr(res), "打分结果里出现了 gold 的期望值 —— 隔离被破坏"
        assert res.totals["mismatch"] == 2, "前提：这两格确实对不上，否则本测试是空转"


class TestScorerThreeStateEmpty:
    """空值三态。混在一起会让「漏抽」白得分。"""

    def test_sentinel_expects_empty(self, tmp_path):
        g = _write_gold(tmp_path, f"anchor,Resp,source\nA,{gold_io.EMPTY_SENTINEL},human\n",
                        expect_data_rows="1")
        assert gold_io.score_against(g, [["Model", "Resp"], ["A", ""]]).totals["match"] == 1
        assert gold_io.score_against(g, [["Model", "Resp"], ["A", "PD"]]).totals["mismatch"] == 1

    def test_blank_means_unlabeled_not_scored(self, tmp_path):
        """gold 留空 = 还没标，**不能进分母** —— 否则漏抽的格子会白白算对。"""
        g = _write_gold(tmp_path, "anchor,Resp,source\nA,,human\n", expect_data_rows="1")
        t = gold_io.score_against(g, [["Model", "Resp"], ["A", "whatever"]]).totals
        assert t["unlabeled"] == 1 and t["scored"] == 0


class TestScorerSourceFilter:
    def test_default_counts_only_human(self, tmp_path):
        g = _write_gold(tmp_path, "anchor,Resp,source\nA,PD,human\nB,CR,agreed\n")
        rows = [["Model", "Resp"], ["A", "PD"], ["B", "WRONG"]]
        assert gold_io.score_against(g, rows).totals["scored"] == 1
        both = gold_io.score_against(g, rows, {"human", "agreed"}).totals
        assert both["scored"] == 2 and both["mismatch"] == 1


class TestScorerImageTableSupport:
    def test_column_index_spec(self, tmp_path):
        """图片表的表头是 OCR 垃圾，只能按序号引用。"""
        g = _write_gold(tmp_path, "anchor,#1,source\nBT-29,PD2,human\n",
                        anchor_cols="#0", check_cols="#1", expect_data_rows="1", expect_cols="2")
        rows = [["Linte", "Growp Response"], ["BT-29", "PD2"]]
        assert gold_io.score_against(g, rows).totals["match"] == 1

    def test_duplicate_anchor_is_finding_not_crash(self, tmp_path):
        """`pbc_24724 fig_1` 末尾 3 行全空 → 锚点全是空串。

        原实现在这里 raise，而那恰恰是最需要打分的一张表。
        """
        g = _write_gold(tmp_path, "anchor,Resp,source\nA,PD,human\n", expect_data_rows="3")
        rows = [["Model", "Resp"], ["A", "PD"], ["", ""], ["", ""]]
        res = gold_io.score_against(g, rows)
        assert res.anchor_not_unique, "锚点重复必须被记为 finding"
        assert not res.fatal, "但不能中断打分"
