# `gen/` —— PPTP 泛化跑（第三批料，边界与前两批都不同）

```bash
python gen/stage.py --dry-run && python gen/stage.py   # 摊平语料（必须先跑）
bash   gen/run.sh sweep                                # 97 篇 --list，约 3 分钟
python holdout/report.py --pdf-dir ~/Downloads/pte_pptp_gen/sweep97 --raw
bash   gen/run.sh noocr                                # 深挖 12 篇，--no-ocr
python gen/check.py --outdir ~/Downloads/pte_pptp_gen/out_noocr --prereg gen/prereg.csv
```

## 一句话职责

拿 `01_paper_extract` 抓的 PPTP collection（105 篇，**排除与开发集重合的 8 篇后剩 97 篇**），
**真正跑抽取**，验证 v1 在没参与开发的文献上抽得对不对。

## 三批料各管一件事，不要混用

| | 语料 | 篇数 | 跑什么 | 验的是哪个轴 |
|---|---|---|---|---|
| `0.pdf_input/` | 开发集 | 12 | 全部 | —— 阈值就是在它上面拟合的，**不是证据** |
| `holdout/` | Zotero precancer 2020–2025 | 25 | **只 `--list`** | **领域轴**：换领域后 4 个阈值判据还成不成立 |
| **`gen/`（本目录）** | **PPTP collection** | **97 / 深挖 10** | **`--list` + 抽取** | **出版社 / 年代 / 版式轴**，以及**抽取质量**本身 |

**本目录能证明什么**：

- 抽取质量（形状、跨页拼接、横向转正、命中与漏抽）在新文献上的表现 —— 这是 `holdout/` 刻意不做的。
- 出版社与年代跨度：Wiley PBC 2007–2020（含 `TABLE I.` 与 `TABLE 1` 两代模板）、Springer 2011、
  AACR（Cancer Res / CCR / MCT）2008–2026、Blood 2015–2016、Cell Press 2019、Elsevier 综述 2024。
- **`VERTICAL_LINES_THRESHOLD` 的下侧余量** —— `NEXT.md` §5 的第一缺料。开发集只有 3 张横向转正表、
  留出集只有 1 张（还只有 3 个数据行）；这批一次给出 **22 页 / 16 篇**。

**本目录不能证明什么（不要越界）**：

- **不能证明跨领域泛化。** 105 篇全是 PPTP 儿童肿瘤临床前测试，与开发集**同领域**。
  领域轴是 `holdout/` 的活。从本目录得出"它泛化了"这种全局结论，就是重犯 `AGENTS.md`
  记的那个「分母不对，比例就是假的」的坑。
- **`j_celrep_2019_09_071` 只是半独立的。** 它是开发集成员 `566455`（`10.1101/566455`）的
  **正式发表版**，标题逐字相同、表也是同一批。留着它是因为它是这 10 篇里唯一的 Cell Press 版式，
  也是唯一"零真表 caption"的样本；但**不计入"N 篇从没见过"的计数**。
- **图片路径的质量**在 `--no-ocr` 档下测不到（只测得到"该走图片路径的有哪些"）。

## 它是 test set，不是 validation set

**跑完不许拿这批观测值去调阈值。** 允许的动作只有"报出失准"。一旦照着它调参，它就退化成第二个
开发集，往后再也不能当泛化证据 —— `AGENTS.md` 记过留出集被这样污染过一次（2026-07-26）。

## 为什么产物在仓库外

`~/Downloads/pte_pptp_gen/`。两个理由：

1. **抽取产物含已发表文献的表格原文**，按 `.gitignore` 表头的版权约定不能提交。而本仓库
   **现在有 git 了**（`AGENTS.md` 里"本项目没有 git"那句已过期），误提交一次就永久进历史 ——
   每新增一个输出目录就要记得加一行 ignore，漏一次不可逆。
2. `1.output/` 不能用：`eval/matrix.sh` 开头就 `rm -rf` 它，长跑产物离被删只差一条日常命令。

沿用 `spike/build_threshold_corpus.py` → `~/Downloads/pte_threshold_corpus` 的先例。
仓库里只放脚本、`prereg.csv`（只有标号和预期，无 caption 原文、无单元格值）、findings 与 log。

## 目录布局

```
~/Downloads/pte_pptp_gen/
  sweep97/     97 个 symlink，名字用 article_id（不必发明命名规则，可直接回查 articles.csv）
  deep10/      12 个 symlink：10 篇深挖 + 1 篇半独立(celrep) + 1 篇阳性对照(posctl_pbc_29304)
  corpus.csv   article_id ↔ 短名 ↔ doi/journal/year/md5/页数/选它的理由
  list97/      97 篇 --list 的 captions.csv
  out_noocr/   --no-ocr 档产物      ← 两档**必须分开放**，见下
  out_full/    OCR 档产物
  logs/  png/
```

## 三个必须知道的坑

1. **`out_noocr` 与 `out_full` 不能共用一个根目录。** `emit.describe_query()` 不看
   `--no-ocr`，两档的 `query` 列都是 `response` ⇒ 后跑的会把先跑的 manifest 行**静默替换**，
   而先跑的 CSV 还留在盘上、读起来像命中。
2. **阳性对照 `posctl_pbc_29304` 排在最前，不过就别往下跑。** 它是开发集成员，正确答案钉在
   `eval/expected.csv`（`p03_table_1.csv` 42 行 = 41 数据行、`matched_on=caption`、`high`），
   走完全相同的摊平路径。理由见 `log.md` §35.1：那次 macOS 上没有 `timeout`，25 个输出目录
   全空，而"这批料没有图片表"这个结论**看起来完全合理**。任何"0 命中"都得有同批阳性对照垫底。
3. **`--dump-images` 有 bug、macOS 没有 `timeout`。** 人核用
   `python eval/llm.py submit <outdir> --pdf-dir <deep10> --dry-run <png目录>` 渲染 ——
   它从**归一化后**的 PDF 渲染（横向表正立）、跨页表竖向拼好，且在 `import anthropic`
   之前就返回，**不调 API 不花钱**。不要跑不带 `--dry-run` 的 `submit`（LLM 裁决只对开发集开放）。
