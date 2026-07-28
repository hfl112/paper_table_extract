# 后续计划

（2026-07-26 夜重写。上一版是上一轮结束时的自我总结，有两处判断已被数据推翻，见下。
本轮做了什么见 `log.md` §25 与 `eval/findings.md` F-014 / F-015。）

---

## 0. 现在的状态

**本轮只出证据，`pdf_table_extract/` 一个字节没动**（校验和已复核）。
交付物是两个检测器的取证 + 打分器 + 5 张 gold 的准备工作。

| 已完成 | 状态 |
|---|---|
| F-001 根因判定 | ✅ **是 TableFormer 自己合并的**，`export_to_dataframe` 无辜 → 修法不在 `_df_to_rows` |
| D-cell 检测器（cell bbox 高度） | ✅ 28 个页-表 **1/1 真阳性、0 误报** |
| D-band 检测器（词坐标行带） | ✅ M1 优于 M2；开发集 19 张表**带多方向 1/1、0 误报** |
| 打分器 `gold.py --score` | ✅ 非对称输出 + `ZZZSECRET` 不泄露单测 |
| eval 两个 bug | ✅ `report.py` argv / `llm.py` sanitize_name |
| spike 瘦身 | ✅ 删 9 个一次性探针（备份 `/tmp/pte_spike_backup/`） |

**上一版 NEXT.md 被推翻的两处**：

1. 「7 条阈值从没被碰到」—— 把**没被量过**说成了**没被执行过**。那些阈值天天在跑，
   缺的是余量测量，不是覆盖。
2. 「语料已收敛，不用再找料」—— 与它自己 §4 的缺料清单自相矛盾，且 F-001 那条已承认
   留出集在这一项上没有判别力。

---

## 1. 下一步：gold（**需要你动手**）

打分器已就绪且**刻意写在 gold 数据之前**。缺的只有 gold 数据本身。

### 5 张 gold 表（已按实际表头核过）

| # | 表 | 形态 | anchor_cols | check_cols | 期望数据行 |
|---|---|---|---|---|---|
| 1 | `pbc_21296` p4 `table_i` | 横向文字表，**F-001 正例** | `Xenograft line` | `Histology`, `Response activity` | **44**（现 43） |
| 2 | `pbc_28772` p3 `table_1` | 文字表，**负例（测误报）** | `Model` + `Agent` | `Obj. Response` | 44 |
| 3 | `pbc_26870` p18 `table_1` | 横向+跨页+**续表 caption 为空** | `Tumor Line` + `Treatment Group`，`section_rows: true` | `Response` | **41**（另 3 行是 `Stage 1/2/3` 分节行） |
| 4 | `pbc_24724` p5 `fig_1` | 图片表，有效 300dpi | `#0`（首列名缺失） | `#5` | **44**（现 46 + 3 全空行） |
| 5 | `pbc_21296` p3 `fig_1` | **低分辨率图片表**（120dpi） | `#0`（`Linte`） | `#4`（`Growp Response`） | 44 |

> 原先选的 `pbc_21078 p04_table_i` **已换掉** —— 实测它是**人口学表**
> （`Xenograft/Panel/Patient age/Sex/Diagnosis/...`），没有处理组也没有响应列。

### 流程

1. `eval/llm.py` 加三样：`--only`（现在会把 16 张全提交）、`--dry-run`（渲染 PNG 不调 API ——
   5 张里 3 张跨页，`render_pages` 的竖向拼接必须先肉眼确认再花钱）、`gold-draft` 子命令
2. LLM 读**整页原图**产出全表（**不裁列** —— 用户 2026-07-26 定：TABLE_SCHEMA 本来就返回全表，
   限制反而要多写代码。gold 草稿保留全部列，只有**打分**按 `check_cols` 聚焦三类列）
3. 你逐张订正：改过的格子 `source` 改 `human`，没改的留 `agreed`
4. 跑两档 `--source human` 与 `--source human,agreed`，差值就是「LLM 与工具一致但没人看过」的规模

### 隔离（**漏了这一步 gold 就废了**）

- gold 放 `/Users/funanhe/pte_gold/`（仓库外），`llm.py collect` / `gold-draft` 的输出**也全写那儿**
  —— 分歧清单含 LLM 读出的真值，AI 读到就等于看了答案
- 跑那两条命令时 **AI 不许打印 stdout**（`> /dev/null`，只看退出码）
- **你**在 `.claude/settings.local.json` 加 `deny`：`Read(/Users/funanhe/pte_gold/**)` 及
  `cat`/`head`/`tail`/`grep` 同路径
- API key 已在 `/Users/funanhe/00_MyCode/idea_generator/.env`

---

## 2. 再下一步：把 D-cell 推进产品

D-cell 便宜（三行代码、零额外 IO）、零误报，而目前**这类错完全没有任何防线**。

1. `engine_docling.convert` 顺带读 `ti.data.table_cells`，把 per-cell bbox 带出来
2. `rules.audit` 加判据：排除表头行后，同一数据行 >=2 列的 cell 高于该列中位数 1.5x
   → `notes` 写 `merged_row?`，`decide_confidence` 降级
3. **改前必须先在留出集重跑取证**（`holdout/` 产物已过期，含 `/emspace`）
4. 改完立刻 `bash eval/regression.sh`

**更彻底的修法待验证**：换 `TableFormerMode.ACCURATE`（`AGENTS.md` 记它可用、当前代码没设），
或用 cell bbox + 坐标行带做后处理拆分。**两条都没验过。**

---

## 3. 还欠的

| 事项 | 说明 |
|---|---|
| **F-015** | `pbc_21078 p04_table_i` 实际 60 行、`AGENTS.md` 人核值 59，差 1。两页各自自洽，**疑在拼接环节**。未查清 |
| `GAP_M1` 改自适应 | 现在 1.50 ← 3.0 → 3.61，**上侧只有 1.2x**，3.61 是孤点 |
| D-band 的「带少」方向 | M1 有 6 张，未逐条核实是坐标法过度合并还是 docling 多切了行。优先级低（不是 F-001 那个方向） |
| `holdout/` 重跑 | 产物过期，当独立证据用之前必须先重跑 |
| 接缝行数校验 | 跨页拼接只有"要不要拼"的判据，没有"拼得对不对"的检查。需先 spike |
| `--keep-work` | 让 `normalized.pdf` 留下来，便于人工核对与 eval 复用（现在 eval 得自己重建） |

---

## 4. 明确不做的

- **不做「矫正」** —— 不拿 LLM 或推断去填/改格子值，踩铁律 #1。
  改抽取逻辑让它从 PDF 里抽对要做；补值不做
- **不为了让判据亮灯而调参** —— 开发集若没有断列正例，就记「未验证」
- **`566455` p61 那个 D-band 假阳性不再调** —— 试了三种划界法，再调就是单点拟合（F-013 的教训）。
  且该篇是 preprint、工具拒跑，表不会被产出
- 不内置同义词表、不做关键词表达式语言、不加 `--tables-only`（`AGENTS.md` 非目标）

---

## 5. 还缺的测试料

| 缺什么 | 状态 |
|---|---|
| 竖排方向 `(0,1)` 的**真横向表** | 触发过 4 次全是误报，**正例仍然 0 个**，`270°` 分支至今没被验证 |
| 「整本扫描但没做 OCR」的 PDF | `is_scanned_document` 的字数下限**负例侧无样本** |
| arXiv 以外平台的预印本 | 结构判据的 4 个正例全来自 arXiv |
| **更多横向转正表** | F-001 那类 bug 的分母 —— 开发集只有 3 张、留出集只有 1 张（还只有 3 个数据行） |
