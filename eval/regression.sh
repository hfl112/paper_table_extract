#!/usr/bin/env bash
# 端到端回归。**每一行都是完整命令，可以直接复制单跑。**
#
#   bash eval/regression.sh          # 快档：约 2 分钟，改一次代码跑一次
#   bash eval/regression.sh full     # 全量：约 7 分钟，一轮改动做完再跑
#
# ═══ 为什么分快慢两档（实测数字）═══
#
#   全量 21 条命令 = **424 秒**，其中 12 条在跑 OCR
#   快档 11 条命令 = **121 秒**，一条 OCR 都不跑
#
# 而 OCR 路径的产出**本来就是 `confidence=low`、本来就要人核**（铁律 #4）——
# 花 5 分钟钉住一批"反正不可信"的数字，收益接近零。
# 快档覆盖了文字表的**全部**路径：跨页拼接、横向转正、合并单元格、出版社 OCR 文字层、
# 两个拒跑退出码、以及"该 0 张的不许出表"。
#
# ═══ 它唯一不可替代的能力 ═══
#
# **只有它在查"本该出现的表消失了"。** 单测不跑流水线；`checks.py detect` 只遍历
# 已经存在的 CSV；人眼看一屏数字最容易漏掉的恰恰是"少了一行"。
# 而本项目历史上由用户发现的 3 个 bug **全是**表悄悄没了（log.md §5）。
# 本项目**没有 git**，改坏了没有 diff 可看、没法回滚 —— 这是它存在的根本理由。
#
# 断言的是「行为没变」，不是「行为正确」。已知的错误行为记在 eval/known_issues.md，
# 在 eval/expected.csv 里以错误的值钉着，这样断言器才能全绿、才起得到"防改坏"的作用。
#
# 刻意不写函数、不写循环、不挂管道 —— 就是为了让你能挑任意一行复制出去执行。
# 装了包之后可以直接用 pdf-table-extract 命令；把 `conda run -n gemini ` 去掉即可。

set -u
cd "$(dirname "$0")/.." || exit 1
TIER="${1:-fast}"
OUT=/tmp/pte_regression
rm -rf "$OUT"
mkdir -p "$OUT"   # 负面用例要往 $OUT/.exit 追加退出码，而它们可能在建目录前就退出

echo "########## 快档：文字路径（--no-ocr）##########"
conda run -n gemini pdf-table-extract "0.pdf_input/blood_2014_12_518900.pdf" -k xenograft -k PDX --no-ocr -o "$OUT" --prefix xeno_blood   # table_1 18 / table_2 15 / table_3 7
conda run -n gemini pdf-table-extract "0.pdf_input/pbc_29304.pdf" -k xenograft -k PDX --no-ocr -o "$OUT" --prefix xeno_29304              # table_1 42 行（横向+跨页）
conda run -n gemini pdf-table-extract "0.pdf_input/pbc_26870.pdf" -k xenograft -k PDX --no-ocr -o "$OUT" --prefix xeno_26870              # 0 张（Table 1 的 caption 无 xenograft）
conda run -n gemini pdf-table-extract "0.pdf_input/pbc_21078.pdf" -k Demographic --no-ocr -o "$OUT" --prefix demo_21078                   # table_i 61 行
conda run -n gemini pdf-table-extract "0.pdf_input/pbc_24724.pdf" -k Demographic --no-ocr -o "$OUT" --prefix demo_24724                   # 0 张（不该误报）

echo "########## 快档：金标准 44 行数据 + 1 行表头 = 45 ##########"
conda run -n gemini pdf-table-extract "0.pdf_input/pbc_28772.pdf" --all --no-ocr -o "$OUT" --prefix gold_28772                            # table_1 45 行

echo "########## 快档：续表 caption 为空，靠列名一致拼接 ##########"
conda run -n gemini pdf-table-extract "0.pdf_input/pbc_26870.pdf" --all --no-ocr -o "$OUT" --prefix all_26870                             # table_1 45 行（27 + 19-1）

echo "########## 快档：gold 打分需要的文字表（其余 gold 表已被上面的命令覆盖）##########"
conda run -n gemini pdf-table-extract "0.pdf_input/pbc_21296.pdf" -k response --no-ocr -o "$OUT" --prefix gold_21296                       # table_i 44 行（横向转正）

echo "########## 快档：负面测试 ##########"
conda run -n gemini pdf-table-extract "0.pdf_input/566455.pdf" -k Demographic -o "$OUT" --prefix neg_preprint                             # 应退出码 3：preprint 拒跑
echo "neg_preprint $?" >> "$OUT/.exit"
conda run -n gemini pdf-table-extract "0.pdf_input/rstl.1665.0039.pdf" -k response -o "$OUT" --prefix neg_scanned                         # 应退出码 4：零标号拒跑
echo "neg_scanned $?" >> "$OUT/.exit"
conda run -n gemini pdf-table-extract "0.pdf_input/pbc_21078.pdf" -k zzzznotexist --no-ocr -o "$OUT" --prefix neg_nomatch                 # 0 张，干净空 manifest
conda run -n gemini pdf-table-extract "0.pdf_input/EAP in advanced gastric cancer..pdf" --all --no-ocr -o "$OUT" --prefix ocr_layer        # table_1 22 行，confidence 必须 low

if [ "$TIER" = "full" ]; then
  echo
  echo "########## 全量追加：-k response（含图片路径，慢）##########"
  conda run -n gemini pdf-table-extract "0.pdf_input/pbc_21078.pdf" -k response -o "$OUT" --prefix resp_21078                   # table_ii 32(caption) + table_iii 36(cell)
  conda run -n gemini pdf-table-extract "0.pdf_input/pbc_24724.pdf" -k response -o "$OUT" --prefix resp_24724                   # table_ii 42(cell) + fig_1 47(caption,图片表)
  conda run -n gemini pdf-table-extract "0.pdf_input/pbc_21296.pdf" -k response -o "$OUT" --prefix resp_21296                   # table_i 44(cell)；Fig.1 120dpi 还原失败留失败行
  conda run -n gemini pdf-table-extract "0.pdf_input/pbc_28772.pdf" -k response -o "$OUT" --prefix resp_28772                   # table_1 45(cell) ← caption 无 response，靠第 11 列 Obj. Response
  conda run -n gemini pdf-table-extract "0.pdf_input/pbc_29304.pdf" -k response -o "$OUT" --prefix resp_29304                   # table_1 42(cell)
  conda run -n gemini pdf-table-extract "0.pdf_input/pbc_30017.pdf" -k response -o "$OUT" --prefix resp_30017                   # table_1 13(caption)
  conda run -n gemini pdf-table-extract "0.pdf_input/blood_2014_12_518900.pdf" -k response -o "$OUT" --prefix resp_blood         # table_2 15(cell) + table_3 7(cell)
  conda run -n gemini pdf-table-extract "0.pdf_input/1078-0432.CCR-18-2728.pdf" -k response -o "$OUT" --prefix resp_ccr          # 0 张 + 5 个 supp 标号提示
  conda run -n gemini pdf-table-extract "0.pdf_input/pbc_24724.pdf" --list -o "$OUT" --prefix list_24724
fi

echo
echo "########## 断言：与 eval/expected.csv 的现状快照对账 ##########"
conda run -n gemini python eval/checks.py regression "$OUT" eval/expected.csv --tier "$TIER" || RC=1

# ═══ gold 打分：断言「逐格准确性」没退步 ═══
#
# expected.csv 断言的是**形状**（行列数、有没有出表）；gold 分断言的是**内容对不对**。
# 两者不能互相替代：`pbc_21296 p04_table_i` 曾经形状全绿、同时漏掉 2 个数据行。
#
# 为什么非要接进来（`log.md` §37.1 的真事故）：打分器不在回归里 ⇒ 没人重跑产物 ⇒
# `1.output/` 比代码旧了 3 天 ⇒ 打分报「漏行 5」，而那 5 行早就修好了。
# **过期产物给出的是一个自信的错误结论**，比不打分更危险。
#
# `--map` 是必需的：上面用的目录名是 `--prefix`（`gold_28772`），不是 PDF 的 stem，
# 打分器默认按 stem 找会全部 miss、而 miss 是静默的（只少几行输出）。
GOLD_MAP="pbc_28772=gold_28772,pbc_26870=all_26870,pbc_21296=gold_21296"
if [ "$TIER" = "full" ]; then
  # 全量档才有图片表的产物（快档 --no-ocr，图片路径整个不跑）
  GOLD_MAP="pbc_28772=gold_28772,pbc_26870=all_26870,pbc_21296=resp_21296,pbc_24724=resp_24724"
fi
echo
echo "########## 断言：与 eval/expected_gold.csv 的逐格准确性基线对账 ##########"
conda run -n gemini python eval/gold.py --score --outdir "$OUT" --source human,agreed \
  --map "$GOLD_MAP" --baseline eval/expected_gold.csv --tier "$TIER" || RC=1

exit "${RC:-0}"
