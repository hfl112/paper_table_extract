#!/usr/bin/env bash
# 关键词矩阵：对 0.pdf_input/ 全部 PDF 跑 --list + 三组关键词（response / xenograft+PDX / Demographic）。
#
#   bash eval/matrix.sh
#
# 产物：1.output/<pdf名>/
#           captions.csv    ← --list 的机读清单（全部图表的标号 + 完整 caption/legend）
#           manifest.csv    ← **累加**：三组检索的行都在同一份里，靠 query 列区分
#           p0X_*.csv       ← 命中的表，同一张表被两个词命中时只存一份
#
# 每篇一个文件夹、每篇一份 manifest —— 换关键词不覆盖历史（靠 query 列区分）。

set -u
cd "$(dirname "$0")/.." || exit 1
OUT=1.output
rm -rf "$OUT"

for f in 0.pdf_input/*.pdf; do
  echo "================================================================ $(basename "$f")"
  conda run -n gemini pdf-table-extract "$f" --list -o "$OUT" > /dev/null 2>&1
  conda run -n gemini pdf-table-extract "$f" -k response  -o "$OUT" 2>&1 | grep -E "^错误|本次检索|^导出"
  conda run -n gemini pdf-table-extract "$f" -k xenograft -k PDX -o "$OUT" 2>&1 | grep -E "^错误|本次检索|^导出"
  conda run -n gemini pdf-table-extract "$f" -k Demographic -o "$OUT" 2>&1 | grep -E "^错误|本次检索|^导出"
done

echo
echo "################ 汇总清单 ################"
conda run -n gemini python eval/report.py matrix "$OUT"
