#!/usr/bin/env bash
# 留出集：验证**阈值**在从未参与开发的论文上还成不成立。
#
#   bash holdout/run.sh
#
# ═══ 职责边界（只做这一件事）═══
#
# 只验阈值，**不验抽取质量**。所以：
#   - 只跑 `--list`，不跑抽取、不跑 ML，几秒/篇
#   - 不需要 ground truth（那 25 篇没人核对过）
#   - **不依赖 eval/ 的任何代码**，holdout/ 自包含
#
# "抽出来的表对不对"是 eval 的事，不在这里。
#
# ═══ 这批料该怎么用（别用错）═══
#
# 它是 **test set，不是 validation set**：只在"我说做完了"的时候跑，
# **跑完不许拿它调参**，否则它就退化成第二个开发集。
#
# ⚠ "跑了留出集"不等于"验过了"。还要看这批料里有没有该问题出现的场景。
#   实测教训：某个 bug 在留出集 0 命中，但留出集里能出现该 bug 的表只有 1 张 ——
#   分母不对，"没命中"什么都说明不了。见 `log.md` §19。
#
# ═══ 检得到 / 检不到 ═══
#
# 检得到：标号量级、VERTICAL_LINES_THRESHOLD + VERTICAL_RATIO_THRESHOLD（转正）、
#         FULL_PAGE_IMAGE_COVER + OCR_LAYER_MIN_CHARS（OCR 文字层）、PREPRINT_MIN_PAGE_RATIO
# 检不到：图片路径的任何东西；CROP_PAD_RATIO / MIN_RIVAL_OVERLAP / MIN_IMAGE_COVER_H
#         （`--list` 根本不经过）；以及**标号漏检**（漏了只显示成"这篇表少一点"，
#         看起来完全合理 —— 这是本工具最大的盲区）

set -u
cd "$(dirname "$0")/.." || exit 1
IN="/Users/funanhe/Downloads/Zotero_Precancer_PDFs"
OUT=holdout

echo "################ --list 全部 $(ls "$IN"/*.pdf | wc -l | tr -d ' ') 篇（不跑 ML）"
for f in "$IN"/*.pdf; do
  conda run -n gemini pdf-table-extract "$f" --list -o "$OUT" > /dev/null 2>&1 \
    || echo "  ✗ 失败: $(basename "$f")"
done

echo
conda run -n gemini python holdout/report.py --pdf-dir "$IN"
