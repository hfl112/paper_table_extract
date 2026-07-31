#!/usr/bin/env bash
# PPTP 泛化跑的命令序列。先 `python gen/stage.py`。
#
#   bash gen/run.sh sweep              # 97 篇 --list（不跑 ML，约 5 分钟）
#   bash gen/run.sh noocr [篇名]       # 深挖 12 篇，--no-ocr（约 16 s/篇/条命令）
#   bash gen/run.sh full  [篇名]       # 同上但开 OCR —— **贵且不封顶**，先拿一篇探成本
#
# 刻意的决定，每条都有理由：
#
# 1. **`noocr` 与 `full` 写到不同的输出根目录。** `emit.describe_query()` 不看 `--no-ocr`，
#    两档跑出来的 `query` 列都是 `response` ⇒ 共用一个根目录时，第二次会把第一次的
#    manifest 行**静默替换掉**，而第一次的 CSV 还留在盘上、看起来像命中。
# 2. **不用 `> /dev/null`**（`holdout/run.sh` 那样）。`--no-ocr：跳过 N 个图片候选`
#    这类信息只在 stdout 上，是图片路径的免费普查数据，丢了就得重跑。
# 3. **不用 `timeout`** —— macOS 上没有这个命令（`gtimeout` 也没装）。log.md §35.1 记过：
#    那次 25 个输出目录全空，而"这批料没有图片表"这个结论看起来完全合理。
# 4. **不用 `--dump-images`** —— 拷出循环的 `seq` 在 `kept + img_tables` 上重算，而文件名
#    是在 `cands` 上铸的，通常静默拷不出东西或拷错张。人核改用 `eval/llm.py --dry-run` 渲染。
# 5. **全程串行、不加 `&`** —— `emit.write_manifest()` 无锁（AGENTS.md 已知限制 #6）。
# 6. **不 `rm -rf`** 任何东西。要重来自己删，脚本不替你决定。

set -u
cd "$(dirname "$0")/.." || exit 1

PHASE="${1:-}"
ONLY="${2:-}"
B="$HOME/Downloads/pte_pptp_gen"
LOGS="$B/logs"

[ -d "$B/deep10" ] || { echo "✗ 未摊平，先跑 python gen/stage.py"; exit 1; }
mkdir -p "$LOGS"

# 深挖名单。阳性对照与最大压力篇排在**最前**：
#   posctl 不过就别往下跑；pharmthera 68 页是耗时和内存的探针，早失败好过晚失败。
PAPERS="posctl_pbc_29304 j_pharmthera_2024_108742 pbc_26825 pbc_22188 pbc_22576 \
        pbc_22741 s00280_011_1618_8 0008_5472_can_16_0122 1078_0432_ccr_18_2675 \
        blood_2016_03_707414 pbc_22921 j_celrep_2019_09_071"

run() {  # run <标签> <输出根> <篇名> [额外参数...]
  local tag=$1 out=$2 n=$3; shift 3
  local lg="$LOGS/${tag}__${n}.log"
  conda run -n gemini pdf-table-extract "$B/deep10/$n.pdf" "$@" -o "$out" > "$lg" 2>&1
  local rc=$?
  echo "$tag $n $rc" >> "$out/.exit"
  printf '  %-6s %-26s rc=%d  %s\n' "$tag" "$n" "$rc" \
    "$(grep -E '^导出|^错误' "$lg" | head -1)"
}

case "$PHASE" in
  sweep)
    OUT="$B/list97"; mkdir -p "$OUT"
    n=0
    for f in "$B"/sweep97/*.pdf; do
      n=$((n + 1))
      conda run -n gemini pdf-table-extract "$f" --list -o "$OUT" \
        > "$LOGS/sweep__$(basename "$f" .pdf).log" 2>&1 \
        || echo "  ✗ 失败: $(basename "$f")"
    done
    echo "sweep 完成 $n 篇 → $OUT"
    ;;

  noocr|full)
    OUT="$B/out_$PHASE"; mkdir -p "$OUT"
    EXTRA=""; [ "$PHASE" = noocr ] && EXTRA="--no-ocr"
    LIST="$PAPERS"; [ -n "$ONLY" ] && LIST="$ONLY"
    for n in $LIST; do
      echo "──────────────── $n"
      run list "$OUT" "$n" --list
      run resp "$OUT" "$n" -k response $EXTRA
      run xeno "$OUT" "$n" -k xenograft -k PDX $EXTRA
    done
    echo
    echo "退出码汇总（3=preprint 拒跑、4=零标号，都是**有意义的正确结果**，不是失败）:"
    sort "$OUT/.exit" | uniq -c | awk '$4!=0 || $1>0'
    ;;

  *)
    sed -n '2,10p' "$0"; exit 2 ;;
esac
