"""探针：留出集 17/25 篇检出 0 张表，是真没有还是 caption 判据漏检？（findings.md F-003）

    conda run -n gemini python spike/probe_zero_table_papers.py

**为什么需要非循环的检查**：直接问"caption 判据找到几个"是循环论证 —— 判据漏了就是漏了。
这里换个角度：`AGENTS.md` 事实 #2 的判据要求标号位于**某一行的行首**。那就对比

    A. 行首规则（现行判据）找到的标号
    B. 全文任意位置出现的 `Table <数字/罗马>` 串

**B − A 就是"标号出现了、但不在行首"的情况**。逐条看它们的上下文即可判断：
  - 上下文像 `... as shown in Table 1, the ...` ⇒ 正文引用，行首规则**正确地**排除了它
  - 上下文像 `Table 1. Patient characteristics ...` 却不在行首 ⇒ **判据漏检**，是真 bug

这不能替代人眼（真 caption 也可能整个不在文字层里），但能把需要人工翻的范围缩小很多。
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import fitz

HOLDOUT = Path("/Users/funanhe/Downloads/Zotero_Precancer_PDFs")

# 与 AGENTS.md 事实 #2 一致：标号位于某一「行」的开头，标号后紧跟 . : 空格 或行尾
LINE_START = re.compile(
    r"^(?:Supplementary\s+)?(?:Table|TABLE|Tab\.)\s+(?:S?\d+|[IVXivx]{1,4})(?:[.:\s]|$)"
)
# 任意位置出现的表标号
ANYWHERE = re.compile(r"(?:Supplementary\s+)?(?:Table|TABLE|Tab\.)\s+(?:S?\d+|[IVXivx]{1,4})")

# 判断"看起来像 caption 而不是引用"：标号后面跟的是一段描述，而不是 `, the` / `shows` 这类。
# **两条分开写**：标点不能用 `\b` 收尾 —— `) and the ...` 里 `)` 后面是空格，
# 两边都是非词字符、构不成词边界，写成 `(?:\)|\.)\b` 会一个都匹配不上（我第一版就是这么错的，
# 结果 86 处括号引用全被当成"疑似漏检"）。
REFERENCE_PUNCT = re.compile(r"^\s*[)\].,;:]")
REFERENCE_WORD = re.compile(
    r"^\s*(?:and|or|shows?|showed|lists?|summari[sz]e[sd]?|presents?|"
    r"in|of|for|see|from|to|as|is|are|were|was|which|that|with)\b",
    re.I,
)


def looks_like_reference(ctx: str) -> bool:
    return bool(REFERENCE_PUNCT.match(ctx) or REFERENCE_WORD.match(ctx))


def main() -> None:
    # `--pdf-dir` 是后加的，默认仍是留出集 —— 老的调用方式（不带参数）行为不变。
    # 加它是为了能在 `gen/` 那批 PPTP 新料上跑同一个非循环检查。
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf-dir", type=Path, default=HOLDOUT)
    args = ap.parse_args()
    pdfs = sorted(args.pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"✗ {args.pdf_dir} 下没有 PDF")
    print(f"{'PDF':44s} {'行首':>4s} {'任意':>4s} {'仅非行首':>8s}  疑似漏检的上下文")
    print("-" * 120)

    total_suspect = 0
    for path in pdfs:
        doc = fitz.open(path)
        text = "\n".join(p.get_text() for p in doc)
        doc.close()

        # 折叠换行造成的标号断裂（AGENTS.md 事实 #1），但保留行结构用于行首判定
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]

        at_line_start = {
            m.group(0).strip().rstrip(".:").lower()
            for ln in lines if (m := LINE_START.match(ln))
        }
        flat = re.sub(r"\s+", " ", text)
        anywhere: dict[str, list[str]] = defaultdict(list)
        for m in ANYWHERE.finditer(flat):
            key = m.group(0).strip().lower()
            anywhere[key].append(flat[m.end():m.end() + 60])

        only_mid = {k: v for k, v in anywhere.items() if k not in at_line_start}

        # 只挑"后面不像引用措辞"的，那些才可能是被漏掉的 caption
        suspect = []
        for key, ctxs in only_mid.items():
            for c in ctxs:
                if not looks_like_reference(c) and len(c.strip()) > 15:
                    suspect.append((key, c.strip()[:52]))
                    break

        total_suspect += len(suspect)
        note = "" if not suspect else "  ".join(f"{k}→{c!r}" for k, c in suspect[:2])
        print(f"{path.stem[:44]:44s} {len(at_line_start):4d} {len(anywhere):4d} "
              f"{len(only_mid):8d}  {note[:60]}")

    print("-" * 120)
    print(f"合计疑似漏检上下文 {total_suspect} 处。")
    print("「仅非行首」大多应该是正文引用（`Supplementary Table S1` 之类），"
          "只有最后一列里出现「像描述文字」的才值得人眼去核。")


if __name__ == "__main__":
    main()
