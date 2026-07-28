"""扫 Zotero 全库，按**性质**找能验到剩余阈值的测试料。

    conda run -n gemini python spike/probe_find_testmaterial.py

按期刊名挑是碰运气，按性质挑才能覆盖。要找的五类，各对应一个还没验够的阈值：

  ① 扫描件            → FULL_PAGE_IMAGE_COVER + OCR_LAYER_MIN_CHARS（**只有反例，最缺**）
  ② 别的标号写法      → DEFAULT_LABEL_CORE（F-009：eTable / Additional file / Extended Data Table）
  ③ 同页多张表        → MIN_RIVAL_OVERLAP（**单点拟合**，37 篇里只有 1 处）
  ④ 竖排方向 (0,1)    → rotation_for_page 的 270° 分支（**按对称性推断、从未验证**）
  ⑤ 页内竖排水印      → arXiv 那种"只盖第 1 页"的预印本标记

只读不写，纯 PyMuPDF，不跑任何 ML。
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import fitz

ZOTERO = Path("/Users/funanhe/Zotero/storage")

# ① 扫描件：整页被单张位图覆盖 >=95% 宽高，且文字层 >=1000 字（AGENTS.md 事实 2d）
COVER, MIN_CHARS = 0.95, 1000

# ② 现行正则认不出的标号写法
ALT_LABEL = {
    "eTable (JAMA)": re.compile(r"^eTables?\s+\d+", re.M),
    "Additional file (BMC)": re.compile(r"^Additional\s+file\s+\d+", re.M | re.I),
    "Extended Data Table": re.compile(r"^Extended\s+Data\s+Tables?\s+\d+", re.M | re.I),
    "Supplemental Table": re.compile(r"^Supplemental\s+Tables?\s+S?\d+", re.M | re.I),
    "Web Table / Online Table": re.compile(r"^(?:Web|Online)\s+Tables?\s+\d+", re.M | re.I),
}

# ③ 同页多张表：行首 Table 标号
CAPTION = re.compile(r"^(?:Supplementary\s+)?(?:Table|TABLE|Tab\.)\s+(?:S?\d+|[IVXivx]{1,4})(?:[.:\s]|$)")

# ⑤ 竖排水印：第 1 页左边缘的竖排、含冒号+数字
WATERMARK = re.compile(r"(arXiv|bioRxiv|medRxiv|Research\s*Square|SSRN|ChemRxiv|Preprint)", re.I)


def scan(path: Path) -> dict | None:
    try:
        doc = fitz.open(path)
    except Exception:
        return None
    out = defaultdict(list)
    try:
        n = len(doc)
        if n == 0:
            return None
        for pno, page in enumerate(doc, 1):
            text = page.get_text()
            w, h = page.rect.width, page.rect.height

            # ① 扫描件
            if len(text) >= MIN_CHARS:
                for im in page.get_image_info():
                    b = im["bbox"]
                    if (b[2] - b[0]) / w >= COVER and (b[3] - b[1]) / h >= COVER:
                        out["scanned"].append(pno)
                        break

            # ③ 同页多张表
            caps = [ln for ln in (re.sub(r"\s+", " ", x).strip() for x in text.split("\n"))
                    if CAPTION.match(ln)]
            if len(caps) >= 2:
                out["multi_table"].append((pno, len(caps)))

            # ④/⑤ 竖排文字
            up = 0
            for b in page.get_text("dict")["blocks"]:
                for ln in b.get("lines", []):
                    t = "".join(s["text"] for s in ln["spans"]).strip()
                    if not t:
                        continue
                    d = tuple(round(x) for x in ln["dir"])
                    if d == (0, 1):
                        up += 1
                    if pno == 1 and d in ((0, 1), (0, -1)) and WATERMARK.search(t):
                        out["p1_watermark"].append(t[:56])
            if up >= 30:
                out["dir_up"].append((pno, up))

        full = "\n".join(re.sub(r"[ \t]+", " ", p.get_text()) for p in doc)
        for name, pat in ALT_LABEL.items():
            m = pat.findall(full)
            if m:
                out["alt_label"].append((name, len(m)))
    finally:
        doc.close()
    return dict(out) if out else None


OUT_JSON = Path("/tmp/pte_material.json")


def main() -> None:
    pdfs = sorted(ZOTERO.rglob("*.pdf"))
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(pdfs)
    pdfs = pdfs[:limit]
    print(f"扫 {len(pdfs)} 篇 …", flush=True)

    found = defaultdict(list)
    for i, p in enumerate(pdfs, 1):
        if i % 200 == 0:
            print(f"  … {i}/{len(pdfs)}", flush=True)
        r = scan(p)
        if not r:
            continue
        for k, v in r.items():
            found[k].append((p, v))

    # 完整清单落盘 —— 报告只打前几条，组装语料要全量
    import json
    OUT_JSON.write_text(json.dumps(
        {k: [[str(p), str(v)] for p, v in rows] for k, rows in found.items()},
        ensure_ascii=False, indent=1))
    print(f"\n完整清单写到 {OUT_JSON}")

    def report(key: str, title: str, why: str, top: int = 8) -> None:
        rows = found.get(key, [])
        print()
        print("=" * 100)
        print(f"{title} —— {len(rows)} 篇")
        print(f"  （{why}）")
        print("=" * 100)
        for p, v in rows[:top]:
            print(f"  {p.parent.name}/{p.name[:62]}")
            print(f"      {str(v)[:110]}")

    report("scanned", "① 扫描件候选", "整页位图 >=95%×95% 且文字层 >=1000 字 —— 补 OCR 文字层判据的**正例**")
    report("alt_label", "② 别的标号写法", "F-009：现行正则认不出的写法", top=12)
    report("multi_table", "③ 同页多张表", "MIN_RIVAL_OVERLAP 单点拟合，37 篇里只有 1 处")
    report("dir_up", "④ 竖排方向 (0,1)", "rotation 的 270° 分支从未验证")
    report("p1_watermark", "⑤ 第 1 页竖排水印", "arXiv 那种「只盖首页」的预印本标记")

    print()
    print("=" * 100)
    print("汇总")
    for k, t in (("scanned", "扫描件"), ("alt_label", "别的标号写法"), ("multi_table", "同页多张表"),
                 ("dir_up", "竖排 (0,1)"), ("p1_watermark", "首页竖排水印")):
        print(f"  {t:14s} {len(found.get(k, [])):4d} 篇")


if __name__ == "__main__":
    main()
