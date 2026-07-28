"""LLM 独立第二读者：找出零标注判据看不见的错，只产出**分歧清单**。

    conda run -n gemini python eval/llm.py submit  1.output --pdf-dir 0.pdf_input
    conda run -n gemini python eval/llm.py collect msgbatch_xxx

**边界（重要）**：这个文件属于 `eval/`，**不属于 CLI 包**。`AGENTS.md` 的非目标写着
「不联网、不调 API」—— 那说的是 `pdf_table_extract/` 本体，它至今一行网络代码都没有。
评测脚手架用 LLM 是本轮明确批准的例外。

**为什么需要它**：零标注判据（`eval/checks.py detect`）能抓「并行」和「文字层对不上」，但抓不到
两类错：

  1. **错误断列** —— 便宜启发式实测 2/2 全是误报（「空表头 / 高空值率列」判据在 `blood` table_2
     和 `pbc_30017` 上把合法的合并单元格误判成断列）
  2. **图片表的字形错** —— OCR 认的字，文字层里本来就没有，没有本地参照

**为什么不用「纯文字 OCR 当第二意见」**：`engine_paddle.plain_text` 用 `PaddleOCR(lang="en")`、
`recognize_table` 用 `TableRecognitionPipelineV2(text_recognition_model_name="en_PP-OCRv5_mobile_rec")`
—— **共用识别模型**，`Rhabdold` 这种错两边大概率一起犯。看页面图的多模态模型才是真独立。

**三条硬规则**：

  1. **产出只是分歧清单，绝不写回 CSV。** 铁律 #1：前身项目最痛的 bug 就是"信任网格高于原文"，
     忠实地抄出了一张错误的表。LLM 会给出**看起来完全合理、但 PDF 里没有**的值。
  2. **喂整页原图，不喂工具裁剪过的区域。** 否则工具裁掉的行两边都看不见、不产生分歧、
     永远不会被推到人面前 —— 而"漏行"正是最需要第二双眼睛的地方。
  3. **只在开发集 12 篇上跑。** 留出集那 25 篇没人核对过，分歧了没人裁决，跑了也没有裁决者。

**它也不是金标准。** 人裁决过的格子才是（`source=human`）；LLM 与工具一致的格子只是
`source=agreed`（两个独立来源同意，**没人看过**）。这两档在 `eval/gold.py` 里不许混。
校准办法：另抽 1–2 张表让人通读全表，量「一致 = 正确」的成立率。
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import sys
from pathlib import Path

MODEL = "claude-opus-4-8"
RENDER_DPI = 200          # 整页 200dpi 足够读表；再高只是让 base64 变大
MAX_TOKENS = 16000

# 只要能表达「这张表有几行、每行各列是什么」就够，剩下的比对在本地做。
TABLE_SCHEMA = {
    "type": "object",
    "properties": {
        "n_data_rows": {
            "type": "integer",
            "description": "表格的数据行数，不含表头，不含 `Stage 1` 这类分节标题行",
        },
        "header": {
            "type": "array",
            "items": {"type": "string"},
            "description": "表头各列的文字，从左到右",
        },
        "rows": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "string"}},
            "description": "每个数据行的各列取值，从左到右。合并单元格造成的空格子留空字符串",
        },
        "notes": {
            "type": "string",
            "description": "读不清或不确定的地方；没有就留空",
        },
    },
    "required": ["n_data_rows", "header", "rows", "notes"],
    "additionalProperties": False,
}

PROMPT = """这是一篇英文文献 PDF 的第 {pages} 页（整页原图，未经裁剪）。

页面上有一张标号为 `{label}` 的表格。请把**这一张表**逐格读出来。

要求：
- 只读这一张表，忽略页面上的正文、其他表、图。
- 按原文照抄，不要解读、不要换算单位、不要补全缩写。
- 合并单元格（某个值在原表里纵跨多行）造成的空格子，留空字符串，**不要**替你填上继承值。
- `Stage 1` 这类整行只有一个值的分节标题行，**不计入** n_data_rows，也不要放进 rows。
- 表格若跨页，把给你的这几页上属于这张表的数据行**全部**读出来。
- 读不清的格子照你看到的写，并在 notes 里说明。

作为参照，另一个工具从同一张表抽出的表头是：
{tool_header}

这个表头**可能是错的**（列可能被切错、可能少一列）。不要照着它对齐 —— 以页面图为准。
"""


# ————————————————————— 提交 —————————————————————

_SPANS = re.compile(r"spans_pages=(\d+)-(\d+)")


def render_pages(pdf: Path, pages: list[int], dpi: int) -> bytes:
    """把若干页竖向拼成一张 PNG。跨页表要一起看才能判断接缝。

    **传进来的应该是归一化后的 PDF**（见 `normalized_for`）—— 否则横向表会躺着渲染出来。
    """
    import fitz

    doc = fitz.open(pdf)
    try:
        pix = [doc[p - 1].get_pixmap(dpi=dpi) for p in pages if 0 < p <= len(doc)]
        if len(pix) == 1:
            return pix[0].tobytes("png")
        from PIL import Image
        import io

        imgs = [Image.open(io.BytesIO(p.tobytes("png"))) for p in pix]
        w = max(i.width for i in imgs)
        canvas = Image.new("RGB", (w, sum(i.height for i in imgs)), "white")
        y = 0
        for i in imgs:
            canvas.paste(i, (0, y))
            y += i.height
        buf = io.BytesIO()
        canvas.save(buf, format="PNG")
        return buf.getvalue()
    finally:
        doc.close()


def normalized_for(pdf: Path, stack) -> Path:
    """拿到该 PDF 的归一化版本（横向表页已转正）。用 ExitStack 保活到整轮结束。

    ═══ 为什么必须归一化再渲染（dry-run 实测抓到的）═══

    `render_pages` 原先直接渲染原始 PDF，结果**横向表是躺着的** —— `pbc_21296` p4
    渲染成 1650x2200 竖图、整张表要从下往上读。模型虽然能读旋转文字，但这是白白加难度，
    而 gold 初稿的质量直接决定后面所有打分的可信度。

    `page.get_pixmap()` **会**套用 `/Rotate`（这点和 `get_text` 相反 —— 后者返回未旋转帧，
    见 `eval/geom.page_words` 的注释）。所以喂归一化 PDF 就能得到正立的图。
    """
    # 直接跑脚本时 `eval` 包不在 sys.path 上（`python eval/llm.py` 的工作目录是仓库根，
    # 但 sys.path[0] 是 `eval/` 本身）
    root = str(Path(__file__).resolve().parent.parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    from eval import geom

    return stack.enter_context(geom.normalized_pdf(pdf))[0]


def collect_targets(outdir: Path, pdf_dir: Path) -> list[dict]:
    # 双键注册：PDF 的 stem 与它被 `emit.sanitize_name` 清理后的名字都要能查到。
    # 原来只注册 stem，于是 `EAP in advanced gastric cancer..pdf`（输出目录叫
    # `EAP_in_advanced_gastric_cancer`）**整篇被静默跳过**。
    # `eval/checks.py:433-438` 早就踩过并解决了这一课，注释里专门写了"不要猜规则"。
    from pdf_table_extract.emit import sanitize_name

    stems: dict[str, Path] = {}
    for p in pdf_dir.glob("*.pdf"):
        stems.setdefault(p.stem, p)
        stems.setdefault(sanitize_name(p.stem), p)
    out = []
    for prefix_dir in sorted(d for d in outdir.iterdir() if d.is_dir()):
        pdf = stems.get(prefix_dir.name)
        if pdf is None:
            continue
        manifest: dict[str, dict] = {}
        man = prefix_dir / "manifest.csv"
        if man.exists():
            with man.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    name = Path(row.get("csv_path", "") or "").name
                    if name and name not in manifest:
                        manifest[name] = row
        for csv_path in sorted(prefix_dir.glob("*.csv")):
            if csv_path.name in ("manifest.csv", "captions.csv"):
                continue
            meta = manifest.get(csv_path.name, {})
            pages = [int(meta.get("page") or csv_path.name[1:3])]
            m = _SPANS.search(meta.get("notes", ""))
            if m:
                pages = list(range(int(m.group(1)), int(m.group(2)) + 1))
            rows = list(csv.reader(csv_path.open(newline="")))
            out.append({
                "custom_id": f"{prefix_dir.name}__{csv_path.stem}",
                "pdf": pdf,
                "csv_path": csv_path,
                "pages": pages,
                "label": meta.get("label") or csv_path.stem,
                "header": rows[0] if rows else [],
                "n_data_rows": max(len(rows) - 1, 0),
            })
    return out


def cmd_submit(args: argparse.Namespace) -> int:
    targets = collect_targets(args.outdir, args.pdf_dir)
    if not targets:
        print("没有找到任何表（要求 prefix == PDF 文件名 stem）")
        return 1

    if args.only:
        want = list(args.only)
        all_ids = [t["custom_id"] for t in targets]
        missed = [w for w in want if not any(w in cid for cid in all_ids)]
        if missed:
            # 不静默 —— 打错一个 id 就少提交一张表，而 batch 回来才发现太晚
            print(f"⚠ 这些 --only 没匹配到任何表: {missed}")
            print(f"  可选的 custom_id: {all_ids}")
        targets = [t for t in targets if any(w in t["custom_id"] for w in want)]
        if not targets:
            print("--only 过滤后一张表都不剩")
            return 1

    # ═══ --dry-run：渲染 + 落 prompt，**不调 API** ═══
    #
    # 为什么必须有这一步：5 张 gold 里 3 张是跨页表，`render_pages` 把多页竖向拼成一张 PNG。
    # 拼错了（顺序反了、缺页、缩放不一致）模型就读错，而那要等 batch 回来才发现、钱已经花了。
    # 先肉眼看一眼拼图，确认了再提交。
    if args.dry_run:
        import contextlib
        args.dry_run.mkdir(parents=True, exist_ok=True)
        stack = contextlib.ExitStack()
        norm_cache: dict[str, Path] = {}
        print(f"DRY-RUN：{len(targets)} 张表，渲染到 {args.dry_run}（不调 API、不花钱）")
        print("-" * 96)
        print("%-34s %-12s %8s %7s %10s  %s" %
              ("custom_id", "页", "PNG尺寸", "KB", "工具行数", "工具表头(前 3 列)"))
        for t in targets:
            key = str(t["pdf"])
            if key not in norm_cache:
                norm_cache[key] = normalized_for(t["pdf"], stack)
            png = render_pages(norm_cache[key], t["pages"], args.dpi)
            out = args.dry_run / f"{t['custom_id']}.png"
            out.write_bytes(png)
            prompt = PROMPT.format(
                pages=",".join(str(p) for p in t["pages"]),
                label=t["label"],
                tool_header=" | ".join(t["header"]),
            )
            (args.dry_run / f"{t['custom_id']}.prompt.txt").write_text(prompt)
            try:
                from PIL import Image
                import io
                im = Image.open(io.BytesIO(png))
                dim = f"{im.width}x{im.height}"
            except Exception:
                dim = "?"
            print("%-34s %-12s %8s %7d %10d  %s" %
                  (t["custom_id"][:34], ",".join(map(str, t["pages"])), dim,
                   len(png) // 1024, t["n_data_rows"], " | ".join(t["header"][:3])[:34]))
        print("-" * 96)
        print(f"PNG 与 prompt 已写入 {args.dry_run}")
        print("**跨页表请重点看拼接处**：页序对不对、有没有缺页、两页缩放是否一致。")
        stack.close()
        return 0

    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    import contextlib

    requests = []
    stack = contextlib.ExitStack()
    norm_cache: dict[str, Path] = {}
    for t in targets:
        key = str(t["pdf"])
        if key not in norm_cache:
            norm_cache[key] = normalized_for(t["pdf"], stack)
        png = render_pages(norm_cache[key], t["pages"], args.dpi)
        requests.append(Request(
            custom_id=t["custom_id"],
            params=MessageCreateParamsNonStreaming(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                thinking={"type": "adaptive"},
                output_config={"format": {"type": "json_schema", "schema": TABLE_SCHEMA}},
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.standard_b64encode(png).decode("utf-8"),
                        }},
                        {"type": "text", "text": PROMPT.format(
                            pages=",".join(str(p) for p in t["pages"]),
                            label=t["label"],
                            tool_header=" | ".join(t["header"]),
                        )},
                    ],
                }],
            ),
        ))

    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)

    index = {t["custom_id"]: {
        "csv_path": str(t["csv_path"]),
        "pages": t["pages"],
        "label": t["label"],
        "n_data_rows": t["n_data_rows"],
    } for t in targets}
    args.index.write_text(json.dumps({"batch_id": batch.id, "targets": index},
                                     ensure_ascii=False, indent=2))

    print(f"已提交 {len(requests)} 张表  batch={batch.id}  status={batch.processing_status}")
    print(f"索引写到 {args.index}")
    print(f"完成后跑： conda run -n gemini python eval/llm.py collect {batch.id}")
    return 0


# ————————————————————— 收集与比对 —————————————————————

def _norm(s: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKC", s)
    for a, b in (("−", "-"), ("–", "-"), ("—", "-")):
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip().lower()


def diff_one(csv_path: Path, llm: dict) -> list[str]:
    """本地比对，产出分歧。**只报分歧，不改任何东西。**"""
    rows = list(csv.reader(csv_path.open(newline="")))
    tool_header, tool_rows = (rows[0], rows[1:]) if rows else ([], [])
    out: list[str] = []

    if llm["n_data_rows"] != len(tool_rows):
        out.append(f"行数分歧：工具 {len(tool_rows)}，LLM {llm['n_data_rows']}")
    if len(llm["header"]) != len(tool_header):
        out.append(f"列数分歧：工具 {len(tool_header)}，LLM {len(llm['header'])}")

    for i, (a, b) in enumerate(zip(tool_header, llm["header"])):
        if _norm(a) != _norm(b):
            out.append(f"表头列 {i} 分歧：工具 {a!r}，LLM {b!r}")

    # 逐行比对只在行数一致时做 —— 行数不一致时按序号比会产生一片假分歧
    if llm["n_data_rows"] == len(tool_rows):
        for r, (ta, la) in enumerate(zip(tool_rows, llm["rows"])):
            for c, (av, bv) in enumerate(zip(ta, la)):
                if _norm(av) != _norm(bv):
                    col = tool_header[c] if c < len(tool_header) else f"col{c}"
                    out.append(f"行 {r} / {col}: 工具 {av!r}，LLM {bv!r}")
    if llm.get("notes"):
        out.append(f"LLM 备注: {llm['notes']}")
    return out


def cmd_collect(args: argparse.Namespace) -> int:
    import anthropic

    meta = json.loads(args.index.read_text())
    targets = meta["targets"]
    client = anthropic.Anthropic()

    batch = client.messages.batches.retrieve(args.batch_id)
    if batch.processing_status != "ended":
        print(f"批次还没结束：{batch.processing_status}  "
              f"（processing={batch.request_counts.processing}）")
        return 2

    report: list[str] = []
    n_ok = n_err = n_disagree = 0
    for result in client.messages.batches.results(args.batch_id):
        t = targets.get(result.custom_id)
        if t is None:
            continue
        if result.result.type != "succeeded":
            n_err += 1
            report.append(f"\n## {result.custom_id}\n  ✗ {result.result.type}")
            continue
        msg = result.result.message
        if msg.stop_reason == "refusal":
            n_err += 1
            report.append(f"\n## {result.custom_id}\n  ✗ 模型拒答（stop_reason=refusal）")
            continue
        text = next((b.text for b in msg.content if b.type == "text"), "")
        try:
            llm = json.loads(text)
        except json.JSONDecodeError:
            n_err += 1
            report.append(f"\n## {result.custom_id}\n  ✗ 返回不是合法 JSON")
            continue
        n_ok += 1
        diffs = diff_one(Path(t["csv_path"]), llm)
        if diffs:
            n_disagree += 1
            report.append(f"\n## {result.custom_id}  ({t['label']}, p{t['pages']})")
            report.append(f"  CSV: {t['csv_path']}")
            for d in diffs[:40]:
                report.append(f"    · {d}")
            if len(diffs) > 40:
                report.append(f"    · …另有 {len(diffs) - 40} 处")

    head = [
        "# LLM 第二读者：分歧清单",
        "",
        f"模型 `{MODEL}`，整页原图 {args.dpi}dpi，Batch API。",
        "",
        "**这不是金标准。** 分歧处需要人裁决才算 `source=human`；",
        "没有分歧的格子只是 `source=agreed`（两个独立来源同意，没人看过）。",
        "LLM 会自信地读错密集数字表 —— 分歧不等于工具错了。",
        "",
        f"成功 {n_ok} 张，失败 {n_err} 张，其中 **{n_disagree} 张有分歧**。",
    ]
    args.out.write_text("\n".join(head + report) + "\n")
    print("\n".join(head))
    print(f"\n写到 {args.out}")
    return 0


# ————————————————————— gold 草稿 —————————————————————

# 5 张 gold 表的配置。**形状（行数/列数）不写死** —— 由 LLM 读出来当草稿值，人订正。
# 写死人核值等于把答案先塞进去，那就不是独立第二读者了。
GOLD_SPECS: dict[str, dict] = {
    "pbc_21296__p04_table_i": {
        "pdf": "pbc_21296.pdf", "csv": "p04_table_i.csv", "label": "TABLE I", "pages": "4",
        "anchor_cols": "Xenograft line", "check_cols": "Histology, Response activity",
        "section_rows": False,
        "why": "横向文字表，F-001 正例（docling 把 ALL-8/ALL-16 压成一行）",
    },
    "pbc_28772__p03_table_1": {
        "pdf": "pbc_28772.pdf", "csv": "p03_table_1.csv", "label": "TABLE 1", "pages": "3-4",
        "anchor_cols": "Model + Agent", "check_cols": "Obj. Response",
        "section_rows": False,
        "why": "金标准表，**负例** —— 用来测误报率。处理组 Agent 是锚点的一部分，"
               "所以处理组错会表现为漏行/多出行，等于隐式被检",
    },
    "pbc_26870__p18_table_1": {
        "pdf": "pbc_26870.pdf", "csv": "p18_table_1.csv", "label": "Table 1", "pages": "18-19",
        "anchor_cols": "Tumor Line + Treatment Group", "check_cols": "Response",
        "section_rows": True,
        "why": "横向 + 跨页 + 续表 caption 为空；有 Stage 1/2/3 分节行。三类列齐全",
    },
    "pbc_24724__p05_fig_1": {
        "pdf": "pbc_24724.pdf", "csv": "p05_fig_1.csv", "label": "Fig. 1", "pages": "5",
        "anchor_cols": "#0", "check_cols": "#5",
        "section_rows": False,
        "why": "图片表，有效 300dpi。工具侧首列名缺失导致整体左移，只能用列序号",
    },
    "pbc_21296__p03_fig_1": {
        "pdf": "pbc_21296.pdf", "csv": "p03_fig_1.csv", "label": "Fig. 1", "pages": "3",
        "anchor_cols": "#0", "check_cols": "#4",
        "section_rows": False,
        "why": "低分辨率图片表（120dpi）。工具侧表头是 OCR 垃圾（Linte / Growp Response）",
    },
}


def _resolve_col(header: list[str], spec: str) -> int | None:
    """把列声明解析成下标。支持 `#N` 序号、精确名、归一化名。"""
    m = re.match(r"^#(\d+)$", spec)
    if m:
        i = int(m.group(1))
        return i if i < len(header) else None
    if spec in header:
        return header.index(spec)
    n = _norm(spec)
    for i, h in enumerate(header):
        if _norm(h) == n:
            return i
    return None


def _rename_for(rows: list[list[str]], specs: list[str], tool_header: list[str]) -> list[list[str]] | None:
    """把 rows 的表头在指定位置改名成 spec 字面量，好让 gold.anchors_of 按名字找到。

    解析顺序：先在**本表表头**上找（`#N` / 精确 / 归一化）；找不到且**列数与工具侧一致**时，
    退回用工具侧的下标。两者都不成立就返回 None —— **中止，不猜**（`NEXT.md` H2）。
    """
    out = [list(rows[0])] + [list(r) for r in rows[1:]]
    for spec in specs:
        i = _resolve_col(rows[0], spec)
        if i is None and len(rows[0]) == len(tool_header):
            i = _resolve_col(tool_header, spec)
        if i is None:
            return None
        out[0][i] = spec
    return out


def cmd_gold_draft(args: argparse.Namespace) -> int:
    """把 batch 结果转成 gold 草稿。**输出全写 gold 目录，stdout 只有计数。**

    每格的 source：LLM 与工具**一致** → `agreed`；**不一致** → 写 LLM 的值、
    标 `disputed_unresolved`（写工具的值会让人"看着挺对"直接放过，反过来更能逼出裁决）。

    正文保留 **LLM 读到的全部列**（用户 2026-07-26 定）—— `check_cols` 只决定打分时看哪几列，
    多存的列零成本，以后想扩随时能用。
    """
    import anthropic
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from eval import gold as gold_io

    meta = json.loads(args.index.read_text())
    targets = meta["targets"]
    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(args.batch_id)
    if batch.processing_status != "ended":
        print(f"批次还没结束：{batch.processing_status}")
        return 2

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_written = n_skip = 0
    summary: list[str] = []

    for result in client.messages.batches.results(args.batch_id):
        cid = result.custom_id
        spec = GOLD_SPECS.get(cid)
        t = targets.get(cid)
        if spec is None or t is None:
            continue
        if result.result.type != "succeeded":
            summary.append(f"{cid}: 批次失败 {result.result.type}")
            n_skip += 1
            continue
        try:
            llm = json.loads(next((b.text for b in result.result.message.content
                                   if b.type == "text"), ""))
        except (json.JSONDecodeError, StopIteration):
            summary.append(f"{cid}: 返回不是合法 JSON")
            n_skip += 1
            continue

        tool_rows = list(csv.reader(Path(t["csv_path"]).open(newline="")))
        tool_header = tool_rows[0] if tool_rows else []
        llm_rows = [list(llm["header"])] + [list(r) for r in llm["rows"]]

        anchors = [c.strip() for c in spec["anchor_cols"].split("+") if c.strip()]
        checks = [c.strip() for c in spec["check_cols"].split(",") if c.strip()]

        llm_named = _rename_for(llm_rows, anchors, tool_header)
        tool_named = _rename_for(tool_rows, anchors + checks, tool_header)
        if llm_named is None or tool_named is None:
            summary.append(f"{cid}: 列声明在 LLM/工具表头上都解析不出来 —— **中止，请手写草稿**")
            n_skip += 1
            continue

        try:
            # LLM 侧**永远按 section_rows=False 算** —— prompt 明确要求它不要输出
            # `Stage 1` 这类分节标题行，所以它的表里根本没有分节行可识别。
            lkeys, ldata = gold_io.anchors_of(llm_named, anchors, False)
            tkeys, tdata = gold_io.anchors_of(tool_named, anchors, spec["section_rows"])
        except gold_io.GoldError as e:
            summary.append(f"{cid}: 锚点算不出来 —— {e}")
            n_skip += 1
            continue

        # 分节标签**按行序从工具侧搬到 LLM 侧**。
        #
        # 不搬的后果（实测踩到）：`pbc_26870` 工具侧锚点是 `Stage 1|ES-2|Patritumab`、
        # LLM 侧是 `|ES-2|Patritumab`，**41 行全部对不上、全标成待裁决**，
        # 而实际上两边行数一致、内容大体相同 —— 纯粹是我这边口径不一致造成的假分歧。
        #
        # 只在**行数相等**时搬（都是文档顺序，位置对齐安全）。行数不等就不搬 ——
        # 那时位置对齐会把「漏 1 行」放大成「错 40 处」，正是 `gold.py` 反复警告的那件事。
        if spec["section_rows"] and len(lkeys) == len(tkeys):
            _, _, tsections = gold_io.split_rows(tool_named, True)
            lkeys = [gold_io.ANCHOR_SEP.join([tsections[i], k]) for i, k in enumerate(lkeys)]

        th = tool_named[0]
        tool_by_anchor = {k: tdata[i] for i, k in enumerate(tkeys)}
        lh = llm_named[0]

        body = [["anchor"] + lh + ["source"]]
        n_disp = 0
        for i, k in enumerate(lkeys):
            row = ldata[i]
            trow = tool_by_anchor.get(k)
            agree = trow is not None and all(
                _norm(row[_resolve_col(lh, c)] if _resolve_col(lh, c) is not None
                      and _resolve_col(lh, c) < len(row) else "")
                == _norm(trow[th.index(c)] if c in th and th.index(c) < len(trow) else "")
                for c in checks
            )
            if not agree:
                n_disp += 1
            body.append([k] + [(row[j] if j < len(row) else "") for j in range(len(lh))]
                        + ["agreed" if agree else "disputed_unresolved"])

        head = [
            f"pdf: {spec['pdf']}",
            f"csv: {spec['csv']}",
            f"label: {spec['label']}",
            f"pages: {spec['pages']}",
            f"expect_data_rows: {len(lkeys)}",
            f"expect_cols: {len(lh)}",
            f"anchor_cols: {spec['anchor_cols']}",
            f"check_cols: {spec['check_cols']}",
            f"section_rows: {'true' if spec['section_rows'] else 'false'}",
            "",
            f"# 草稿：LLM 读出来的，**还不是 gold**。{spec['why']}",
            "# 订正规则：你改过的格子把 source 改成 human；没改且标 agreed 的可以留着；",
            "#           标 disputed_unresolved 的必须裁决（写的是 LLM 的值，工具值见 .review.md）。",
            "# 形状（expect_data_rows / expect_cols）也是 LLM 读的，请一并核。",
            "---",
        ]
        buf = io_StringIO()
        csv.writer(buf).writerows(body)
        (args.out_dir / f"{cid}.gold").write_text("\n".join(head) + "\n" + buf.getvalue())

        # 并排裁决表：只有分歧行，工具值 vs LLM 值
        rev = [f"# {cid} 待裁决（{n_disp} 行有分歧 / 共 {len(lkeys)} 行）", "",
               "| anchor | 列 | 工具 | LLM |", "|---|---|---|---|"]
        for i, k in enumerate(lkeys):
            trow = tool_by_anchor.get(k)
            for c in checks:
                lj = _resolve_col(lh, c)
                lv = row_at(ldata[i], lj)
                tv = row_at(trow, th.index(c)) if trow is not None and c in th else "(该行工具侧没有)"
                if _norm(lv) != _norm(tv):
                    rev.append(f"| `{k}` | {c} | `{tv}` | `{lv}` |")
        for k in tkeys:
            if k not in set(lkeys):
                rev.append(f"| `{k}` | — | (工具有此行) | **LLM 没读到这一行** |")
        if llm.get("notes"):
            rev += ["", f"LLM 备注：{llm['notes']}"]
        (args.out_dir / f"{cid}.review.md").write_text("\n".join(rev) + "\n")

        n_written += 1
        summary.append(f"{cid}: {len(lkeys)} 行，其中 {n_disp} 行待裁决"
                       f"（LLM 读到 {llm['n_data_rows']} 数据行 / 工具 {t['n_data_rows']}）")

    print(f"gold 草稿写到 {args.out_dir}：成功 {n_written} 张，跳过 {n_skip} 张")
    print("-" * 88)
    for s in summary:
        print("  " + s)
    print("-" * 88)
    print("**stdout 只有计数，格子的值全在 gold 目录里** —— 那是刻意的，见 gold.py 的非对称契约。")
    return 0 if n_skip == 0 else 1


def row_at(row, j) -> str:
    if row is None or j is None or j >= len(row):
        return ""
    return row[j]


def io_StringIO():
    import io
    return io.StringIO()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit", help="渲染整页图并提交 Batch")
    s.add_argument("outdir", type=Path)
    s.add_argument("--pdf-dir", type=Path, required=True)
    s.add_argument("--dpi", type=int, default=RENDER_DPI)
    s.add_argument("--index", type=Path, default=Path("eval/.llm_batch.json"))
    s.add_argument("--only", action="append", metavar="SUBSTR",
                   help="只处理 custom_id 含该串的表，可重复。不给就是全部 16 张")
    s.add_argument("--dry-run", type=Path, metavar="DIR",
                   help="渲染 PNG + 落 prompt 到 DIR，**不调 API**。跨页拼接必须先肉眼确认")
    s.set_defaults(func=cmd_submit)

    c = sub.add_parser("collect", help="取回结果并产出分歧清单")
    c.add_argument("batch_id")
    c.add_argument("--index", type=Path, default=Path("eval/.llm_batch.json"))
    c.add_argument("--out", type=Path, default=Path("eval/llm_disagreements.md"))
    c.add_argument("--dpi", type=int, default=RENDER_DPI)

    g = sub.add_parser("gold-draft", help="把 batch 结果转成 gold 草稿（写进 gold 目录）")
    g.add_argument("batch_id")
    g.add_argument("--index", type=Path, default=Path("eval/.llm_batch.json"))
    g.add_argument("--out-dir", type=Path, default=Path("/Users/funanhe/pte_gold"),
                   help="**必须在仓库外** —— 草稿里有 LLM 读出的真值，AI 看到就等于看了答案")
    g.set_defaults(func=cmd_gold_draft)
    c.set_defaults(func=cmd_collect)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
