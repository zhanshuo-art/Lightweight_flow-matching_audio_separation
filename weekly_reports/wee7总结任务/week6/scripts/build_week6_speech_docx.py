from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "week6" / "展示ppt生成" / "week6汇报文字稿.docx"


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_width(cell, width_dxa):
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.first_child_found_in("w:tcW")
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_cell_margins(table, top=80, start=120, bottom=80, end=120):
    tbl_pr = table._tbl.tblPr
    tbl_cell_mar = tbl_pr.first_child_found_in("w:tblCellMar")
    if tbl_cell_mar is None:
        tbl_cell_mar = OxmlElement("w:tblCellMar")
        tbl_pr.append(tbl_cell_mar)
    for m, v in [("top", top), ("start", start), ("bottom", bottom), ("end", end)]:
        node = tbl_cell_mar.find(qn(f"w:{m}"))
        if node is None:
            node = OxmlElement(f"w:{m}")
            tbl_cell_mar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")


def set_table_width(table, width_dxa, col_widths):
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.first_child_found_in("w:tblW")
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(width_dxa))
    tbl_w.set(qn("w:type"), "dxa")

    tbl_grid = table._tbl.tblGrid
    if tbl_grid is None:
        tbl_grid = OxmlElement("w:tblGrid")
        table._tbl.insert(0, tbl_grid)
    for child in list(tbl_grid):
        tbl_grid.remove(child)
    for width in col_widths:
        grid_col = OxmlElement("w:gridCol")
        grid_col.set(qn("w:w"), str(width))
        tbl_grid.append(grid_col)

    for row in table.rows:
        for idx, cell in enumerate(row.cells):
            set_cell_width(cell, col_widths[idx])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_run(paragraph, text, bold=False, italic=False, color=None):
    run = paragraph.add_run(text)
    run.bold = bold
    run.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)
    return run


def add_bullet(doc, text):
    p = doc.add_paragraph(style="List Bullet")
    p.add_run(text)
    return p


def add_note_box(doc, title, lines):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_width(table, 9360, [9360])
    set_cell_margins(table, top=120, bottom=120, start=180, end=180)
    cell = table.cell(0, 0)
    set_cell_shading(cell, "F4F6F9")
    p = cell.paragraphs[0]
    add_run(p, title, bold=True, color="1F4D78")
    for line in lines:
        p = cell.add_paragraph()
        p.style = doc.styles["Normal"]
        p.add_run(line)
    doc.add_paragraph()


def add_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_width(table, sum(widths), widths)
    set_cell_margins(table, top=80, bottom=80, start=120, end=120)
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        set_cell_shading(hdr[i], "E8EEF5")
        hdr[i].paragraphs[0].add_run(h).bold = True
    for row in rows:
        cells = table.add_row().cells
        for i, value in enumerate(row):
            cells[i].paragraphs[0].add_run(str(value))
    doc.add_paragraph()
    return table


def configure_document(doc):
    sec = doc.sections[0]
    sec.top_margin = Inches(1)
    sec.bottom_margin = Inches(1)
    sec.left_margin = Inches(1)
    sec.right_margin = Inches(1)
    sec.header_distance = Inches(0.492)
    sec.footer_distance = Inches(0.492)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25

    for name, size, color, before, after in [
        ("Heading 1", 16, "2E74B5", 18, 10),
        ("Heading 2", 13, "2E74B5", 14, 7),
        ("Heading 3", 12, "1F4D78", 10, 5),
    ]:
        style = styles[name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.line_spacing = 1.25

    for name in ["List Bullet", "List Number"]:
        style = styles[name]
        style.font.name = "Calibri"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.font.size = Pt(11)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.line_spacing = 1.25


def add_slide_script(doc, slide_no, title, target, script, notes=None):
    doc.add_heading(f"{slide_no}. {title}", level=2)
    add_note_box(doc, "本页要讲清楚", [target])
    doc.add_heading("建议讲稿", level=3)
    for para in script:
        doc.add_paragraph(para)
    if notes:
        doc.add_heading("提醒", level=3)
        for item in notes:
            add_bullet(doc, item)


slides = [
    (
        "标题页",
        "简短开场，说明本周主题是 TF-mask 后处理与 IRM 辅助微调。",
        [
            "各位老师、同学好，我是潘羽。今天汇报的是 Week6 的工作，主题是 TF-mask 后处理与 IRM 辅助微调。",
            "前面几周我们已经完成了从 self-attention baseline 到 Mamba-style Mini-DiT，再到 Week5 waveform L1 fine-tuning 的路线。本周不再更换主干，而是在 Week5 Mamba FT 的基础上加入 mask 约束，看看目标声源预测还能不能继续提升。",
        ],
    ),
    (
        "本周主线",
        "先让听众分清楚：TF-mask 是推理后的输出修正，IRM 微调是训练时加入 mask 监督。",
        [
            "这一页我先讲 Week6 的主线。我们不是重新换主干，也不是再做 self-attention 对照，而是在 Week5 已经训练好的 Mamba 分离器上，检查一个更贴近传统源分离的约束：mask 约束。",
            "这里有两个容易混淆的概念。TF-mask 后处理是不训练模型的，它只是在推理之后把模型预测的 target 幅度谱转成 mask，再用 mixture 的幅度做一次约束。IRM 微调则是训练，它把理想 mask 作为一个辅助监督项加到 loss 里。",
            "所以 Week6 的逻辑可以概括为三步：先用 TF-mask 看简单后处理有没有收益；再用 Oracle IRM 看理论上界还有多大；如果上界足够高，就继续训练一次，让模型在训练时学习 mask 结构。",
        ],
        [
            "这里重点不要把 TF-mask 和 IRM 说成同一件事。TF-mask 是推理修正，IRM 是训练监督。",
            "可以用一句话收住：Week6 的目标是让给定 label 后的 target 预测更准。",
        ],
    ),
    (
        "到 Week5 为止",
        "说明 Week6 的起点和边界：不换主干，不重做对照，只在 Week5 Mamba FT 上继续优化 target。",
        [
            "前几周已经完成了一个比较完整的路线。Week3 做 self-attention Mini-DiT baseline；Week4 把底层序列算子换成 Mamba-style SSM，并且在同一任务设置下取得更好的结果；Week5 在 Week4 的 Mamba checkpoint 上继续训练，加了 waveform L1，并补齐 target 和 residual 的音频输出。",
            "因此 Week6 的起点不是从零开始训练，而是 Week5 的 Mamba FT 模型。为了让比较干净，Week6 不再更换主干，也不重新展开 self-attention 对照，所有比较都固定 test split、1-step inference 和同一个 Week5 checkpoint。",
            "Week5 的 target SI-SDR 是 1.417 dB。Week6 要回答的问题就是：在这个基础上，mask 约束还能不能继续提升 target prediction。",
        ],
    ),
    (
        "TF-mask 后处理怎么做",
        "解释 TF-mask 的计算链条：预测幅度谱 -> 除以 mixture 幅度得到 mask -> clip -> 乘回 mixture 幅度 -> iSTFT。",
        [
            "这一页是 TF-mask 的具体操作。Week5 模型的输入是 mixture 的频谱和 target label，输出是 target 的幅度谱预测。这个预测本身还没有显式保证它是 mixture 的一部分。",
            "TF-mask 的做法是，把预测 target magnitude 除以 mixture magnitude，得到一个 time-frequency mask。直观理解就是：在每个时频点上，模型认为有多少能量应该分给 target。",
            "然后我们把这个 mask 限制在 0 到 1 之间，也就是 clip。这样可以防止模型给 target 分配超过 mixture 的幅度能量。最后用这个 mask 乘回 mixture magnitude，并继续使用 mixture phase 做 iSTFT，得到新的 target waveform。",
            "这一步不改变模型权重，也没有训练；它只是推理后的输出修正。residual 仍然用 mixture 减去 target 得到。",
        ],
        [
            "可以强调：TF-mask 是工程先验，不是严格物理定律。相位抵消时 target magnitude 可能超过 mixture magnitude，所以后面要用实验验证它是否真的有效。",
        ],
    ),
    (
        "文献依据",
        "说明 Week6 不是随手加 trick，而是沿着频谱 mask、mixture consistency、Wiener refinement 这一类源分离思路做轻量补充。",
        [
            "这一页主要是说明方法来源。传统频谱域源分离里，mask 是非常常见的建模方式。IRM，也就是 ideal ratio mask，是一种理想比例 mask，经常作为监督目标或者上界分析使用。",
            "Phase-sensitive mask 和 complex ratio mask 说明了一件事：只看幅度并不总是足够的，phase 会影响最终波形质量。我们现在仍使用 mixture phase，所以这个限制需要在结论里说明。",
            "Mixture consistency 的思想是让输出和 mixture 对齐；Wiener refinement 则是源分离系统里常见的软 mask 后处理。Week6 的 TF-mask 和 IRM 微调，就是在我们现有 Mini-DiT/Mamba 框架上引入这些轻量约束。",
        ],
    ),
    (
        "TF-mask 候选怎么选",
        "说明候选公式都属于推理阶段后处理，模型权重不变；最终选择 clip 是因为简单且稳定。",
        [
            "这里不是又开了一条新的主线，而是在 TF-mask 后处理内部比较几个候选公式。所有候选都只发生在推理阶段，模型权重不变。",
            "我们比较了 Week5 direct，也就是不加 mask 的原始输出；TF-mask clip，也就是最简单的有界 mask；Wiener p=1，作为 soft mask 的对照；以及 Wiener p=2，它更偏向功率谱比例，抑制会更强。",
            "选择规则也很简单：在 full test split 上，用同样的 1-step inference，主要看 target SI-SDR 和 Wave L1。结果发现 clip 和 Wiener p=1 很接近，所以优先选择更简单的 clip；Wiener p=2 虽然有些频谱指标好看，但 SI-SDR 明显下降，因此不作为主结果。",
            "Oracle IRM 也会计算，但它使用真实 target 和 residual，因此只能作为上界诊断，不能当成模型结果。",
        ],
    ),
    (
        "后处理结果：指标图",
        "先从图上讲趋势：简单后处理小幅提升，Oracle IRM 上界很高。",
        [
            "这一页先看图上的趋势。蓝色柱子是实际模型或后处理结果，金色柱子是 Oracle IRM，只作为上界诊断。",
            "从图上可以看到，TF-mask clip 和 Wiener p=1 相比 Week5 direct 有小幅提升，但提升不大。Wiener p=2 在某些频谱指标上看起来更低，但 target SI-SDR 反而变差。",
            "最关键的是 Oracle IRM 的上界很高。这说明如果 mask 估计足够好，理论上 target 分离质量还有明显空间。",
        ],
    ),
    (
        "后处理结果：数值表",
        "用具体数值说明：TF-mask 后处理有效但收益小，因此需要进入训练侧约束。",
        [
            "从数值表看，Week5 direct 的 target SI-SDR 是 1.417 dB，TF-mask clip 后变成 1.431 dB，提升大约 0.013 dB。这个提升很小，但方向是正的。",
            "Wiener p=1 和 clip 基本一致，说明简单有界 mask 已经覆盖了这类后处理的大部分收益。Wiener p=2 的 target SI-SDR 下降到 0.422 dB，因此不作为主结果。",
            "这个表给出的结论是：后处理本身有效，但效果有限。所以 Week6 后面不只停留在推理修正，而是把 mask 约束加入训练目标。",
        ],
    ),
    (
        "为什么进入 IRM 微调",
        "用后处理收益小 + Oracle 上界高解释训练必要性。",
        [
            "这一页回答为什么要继续训练。单纯后处理的收益只有 0.013 dB 左右，说明只在推理后做 clip 修正，能解决的问题很有限。",
            "但是 Oracle IRM 的上界非常高，比 direct 高大约 13.45 dB。这说明瓶颈不完全在后处理公式本身，而在模型是否学到了合理的 mask 结构。",
            "所以我们采用一个很稳妥的策略：不推翻原来的 flow matching 和 Mamba 主干，只是在 Week5 loss 后面加一个 IRM L1 辅助项。这样模型仍然做原来的 target prediction，但同时被鼓励预测出更接近理想比例 mask 的结果。",
        ],
    ),
    (
        "IRM 微调怎么做",
        "说明训练起点、IRM target、预测 mask 和 total loss。",
        [
            "IRM 微调从 Week5 Mamba FT 的 best checkpoint 开始，主干、输入和 1-step inference 设置都不变。训练 5 个 epoch，学习率是 3e-5。",
            "IRM target 是用真实 target magnitude 和 residual magnitude 算出来的理想比例 mask，公式是 target magnitude 平方除以 target magnitude 平方加 residual magnitude 平方。",
            "模型侧的预测 mask 则来自预测 target magnitude 除以 mixture magnitude，然后 clip 到 0 到 1。IRM L1 就是预测 mask 和理想 mask 的 L1 距离。",
            "最终 loss 是原来的 flow loss，加上 0.1 倍 waveform loss，再加上 0.1 倍 IRM loss。也就是说，我们没有替换原目标，而是在原目标上加了一个与源分离经验更一致的辅助约束。",
        ],
        [
            "这里可以把 IRM 解释成：让模型学会在 mixture 的每个时频点上，应该分给 target 多少比例。",
        ],
    ),
    (
        "云端训练情况",
        "说明训练完成、验证集最佳在 epoch 2，最终用 best checkpoint 做 test。",
        [
            "训练总共跑了 5 个 epoch。验证集最优出现在 epoch 2，当时 val loss 是 0.02887，val IRM L1 是 0.08558。",
            "后面的 epoch 中，训练 loss 仍然有下降，但验证集没有稳定继续变好，所以最终采用 epoch 2 的 best checkpoint 进入 test 评估。",
            "这也说明我们这次微调不是追求更长训练，而是做一个轻量的、有约束的 continued training。",
        ],
    ),
    (
        "Week6 最终结果",
        "先从图上说明最终趋势：正式结果看蓝色方法，Oracle 只看上界。",
        [
            "这一页是 Week6 最终结果图。这里仍然要提醒：金色的 Oracle IRM 只是上界诊断，不是模型结果；正式模型结果看蓝色方法。",
            "蓝色方法里，Week6 IRM FT 加 TF-mask 是最终推荐结果。图上可以看到，它在 target SI-SDR 和 Wave L1 上都优于 Week5 Mamba FT。",
            "所以图上的总体结论是：单纯 TF-mask 后处理只有小幅收益，而把 IRM 加入训练以后，target prediction 有更明显提升。",
        ],
    ),
    (
        "主结果表",
        "用主表给出最终数值：Week6 best 相比 Week5 的 target SI-SDR 提升约 1.12 dB。",
        [
            "主结果表给出最终可引用的数值。Week5 Mamba FT 的 target SI-SDR 是 1.417 dB；Week6 TF-mask only 是 1.431 dB；Week6 IRM FT 加 TF-mask 达到 2.535 dB。",
            "也就是说，Week6 best 相比 Week5 提升约 1.12 dB。target Wave L1 从 0.02721 降到 0.02663，Spec L1 也从 0.03097 降到 0.03064。",
            "这里 residual SI-SDR 没有同步提升，所以汇报时不要说二源整体全面提升。更准确的结论是：Week6 明确提升了给定 label 后的 target 预测质量。",
        ],
    ),
    (
        "结果分析",
        "解释为什么 target 改善、residual 没同步改善，以及为什么评价重点放在 target。",
        [
            "结果里有一个需要说明的点：target 指标明显提升，但 residual SI-SDR 下降了。这不是矛盾，因为本周训练目标主要约束 target，而 residual 仍然是 mixture 减 target 得到的，并不是模型显式预测的第二个输出。",
            "在我们的数据构造里，每个 mixture 会拆成两条 target record。也就是说，同一个混合音频里的两个声源可以轮流作为 target 输入 label 来预测。因此正式评价可以重点看给定 label 后的 target prediction 质量。",
            "residual 现在更适合作为试听和 mixture consistency 检查。如果后续真的要优化 residual，需要双输出头或者显式 residual loss，而不是只靠 target subtraction。",
        ],
    ),
    (
        "Week6 小结",
        "总结 Week6 贡献：TF-mask 小幅有效，Oracle 提供上界，IRM 微调带来实际提升。",
        [
            "总结一下 Week6。第一，简单的 TF-mask 后处理带来小幅稳定增益，说明 mixture magnitude 约束是有用的。第二，Oracle IRM 显示理论空间很大，说明 mask 方向值得进一步学习。第三，加入 IRM auxiliary loss 后，target 质量明显提升。",
            "最终 target SI-SDR 从 1.417 dB 提升到 2.535 dB，target Wave L1 从 0.02721 降到 0.02663。这个结果说明，在 Week5 Mamba FT 的基础上，引入 mask 约束可以进一步改善目标声源预测。",
            "同时我们也保留边界：这次没有实现 residual 的全面提升，残差联合建模仍然是后续工作。",
        ],
    ),
    (
        "Week7 做什么",
        "说明 Week7 不再训练，转向最终报告、展示稿、复现材料和边界总结。",
        [
            "Week7 不再新增模型训练，重点是最终整理。我们需要把 Week3 到 Week6 的指标统一放进最终主表，报告里讲清楚 Mamba、waveform loss、TF-mask 和 IRM loss 之间的关系。",
            "最终展示稿要控制篇幅，把重点放在结果和边界。复现材料需要整理好路径、checkpoint、评估命令和试听页入口。",
            "后续工作可以写 target/residual 双输出头，或者显式 residual loss。这样可以解释为什么 Week6 target 提升明显，但 residual 仍然存在局限。",
        ],
    ),
]


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    configure_document(doc)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run("Week6 汇报文字稿")
    r.bold = True
    r.font.size = Pt(22)
    r.font.color.rgb = RGBColor.from_string("0B2545")
    title.paragraph_format.space_after = Pt(4)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.add_run("TF-mask 后处理与 IRM 辅助微调").italic = True
    subtitle.paragraph_format.space_after = Pt(10)

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.add_run("深度学习自选题小组 潘羽｜2026 年 6 月 5 日")

    add_note_box(
        doc,
        "总口径",
        [
            "Week6 固定使用 Week5 Mamba FT，不再改主干。",
            "TF-mask 是推理后修正输出，不训练；IRM 微调是在训练 loss 中加入 mask 监督。",
            "正式结论聚焦 label-conditioned target prediction：Week6 best 的 target SI-SDR 从 1.417 dB 提升到 2.535 dB。",
            "residual 仍由 mixture - prediction_target 得到，不作为 Week6 主优化目标。",
        ],
    )

    doc.add_heading("一、开场讲法", level=1)
    doc.add_paragraph(
        "大家好，我们本周汇报 Week6 的工作。前面 Week5 已经有一个 Mamba-style Mini-DiT 分离器，能够根据 mixture 和 target label 预测目标声源。Week6 没有重新换主干，也没有重新展开 self-attention 对照，而是在 Week5 模型的基础上引入 mask 约束，目标是进一步提升 target prediction 的质量。"
    )
    doc.add_paragraph(
        "本周最容易混淆的是 TF-mask 和 IRM 微调。TF-mask 是推理后的输出修正，不训练模型；IRM 微调是训练时加入 mask 监督，让模型学会更合理地分配 mixture 中的时频能量。"
    )

    doc.add_heading("二、逐页讲稿", level=1)
    for idx, (title_text, target, script, *rest) in enumerate(slides, start=1):
        notes = rest[0] if rest else None
        add_slide_script(doc, idx, title_text, target, script, notes)

    doc.add_heading("三、可能被问到的问题", level=1)
    qa = [
        (
            "TF-mask 和 IRM 是不是一回事？",
            "不是。TF-mask 是推理阶段的后处理，模型权重不变；IRM 微调是训练阶段加入一个 mask L1 监督项，会更新模型权重。",
        ),
        (
            "为什么要先做后处理，再做 IRM 微调？",
            "先做后处理是低成本诊断。如果简单 mask 就能大幅提升，就没必要训练；实际后处理只小幅提升，但 Oracle IRM 上界很高，所以说明值得把 mask 约束放进训练。",
        ),
        (
            "为什么 residual 指标没有同步提升？",
            "因为本周模型仍然只显式优化 target。residual 是 mixture 减 target 得到的辅助输出，不是双输出模型直接预测出来的。数据里两个声源可以轮流作为 target，因此正式评价重点放在给定 label 后的 target prediction。",
        ),
        (
            "Week7 还会训练吗？",
            "不会。Week6 是最后一次模型训练，Week7 只做最终报告、展示稿、复现材料和结论边界整理。",
        ),
    ]
    add_table(doc, ["问题", "建议回答"], qa, [2600, 6760])

    doc.add_heading("四、结束语", level=1)
    doc.add_paragraph(
        "综上，Week6 的核心结果是：在 Week5 Mamba FT 的基础上，简单 TF-mask 后处理带来小幅提升，而 IRM auxiliary loss 的 continued training 带来了更明显的 target 指标改善。最终 Week6 best 的 target SI-SDR 达到 2.535 dB，比 Week5 提升约 1.12 dB。这个结果说明，在当前轻量 Mini-DiT/Mamba 框架下，mask 约束是有效的后续优化方向。"
    )

    doc.core_properties.title = "Week6 汇报文字稿"
    doc.core_properties.subject = "TF-mask 后处理与 IRM 辅助微调"
    doc.core_properties.author = "深度学习自选题小组 潘羽"
    doc.save(OUT)


if __name__ == "__main__":
    main()
