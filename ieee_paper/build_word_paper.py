from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(r"C:\Users\Aryavart\NLP_Project")
OUTDIR = ROOT / "ieee_paper"
FIGDIR = ROOT / "results_final"
OUT = OUTDIR / "Multimodal_Deepfake_Detection_IEEE.docx"
CHART = OUTDIR / "baseline_comparison.png"

def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

def make_baseline_chart():
    W, H = 1500, 800
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    f_title = font(r"C:\Windows\Fonts\arial.ttf", 36)
    f_label = font(r"C:\Windows\Fonts\arial.ttf", 27)
    f_small = font(r"C:\Windows\Fonts\arial.ttf", 22)
    d.text((70, 35), "AUC on FakeAVCeleb: published references and this study", fill="#111827", font=f_title)
    vals = [
        ("Published Multimodal-1", 0.533, "#8A98A8"),
        ("Published Multimodal-2", 0.688, "#8A98A8"),
        ("Published CDCN", 0.662, "#8A98A8"),
        ("Published EfficientNet ensemble", 0.828, "#8A98A8"),
        ("Published VGG audio", 0.978, "#8A98A8"),
        ("Published EfficientNet video", 0.933, "#8A98A8"),
        ("This study: fused OOF", 0.900, "#176B87"),
    ]
    left, right = 470, 1330
    top, row_h, bar_h = 135, 84, 38
    d.line((left, top-20, left, top + row_h*len(vals)-20), fill="#667085", width=2)
    for tick in range(0, 11, 2):
        x = left + int((right-left)*tick/10)
        d.line((x, top-18, x, top + row_h*len(vals)-30), fill="#E5E7EB", width=2)
        d.text((x-20, top + row_h*len(vals)-22), f"{tick/10:.1f}", fill="#475467", font=f_small)
    for i, (label, value, color) in enumerate(vals):
        y = top + i*row_h
        d.text((left-22, y+3), label, fill="#1F2937", font=f_label, anchor="rm")
        bw = int((right-left)*value)
        d.rounded_rectangle((left, y, left+bw, y+bar_h), radius=7, fill=color)
        d.text((left+bw+12, y+4), f"{value:.3f}", fill="#111827", font=f_label)
    d.text((left, H-38), "AUC", fill="#475467", font=f_small)
    im.save(CHART, dpi=(240, 240))

def set_columns(section, count):
    cols = section._sectPr.find(qn("w:cols"))
    if cols is None:
        cols = OxmlElement("w:cols")
        section._sectPr.append(cols)
    cols.set(qn("w:num"), str(count))
    cols.set(qn("w:space"), "360")

def shade(cell, color):
    pr = cell._tc.get_or_add_tcPr()
    shd = pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        pr.append(shd)
    shd.set(qn("w:fill"), color)

def set_cell_borders(cell, color="D9D9D9", size="6"):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    borders = tcPr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tcPr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "w:" + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)

def set_cell_margins(cell, top=70, start=85, bottom=70, end=85):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = tcPr.first_child_found_in("w:tcMar")
    if tcMar is None:
        tcMar = OxmlElement("w:tcMar")
        tcPr.append(tcMar)
    for m, v in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tcMar.find(qn("w:" + m))
        if node is None:
            node = OxmlElement("w:" + m)
            tcMar.append(node)
        node.set(qn("w:w"), str(v))
        node.set(qn("w:type"), "dxa")

def format_table(table, widths=None, header=True):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for ri, row in enumerate(table.rows):
        for ci, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if widths and ci < len(widths):
                cell.width = Inches(widths[ci])
            set_cell_margins(cell)
            set_cell_borders(cell)
            if header and ri == 0:
                shade(cell, "254B62")
            elif ri % 2 == 0:
                shade(cell, "F2F6F8")
            for p in cell.paragraphs:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if ci == 0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.space_before = Pt(0)
                for run in p.runs:
                    run.font.name = "Times New Roman"
                    run.font.size = Pt(7.4)
                    if header and ri == 0:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(255, 255, 255)

def set_repeat_table_header(row):
    trPr = row._tr.get_or_add_trPr()
    tblHeader = OxmlElement("w:tblHeader")
    tblHeader.set(qn("w:val"), "true")
    trPr.append(tblHeader)

def add_table(doc, caption, headers, rows, widths):
    p = doc.add_paragraph()
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(3)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(caption)
    r.font.name = "Times New Roman"
    r.font.size = Pt(8)
    r.font.bold = True
    table = doc.add_table(rows=1, cols=len(headers))
    for i, head in enumerate(headers):
        table.rows[0].cells[i].text = head
    for data in rows:
        cells = table.add_row().cells
        for i, value in enumerate(data):
            cells[i].text = str(value)
    format_table(table, widths)
    set_repeat_table_header(table.rows[0])
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table

def body(doc, text, keep=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Inches(0.12)
    p.paragraph_format.space_after = Pt(2.2)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.widow_control = True
    if keep:
        p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(9)
    return p

def section_title(doc, roman, title):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    label = f"{roman}. {title.upper()}" if roman else title.upper()
    r = p.add_run(label)
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(10)
    return p

def sub_title(doc, label, title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(f"{label}. {title}")
    r.bold = True
    r.italic = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(9)
    return p

def add_figure(doc, path, caption, width=3.18):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(1)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(path), width=Inches(width))
    c = doc.add_paragraph()
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER
    c.paragraph_format.space_after = Pt(3)
    c.paragraph_format.keep_together = True
    r = c.add_run(caption)
    r.font.name = "Times New Roman"
    r.font.size = Pt(7.4)

def add_page_field(paragraph):
    run = paragraph.add_run()
    fldChar1 = OxmlElement("w:fldChar")
    fldChar1.set(qn("w:fldCharType"), "begin")
    instrText = OxmlElement("w:instrText")
    instrText.set(qn("xml:space"), "preserve")
    instrText.text = " PAGE "
    fldChar2 = OxmlElement("w:fldChar")
    fldChar2.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)

def build():
    make_baseline_chart()
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.top_margin, sec.bottom_margin = Inches(.62), Inches(.62)
    sec.left_margin, sec.right_margin = Inches(.7), Inches(.7)
    set_columns(sec, 1)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(9)
    normal.paragraph_format.space_after = Pt(2)
    normal.paragraph_format.line_spacing = 1.0

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(7)
    r = p.add_run("Complementary Voice and Face Cues for Multimodal Deepfake Detection")
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(17)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run("Aryavart Singh Payal")
    r.font.name = "Times New Roman"
    r.font.size = Pt(11)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run("CSET 346: Natural Language Processing | Project PJ_07")
    r.font.name = "Times New Roman"
    r.font.size = Pt(9)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(3)
    r = p.add_run("Abstract—")
    r.bold = True
    r.italic = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(9)
    r = p.add_run("Audio-video deepfakes may manipulate voice, face, or both, leaving single-stream detectors without relevant evidence. We evaluate a lightweight verifier on a 998-clip, speaker-disjoint subset of FakeAVCeleb v1.2. Frozen wav2vec 2.0 and CLIP representations feed separate logistic-regression classifiers; probabilities are averaged, and a validation-tuned abstention band produces REAL, FAKE, or UNCERTAIN. In pooled five-fold evaluation, voice-only, face-only, and fused ROC-AUC are 0.817, 0.746, and 0.900. The system answers 83.8% of clips at 86.6% accuracy among answered clips. Fusion improves ranking, transcripts do not improve detection, and face-swap-only clips remain difficult. We compare the baseline paper in both data composition and reported detector results, while stating where the protocols differ.")
    r.font.name = "Times New Roman"
    r.font.size = Pt(9)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(7)
    r = p.add_run("Index Terms—")
    r.bold = True
    r.italic = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(9)
    r = p.add_run("deepfake detection, audio-visual learning, multimodal fusion, abstention, explainability.")
    r.font.name = "Times New Roman"
    r.font.size = Pt(9)

    # Full-width title and abstract followed by IEEE-like two-column body.
    section = doc.add_section(WD_SECTION.CONTINUOUS)
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    section.top_margin, section.bottom_margin = Inches(.62), Inches(.62)
    section.left_margin, section.right_margin = Inches(.7), Inches(.7)
    set_columns(section, 2)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_field(footer)

    section_title(doc, "I", "Introduction")
    body(doc, "A video may contain a manipulated voice, face, or both. Voice-only detection is poorly suited to a face swap paired with genuine speech, while visual detection can miss cloned speech paired with an authentic face. We examine whether voice and face representations provide complementary evidence, how fusion methods handle conflicting evidence, and how an abstaining system can communicate uncertainty on FakeAVCeleb.")
    body(doc, "This study contributes a comparison of audio, face, transcript, and fused classifiers under speaker-disjoint evaluation; a comparison of early, late, and neural fusion; ablations and held-out manipulation-type tests; and an analysis of abstention, stream-level explanations, difficult cases, and observed demographic-group differences. We evaluate simple models on frozen representations rather than propose a new encoder.")

    section_title(doc, "II", "Baseline Paper and Data Comparison")
    body(doc, "Khalid et al. introduced FakeAVCeleb and benchmarked unimodal, ensemble, and multimodal detectors [1]. Their release description reports 20,000 clips in four audio-video combinations. Our experiment uses a balanced 998-clip subset from that release. Table I makes the sampling difference explicit; the source distribution is highly imbalanced, while our subset gives nearly equal real and fake totals and equal counts across the three fake categories.")
    add_table(doc, "TABLE I. DATA COMPOSITION: BASELINE RELEASE AND OUR EVALUATION SUBSET", ["Audio-video condition", "Full release [1]", "This study"], [
        ("Real video + real audio (RVRA)", "500", "500"),
        ("Real video + fake audio (RVFA)", "500", "166"),
        ("Fake video + real audio (FVRA)", "9,000", "166"),
        ("Fake video + fake audio (FVFA)", "10,000", "166"),
        ("Total clips", "20,000", "998"),
        ("Real / fake balance", "500 / 19,500", "500 / 498"),
    ], [1.48, .76, .82])
    body(doc, "The subset includes every RVRA clip and samples 166 examples from each fake category using seed 42. Thus, it preserves all four manipulation labels while substantially changing class proportions. Any comparison with results reported on the full release is affected by this difference in sample composition as well as by split and model differences.")
    body(doc, "For detection baselines, Khalid et al. report AUC values of 0.533, 0.688, and 0.662 for Multimodal-1, Multimodal-2, and CDCN. Their appendix also reports 0.828 for an EfficientNet-B0 ensemble, 0.978 for VGG audio, and 0.933 for EfficientNet-B0 video [1]. Our pooled speaker-disjoint AUC is 0.900 for voice-face fusion, 0.817 for voice, and 0.746 for face. These values are shown together in Fig. 1, but they are contextual, not head-to-head: the original paper does not report the same 998-clip sample or speaker-disjoint folds. In particular, the published unimodal scores are not a fair matched comparison to our subset.")
    add_figure(doc, CHART, "Fig. 1. Reported AUC values from Khalid et al. and pooled out-of-fold AUC from this study. Protocols are not matched.", 3.15)

    section_title(doc, "III", "Data and Methods")
    sub_title(doc, "A", "Dataset and split")
    body(doc, "FakeAVCeleb v1.2 contains celebrity videos with real and generated audio-video manipulations [1]. The selected subset contains 998 usable clips from 500 speakers: 500 real and 498 fake. Every fold is speaker-disjoint, so a speaker in a test fold is absent from that fold's training data. The subset rule, fixed random seed, feature exclusions, and category counts are reported to support reproducibility. The dataset is available for non-commercial research and education under CC BY 4.0 with download access by request [5]; source media are not redistributed.")
    sub_title(doc, "B", "Feature extraction and classifier")
    body(doc, "Audio is resampled to 16 kHz, limited to approximately ten seconds, and encoded with frozen wav2vec 2.0-base [2]. The time-pooled states are averaged across 13 layers, yielding a 768-dimensional vector. Three sampled frames are processed with a detected face crop or center-crop fallback and encoded by frozen CLIP ViT-B/32; averaged and renormalized embeddings have 512 dimensions [3]. Whisper-base produces English transcripts [4]. Transcript features are evaluated but do not enter the final score.")
    body(doc, "Each stream is standardized and scored by a class-balanced logistic regression. Regularization and thresholds are selected on speaker-disjoint validation data. The fused score is the mean of the voice and face probabilities: p_fuse = (p_voice + p_face) / 2. A validation-tuned cutoff gives REAL or FAKE. Scores in a band around the cutoff yield UNCERTAIN; the band targets approximately 15% abstention on validation data. The stream-level explanation is derived from the two scores and is not an independent causal explanation.")
    sub_title(doc, "C", "Evaluation protocols")
    body(doc, "We report pooled out-of-fold AUC for the final system using five speaker-disjoint folds. Coverage is the fraction receiving a REAL or FAKE decision; selective accuracy is computed among answered clips. Earlier fixed and repeated splits are reported separately for comparisons among fusion methods, and are not substituted for the pooled estimate. Ten alternative speaker partitions quantify fold sensitivity. Group audits use folder-derived demographic labels and are descriptive; no significance tests were run.")

    section_title(doc, "IV", "Results")
    sub_title(doc, "A", "Single-modality and fusion results")
    body(doc, "On the main fixed split, voice AUC is 0.825, face AUC is 0.797, CLIP transcript AUC is 0.620, and TF-IDF transcript AUC is 0.484. Early voice-face fusion reaches 0.925; adding transcript reduces it to 0.915. On repeated speaker-disjoint partitions, early concatenation reaches 0.937, late mean 0.922, and gated fusion with modality dropout 0.938. The best fusion scores differ by only a small margin, so the project retains a simpler late-fusion model for its final abstaining verifier.")
    body(doc, "The final pooled five-fold out-of-fold AUC is 0.900 for fusion, 0.817 for voice, and 0.746 for face. Across ten alternative speaker partitions, the Kaggle feature run yields 0.909 plus or minus 0.013 fused AUC. The former is the single seed-42 out-of-fold estimate; the latter describes variability across alternative partitions and should not be mixed with fixed-split fusion results.")
    add_table(doc, "TABLE II. PUBLISHED DETECTOR REFERENCES AND OUR OUT-OF-FOLD RESULTS", ["Model / stream", "AUC", "Source / protocol"], [
        ("Multimodal-1", "0.533", "Khalid et al. [1]"),
        ("Multimodal-2", "0.688", "Khalid et al. [1]"),
        ("CDCN", "0.662", "Khalid et al. [1]"),
        ("EfficientNet-B0 ensemble", "0.828", "Khalid et al. [1]"),
        ("VGG audio", "0.978", "Khalid et al. [1]"),
        ("EfficientNet-B0 video", "0.933", "Khalid et al. [1]"),
        ("Our voice", "0.817", "5-fold speaker-disjoint OOF"),
        ("Our face", "0.746", "5-fold speaker-disjoint OOF"),
        ("Our voice + face", "0.900", "5-fold speaker-disjoint OOF"),
    ], [1.23, .48, 1.35])
    body(doc, "The source paper reports its multimodal scores as AUC and presents its unimodal and ensemble values in AUC figures [1]. Table II provides the required numerical baseline comparison. Differences in data subset, feature extraction, and evaluation partitioning limit causal attribution of the performance gap; a direct comparison requires rerunning the published methods on our exact subset and speaker folds.")
    sub_title(doc, "B", "Ablation and manipulation types")
    body(doc, "In Experiment 4, voice-plus-face late fusion reaches AUC 0.941, versus 0.834 for voice, 0.786 for face, and 0.624 for transcript alone. Adding transcript lowers AUC to 0.927. Voice alone has only 0.036 accuracy on face-swap-only clips in this split. In leave-one-manipulation-type-out evaluation, early-fusion AUC falls from 0.93-0.95 on seen types to 0.76-0.78 on an unseen type. These results test held-out manipulation categories within FakeAVCeleb and do not establish cross-dataset generalization.")
    sub_title(doc, "C", "Abstention and per-category accuracy")
    body(doc, "The final system answers 83.8% of clips and is correct on 86.6% of answered clips. Counting UNCERTAIN as incorrect gives 72.5% accuracy over all clips. The 3-way confusion counts are 366 REAL and 57 FAKE verdicts for true-real clips, and 55 REAL and 358 FAKE verdicts for true-fake clips; 77 true-real and 85 true-fake clips are uncertain. Answered-clip accuracy is 99.2% for cloned-voice fakes, 56.1% for face-swap-only fakes, and 100% for combined manipulations. Selective accuracy must be read alongside coverage.")
    add_figure(doc, FIGDIR / "fig2_roc.png", "Fig. 2. Pooled out-of-fold ROC curves for voice, face, and fused scores.", 3.12)
    add_figure(doc, FIGDIR / "fig1_confusion_3way.png", "Fig. 3. Three-way out-of-fold confusion counts, including abstentions.", 3.12)
    add_figure(doc, FIGDIR / "fig3_verdict_by_category.png", "Fig. 4. REAL, FAKE, and UNCERTAIN verdicts by manipulation category.", 3.12)
    sub_title(doc, "D", "Explanation and sanity-check analysis")
    body(doc, "The explanation's stream flag matches the audio-manipulation label with 97.6% accuracy and the video-manipulation label with 77.2%; the exact four-way pattern is correct in 75.5%. Among 415 FAKE verdicts, removing the stream named in the explanation cancels 42.7% of decisions, compared with 0% when the other stream is removed. This supports score dependence under the deletion perturbation, not general causal faithfulness. Recombined genuine voice-face pairs from different speakers are called REAL in 70.6% of cases, showing that this verifier does not check identity consistency across streams.")
    sub_title(doc, "E", "Difficult cases")
    body(doc, "The out-of-fold error file contains 112 answered mistakes: 54 face-swap-only fakes and one cloned-voice fake are called REAL, while 57 real clips are called FAKE. We inspected the 15 highest-ranked scored errors at the feature level: 11 are face-swap-only false negatives and four are real-clip false positives. The common false-negative pattern is little usable facial evidence or a weaker face score offset by real-audio evidence. False positives generally show fake-like voice, face, or both scores. These examples are anonymized and the analysis does not claim visual review of the source videos.")
    sub_title(doc, "F", "Observed group differences")
    body(doc, "False-alarm rates range from 8.8% to 21.2% across ethnicity labels and from 9.1% to 17.7% across sex labels. East Asian speakers have 0.816 answer-only accuracy and a 20.9% miss rate; American Caucasian speakers have the highest observed false-alarm rate, 21.2%. These are descriptive results from one dataset, roughly 100 speakers per ethnicity category, with no significance tests. They should not be interpreted as population-level estimates.")
    add_figure(doc, FIGDIR / "fig5_bias_audit.png", "Fig. 5. False-alarm and miss rates by folder-derived demographic labels.", 3.12)
    add_figure(doc, FIGDIR / "fig6_feature_variants.png", "Fig. 6. Stability results for three-, eight-, and sixteen-frame feature variants.", 3.12)

    section_title(doc, "V", "Discussion")
    body(doc, "Voice and face evidence capture different manipulation types, and combining them improves ranking over either stream in the pooled evaluation. Simple fusion performs similarly to more complex heads in these experiments and provides a transparent reference on a small dataset. Averaging does not resolve all conflicting evidence, and face-swap-only clips remain difficult. Abstention sends 16.2% of clips for review while raising accuracy among answered clips. Transcript text does not improve the score on this dataset and feature pipeline.")
    body(doc, "The comparison with Khalid et al. is deliberately qualified. The baseline paper describes a much larger and highly imbalanced release and reports strong standalone audio and visual baselines alongside weaker multimodal methods. Our subset is balanced and our speaker-disjoint split is stricter against speaker leakage; neither fact alone explains the score gap. A controlled replication would use the same sampled videos, folds, and metrics for all methods.")

    section_title(doc, "VI", "Limitations and Ethics")
    body(doc, "The study uses one dataset and a selected subset. Temporal pooling and three averaged frames discard fine-grained cues; CLIP is a semantic image encoder rather than a forensic face detector. Cross-dataset robustness and calibrated confidence are untested. No human evaluation of explanation usefulness was performed, and the feature-level case audit does not replace a manual review of source media. The observed demographic gaps are not significance-tested.")
    body(doc, "FakeAVCeleb contains real people's faces and voices and requires controlled research access. Its official license page specifies non-commercial research and education use under CC BY 4.0, with downloads by request [5]. The project should retain evidence that its team accepted the applicable access terms. A false FAKE verdict can discredit genuine media, so this prototype should support human review rather than serve as evidence of authenticity or falsity. Source videos and identifying metadata should not be redistributed.")

    section_title(doc, "VII", "Conclusion")
    body(doc, "On a speaker-disjoint FakeAVCeleb subset, pooled voice-face fusion achieves 0.900 AUC, above voice-only (0.817) and face-only (0.746) scores. The system answers 83.8% of clips at 86.6% answer-only accuracy, with persistent weakness on face-swap-only examples. We compare the study with the original FakeAVCeleb paper in dataset size, category balance, and reported detector AUCs, while making clear that the protocols are not matched. Cross-dataset evaluation, stronger forensic visual features, and human evaluation of explanations remain important next steps.")

    section_title(doc, "", "References")
    refs = [
        "[1] H. Khalid, S. Tariq, M. Kim, and S. S. Woo, \"FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset,\" in Proc. NeurIPS Datasets and Benchmarks Track, 2021; arXiv:2108.05080v4, 2022.",
        "[2] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, \"wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations,\" in Advances in Neural Information Processing Systems, vol. 33, 2020.",
        "[3] A. Radford et al., \"Learning Transferable Visual Models From Natural Language Supervision,\" in Proc. Int. Conf. Machine Learning, 2021, pp. 8748-8763.",
        "[4] A. Radford et al., \"Robust Speech Recognition via Large-Scale Weak Supervision,\" in Proc. Int. Conf. Machine Learning, vol. 202, 2023, pp. 28492-28518.",
        "[5] DASH Lab, \"FakeAVCeleb License.\" [Online]. Available: https://sites.google.com/view/fakeavcelebdash-lab/license. Accessed: Nov. 25, 2026.",
    ]
    for ref in refs:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.left_indent = Inches(.17)
        p.paragraph_format.first_line_indent = Inches(-.17)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(ref)
        r.font.name = "Times New Roman"
        r.font.size = Pt(7.8)

    doc.core_properties.title = "Complementary Voice and Face Cues for Multimodal Deepfake Detection"
    doc.core_properties.subject = "IEEE-style research paper comparing multimodal results with FakeAVCeleb baselines"
    doc.core_properties.author = "Aryavart Singh Payal"
    doc.save(OUT)
    print(OUT)

if __name__ == "__main__":
    build()
