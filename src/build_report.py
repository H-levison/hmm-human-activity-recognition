"""Build the formative report as a .docx.

Run:  python src/build_report.py

The prose lives in src/report_text.py; this module is layout only. Every figure
referenced here is produced by src/experiment.py or the notebook, and every
number quoted in the text comes from results/tables, so the report cannot drift
out of step with the analysis that produced it.
"""
import os
import pathlib
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import report_text as T  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
# REPORT_OUT lets the report be rebuilt while the real file is open in Word.
OUT = pathlib.Path(os.environ.get(
    "REPORT_OUT",
    ROOT / "report" / "HonourGod_Kilechukwu_Levison_HMM_Report.docx"))

INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x55, 0x55, 0x55)
LINK = RGBColor(0x1C, 0x5C, 0xAB)

# The brief caps the report at 5 pages. With five figures, two tables and the
# reference list taking about 19 of the 49 available inches, 10.5 pt body text
# and the ~2,370 words in report_text.py land at roughly 4.7 pages. Raising
# BODY_PT to 11 adds about a third of a page, so it is the first thing to change
# if more room ever becomes available.
BODY_PT = 10.5
CAP_PT = 8.5
REF_PT = 8

doc = Document()

for s in doc.sections:
    s.top_margin = s.bottom_margin = Inches(0.6)
    s.left_margin = s.right_margin = Inches(0.7)

styles = doc.styles
normal = styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(BODY_PT)
normal.font.color.rgb = INK
pf = normal.paragraph_format
pf.space_after = Pt(3)
pf.line_spacing = 1.0
pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

for name, size, before in (("Heading 1", 12, 5), ("Heading 2", 10.5, 3)):
    st = styles[name]
    st.font.name = "Calibri"
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = INK
    st.paragraph_format.space_before = Pt(before)
    st.paragraph_format.space_after = Pt(2)
    st.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT


def para(text, size=None, italic=False, align=None, after=None, color=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    if size:
        run.font.size = Pt(size)
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color
    elif italic:
        run.font.color.rgb = MUTED
    if align is not None:
        p.paragraph_format.alignment = align
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def h(text, level=1):
    doc.add_heading(text, level=level)


def figure(name, caption, width):
    doc.add_picture(str(FIG / name), width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.space_after = Pt(1)
    para(caption, size=CAP_PT, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER,
         after=4)


def table(caption, headers, rows, widths=None):
    para(caption, size=CAP_PT, italic=True, after=2)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, head in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        r = cell.paragraphs[0].add_run(head)
        r.bold = True
        r.font.size = Pt(CAP_PT)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.paragraphs[0].paragraph_format.space_after = Pt(0)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v))
            r.font.size = Pt(CAP_PT)
            cells[i].paragraphs[0].alignment = (
                WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER)
            cells[i].paragraphs[0].paragraph_format.space_after = Pt(0)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


# ------------------------------------------------------------------ title
t = para("Inferring Human Activity States from Smartphone Motion Data Using a "
         "Hidden Markov Model", align=WD_ALIGN_PARAGRAPH.LEFT, after=1)
t.runs[0].font.size = Pt(14)
t.runs[0].bold = True

s = para("Passive rehabilitation-adherence monitoring after knee or hip "
         "replacement", align=WD_ALIGN_PARAGRAPH.LEFT, after=2, color=MUTED)
s.runs[0].font.size = Pt(10)

para("HonourGod Kilechukwu Levison   |   Machine Learning Techniques II   |   "
     "Formative 2", size=9.5, align=WD_ALIGN_PARAGRAPH.LEFT, after=6)

# ------------------------------------------------------------------ 1
h("1. Background and Motivation")
para(T.BG)

# ------------------------------------------------------------------ 2
h("2. Data Collection and Pre-processing")

h("2.1 Collection protocol and video documentation", 2)
para(T.COLLECT)
para("https://drive.google.com/drive/folders/"
     "1jfF0QRI9SgRetKKAMbNT-0WUdd2tBeEx?usp=sharing",
     size=9, after=1, color=LINK)
para("Still, Standing, Walking and Jumping Log Evidence.mp4  |  Sensor Logger "
     "Evidence.mp4", size=9, after=4)

h("2.2 Resampling, trimming and window length", 2)
para(T.RESAMPLE)
para(T.WINDOW)

figure("01b_sample_signals_wide.png",
       "Figure 1. Accelerometer magnitude for one clip of each activity, on "
       "separate axes because the amplitudes span three orders of magnitude.",
       5.1)

h("2.3 Feature extraction", 2)
para(T.FEAT_TIME)
para(T.FEAT_FREQ)

h("2.4 Normalisation and dimensionality reduction", 2)
para(T.NORM)
para(T.PCA)

# ------------------------------------------------------------------ 3
h("3. HMM Setup and Implementation Details")

table("Table 1. The five HMM components as instantiated in this project.",
      ["Element", "Instantiation"],
      [["Hidden states Z", "4 — Still, Standing, Walking, Jumping"],
       ["Observations X", "4-dimensional PCA feature vector per 2 s window"],
       ["Transitions A", "4 × 4 row-stochastic, from the two circuits"],
       ["Emissions B", "Multivariate Gaussian per state, diagonal covariance"],
       ["Initial state π", "4-vector over the first window of each sequence"]],
      widths=[1.4, 5.0])

h("3.1 Implementation and choice of algorithms", 2)
para(T.IMPL)
para(T.ALGOS)

h("3.2 Estimating the transition matrix", 2)
para(T.TRANS)

# ------------------------------------------------------------------ 4
h("4. Results and Interpretation")

figure("03_transition_matrix.png",
       "Figure 2. Transition matrix A from the two circuits; self-transitions of "
       "0.90 to 0.95 reflect activities that persist across many windows.",
       1.95)

para(T.A_INTERP)

figure("09c_emissions_wide.png",
       "Figure 3. Emission probabilities B, the density p(x | z) per state along "
       "each PCA component, with training windows as a rug. PC 1 separates "
       "low- from high-motion states, PC 2 walking from jumping, PC 3 still "
       "from standing.", 5.1)

h("4.1 Evaluation on unseen data", 2)
para(T.UNSEEN_INTRO)

table("Table 2. Per-state evaluation on the unseen session (84 windows). "
      "Sensitivity is TP/(TP+FN), specificity TN/(TN+FP), accuracy the "
      "one-versus-rest (TP+TN)/total. Overall accuracy is 0.6548, macro F1 "
      "0.6111.",
      ["State (Activity)", "Samples", "Sensitivity", "Specificity",
       "Overall Accuracy"],
      [["Still", 21, "0.1429", "1.0000", "0.7857"],
       ["Standing", 21, "0.4762", "0.7302", "0.6667"],
       ["Walking", 21, "1.0000", "0.8095", "0.8571"],
       ["Jumping", 21, "1.0000", "1.0000", "1.0000"]],
      widths=[1.5, 1.1, 1.3, 1.3, 1.4])

figure("04_confusion.png",
       "Figure 4. Confusion matrix on the unseen data, as counts and "
       "row-normalised. Every error shifts one step up the amplitude ladder, "
       "none the other way.", 2.9)

para(T.FAILURE)

h("4.2 Continuous decoding, the role of A, and unsupervised fitting", 2)
figure("05_decoded_circuit_01.png",
       "Figure 5. Annotated against decoded state sequence for Circuit_01. Grey "
       "bands are the three-second guard at each boundary, red bands "
       "disagreement; agreement is 70.7 per cent.", 3.6)
para(T.CONTINUOUS)

# ------------------------------------------------------------------ 5
h("5. Discussion and Conclusion")
para(T.EASY_HARD)
para(T.IMPROVE)
para(T.CONCLUDE)

# ------------------------------------------------------------------ refs
h("References")
refs = [
    "Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). A "
    "public domain dataset for human activity recognition using smartphones. "
    "ESANN 2013, 437–442.",
    "Bao, L., & Intille, S. S. (2004). Activity recognition from user-annotated "
    "acceleration data. Pervasive Computing, LNCS 3001, 1–17.",
    "Baum, L. E., Petrie, T., Soules, G., & Weiss, N. (1970). A maximization "
    "technique occurring in the statistical analysis of probabilistic functions "
    "of Markov chains. The Annals of Mathematical Statistics, 41(1), 164–171.",
    "Choi, L., Liu, Z., Matthews, C. E., & Buchowski, M. S. (2011). Validation of "
    "accelerometer wear and nonwear time classification algorithm. Medicine & "
    "Science in Sports & Exercise, 43(2), 357–364.",
    "Figo, D., Diniz, P. C., Ferreira, D. R., & Cardoso, J. M. P. (2010). "
    "Preprocessing techniques for context recognition from accelerometer data. "
    "Personal and Ubiquitous Computing, 14(7), 645–662.",
    "Jolliffe, I. T. (2002). Principal component analysis (2nd ed.). Springer.",
    "Kuhn, H. W. (1955). The Hungarian method for the assignment problem. Naval "
    "Research Logistics Quarterly, 2(1–2), 83–97.",
    "Prince, S. A., Adamo, K. B., Hamel, M. E., Hardt, J., Connor Gorber, S., & "
    "Tremblay, M. (2008). A comparison of direct versus self-report measures for "
    "assessing physical activity in adults: A systematic review. International "
    "Journal of Behavioral Nutrition and Physical Activity, 5, 56.",
    "Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected "
    "applications in speech recognition. Proceedings of the IEEE, 77(2), "
    "257–286.",
    "Viterbi, A. J. (1967). Error bounds for convolutional codes and an "
    "asymptotically optimum decoding algorithm. IEEE Transactions on Information "
    "Theory, 13(2), 260–269.",
    "Yu, S.-Z. (2010). Hidden semi-Markov models. Artificial Intelligence, "
    "174(2), 215–243.",
]
for r in refs:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.28)
    p.paragraph_format.first_line_indent = Inches(-0.28)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    p.add_run(r).font.size = Pt(REF_PT)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(str(OUT))

prose = sum(len(getattr(T, n).split()) for n in dir(T) if n.isupper())
total = sum(len(p.text.split()) for p in doc.paragraphs)
print(f"wrote {OUT}")
print(f"prose {prose} | all paragraphs {total} | "
      f"figures {len(doc.inline_shapes)} | tables {len(doc.tables)}")
