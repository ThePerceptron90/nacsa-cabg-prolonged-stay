import json, os, re, sys
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_LINE_SPACING, WD_BREAK
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
sys.path.insert(0, ".")
import ms_text as M

R = {k: v[0] for k, v in json.load(open("keyrefs.json")).items()}
L = {
 "L1": "Almashrafi A, Alsabti H, Mukaddirov M, Baskaran B, Aylin P. Factors associated with prolonged length of stay following cardiac surgery in a major referral hospital in Oman: a retrospective observational study. BMJ Open 2016;6(6):e010764. doi:10.1136/bmjopen-2015-010764",
 "L3": "Diab M, Bilkhu R, Soppa G, McGale N, Hirani SP, Newman SP, et al. Quality of life in relation to length of intensive care unit stay after cardiac surgery. J Cardiothorac Vasc Anesth 2017;31(3):1080-90. doi:10.1053/j.jvca.2016.06.025",
 "L5": "Nilsson J, Algotsson L, Hoglund P, Luhrs C, Brandt J. EuroSCORE predicts intensive care unit stay and costs of open heart surgery. Ann Thorac Surg 2004;78(5):1528-35. doi:10.1016/j.athoracsur.2004.04.060",
 "L6": "Shahian DM, Jacobs JP, Badhwar V, Kurlansky PA, Furnary AP, Cleveland JC Jr, et al. The Society of Thoracic Surgeons 2018 adult cardiac surgery risk models: part 1 - background, design considerations, and model development. Ann Thorac Surg 2018;105(5):1411-8. doi:10.1016/j.athoracsur.2018.03.002",
 "L7": "Miles TJ, Ghanta RK. Machine learning in cardiac surgery: a narrative review. J Thorac Dis 2024;16(4):2644-53. doi:10.21037/jtd-23-1659",
 "L8": "Benedetto U, Dimagli A, Sinha S, Cocomello L, Gibbison B, Caputo M, et al. Machine learning improves mortality risk prediction after cardiac surgery: systematic review and meta-analysis. J Thorac Cardiovasc Surg 2022;163(6):2075-87. doi:10.1016/j.jtcvs.2020.07.105",
 "L10": "Ettema RG, Peelen LM, Schuurmans MJ, Nierich AP, Kalkman CJ, Moons KG. Prediction models for prolonged intensive care unit stay after cardiac surgery: systematic review and validation study. Circulation 2010;122(7):682-9. doi:10.1161/CIRCULATIONAHA.109.926808",
 "L11": "Osnabrugge RL, Speir AM, Head SJ, Jones PG, Ailawadi G, Fonner CE, et al. Prediction of costs and length of stay in coronary artery bypass grafting. Ann Thorac Surg 2014;98(4):1286-93. doi:10.1016/j.athoracsur.2014.05.073",
 "L12": "Couto RC, Pedrosa T, Seara LM, Couto VS, Couto CS. Development of a machine learning model to estimate length of stay in coronary artery bypass grafting. Rev Saude Publica 2024;58:41. doi:10.11606/s1518-8787.2024058006161",
 "L14": "Alshakhs F, Alharthi H, Aslam N, Khan IU, Elasheri M. Predicting postoperative length of stay for isolated coronary artery bypass graft patients using machine learning. Int J Gen Med 2020;13:751-62. doi:10.2147/IJGM.S250334",
 "L16": "Fottinger A, Bader Eddeen A, Lee DS, Woodward G, Sun LY. Derivation and validation of pragmatic clinical models to predict hospital length of stay after cardiac surgery in Ontario, Canada: a population-based cohort study. CMAJ Open 2023;11(1):E180-90. doi:10.9778/cmajo.20220103",
 "L17": "Nashef SAM, Roques F, Sharples LD, Nilsson J, Smith C, Goldstone AR, et al. EuroSCORE II. Eur J Cardiothorac Surg 2012;41(4):734-45. doi:10.1093/ejcts/ezs043",
 "L18": "Richens D. Cardiothoracic Surgery: GIRFT Programme National Specialty Report. London: NHS Improvement; 2018.",
 "L19": "National Institute for Cardiovascular Outcomes Research. National Adult Cardiac Surgery Audit (NACSA) 2021 Summary Report. London: Healthcare Quality Improvement Partnership; 2021.",
 "L21": "Hickey GL, Grant SW, Murphy GJ, Bhabra M, Pagano D, McAllister K, et al. Dynamic trends in cardiac surgery: why the logistic EuroSCORE is no longer suitable for contemporary cardiac surgery and implications for future risk models. Eur J Cardiothorac Surg 2013;43(6):1146-52. doi:10.1093/ejcts/ezs584",
 "L22": "Dong T, Sinha S, Zhai B, Fudulu D, Chan J, Narayan P, et al. Performance drift in machine learning models for cardiac surgery risk prediction: retrospective analysis. JMIRx Med 2024;5:e45973. doi:10.2196/45973",
 "L23": "van Dijk WB, Leeuwenberg AM, Grobbee DE, Siregar S, Houterman S, Daeter EJ, et al. Dynamics in cardiac surgery: trends in population characteristics and the performance of the EuroSCORE II over time. Eur J Cardiothorac Surg 2023;64(3):ezad301. doi:10.1093/ejcts/ezad301",
 "L24": "Zea-Vera R, Ryan CT, Havelka J, Corr SJ, Nguyen TC, Chatterjee S, et al. Machine learning to predict outcomes and cost by phase of care after coronary artery bypass grafting. Ann Thorac Surg 2022;114(3):711-9. doi:10.1016/j.athoracsur.2021.08.040",
 "L25": "Xue B, Li D, Lu C, King CR, Wildes T, Avidan MS, et al. Use of machine learning to develop and evaluate models using preoperative and intraoperative data to identify risks of postoperative complications. JAMA Netw Open 2021;4(3):e212240. doi:10.1001/jamanetworkopen.2021.2240",
 "L28": "Ghassemi M, Oakden-Rayner L, Beam AL. The false hope of current approaches to explainable artificial intelligence in health care. Lancet Digit Health 2021;3(11):e745-50. doi:10.1016/S2589-7500(21)00208-9",
 "L29": "O'Brien SM, Feng L, He X, Xian Y, Jacobs JP, Badhwar V, et al. The Society of Thoracic Surgeons 2018 adult cardiac surgery risk models: part 2 - statistical methods and results. Ann Thorac Surg 2018;105(5):1419-28. doi:10.1016/j.athoracsur.2018.03.003",
 "L31": "Penny-Dimri JC, Bergmeir C, Perry L, Hayes L, Bellomo R, Smith JA. Machine learning to predict adverse outcomes after cardiac surgery: a systematic review and meta-analysis. J Card Surg 2022;37(11):3838-45. doi:10.1111/jocs.16842",
 "L32": "Safavi KC, Khaniyev T, Copenhaver M, Seelen M, Zenteno Langle AC, Zanger J, et al. Development and validation of a machine learning model to aid discharge processes for inpatient surgical care. JAMA Netw Open 2019;2(12):e1917221. doi:10.1001/jamanetworkopen.2019.17221",
 "L33": "Stone K, Zwiggelaar R, Jones P, Mac Parthalain N. A systematic review of the prediction of hospital length of stay: towards a unified framework. PLOS Digit Health 2022;1(4):e0000017. doi:10.1371/journal.pdig.0000017",
 "L34": "Bruno VD, Guida G, Jones C, Bates M, Di Tommaso E, Rajakaruna C. A machine learning approach to predict the postoperative length of stay after coronary artery bypass grafting using preoperative characteristics. Br J Surg 2021;108(Suppl 6):znab258.048. doi:10.1093/bjs/znab258.048",
 "L36": "Greco G, Shi W, Michler RE, Meltzer DO, Ailawadi G, Hohmann SF, et al. Costs associated with health care-associated infections in cardiac surgery. J Am Coll Cardiol 2015;65(1):15-23. doi:10.1016/j.jacc.2014.09.079",
}
D = {   # Daniel Fudulu's previous NACSA analyses (data flow); verified on Crossref 28 Oct 2026
 "D1": "Fudulu DP, Dimagli A, Sinha S, Narayan P, Chan J, Dong T, et al. Weekday and outcomes of elective cardiac surgery in the UK: a large retrospective database analysis. Eur J Cardiothorac Surg 2022;61(6):1381-8. doi:10.1093/ejcts/ezac038",
 "D2": "Fudulu DP, Dimagli A, Sinha S, Ackah J, Narayan P, Chan J, et al. Impact of preoperative atrial fibrillation on in-hospital outcomes of coronary artery bypass grafting. Semin Thorac Cardiovasc Surg 2023;35(2):261-6. doi:10.1053/j.semtcvs.2022.07.002",
 "D3": "Fudulu DP, Dong T, Kota R, Sinha S, Chan J, Rajakaruna C, et al. In-hospital outcomes predictors and trends of redo sternotomy aortic root replacements: insights from a UK registry analysis. Front Cardiovasc Med 2024;10:1295968. doi:10.3389/fcvm.2023.1295968",
 "D4": "Fudulu DP, Argyriou A, Kota R, Chan J, Vohra H, Caputo M, et al. Effect of on-pump vs. off-pump coronary artery bypass grafting in patients with non-dialysis-dependent severe renal impairment: propensity-matched analysis from the UK registry dataset. Front Cardiovasc Med 2024;11:1341123. doi:10.3389/fcvm.2024.1341123",
 "D5": "Rogers LJ, Dong T, Kota R, Chan J, Mohamed S, Ciulli F, et al. Reoperative mitral valve surgery in the UK: 20-year trends and early outcomes. Int J Cardiol 2025;438:133602. doi:10.1016/j.ijcard.2025.133602",
 "D6": "Mashat MA, Dong T, Kota R, Di Tommaso E, Narayan P, Tan C, et al. Coronary artery bypass surgery in patients on dialysis: in-hospital outcomes from UK registry analysis. Interdiscip Cardiovasc Thorac Surg 2025;40(12):ivaf291. doi:10.1093/icvts/ivaf291",
}
L.update(D)
ALIAS = {"L27": "R54", "L26": "R53"}   # Lundberg papers already in the R-list

def expand(code):
    m = re.match(r"([RLD])(\d+)-[RLD]?(\d+)$", code)
    if m:
        return [f"{m.group(1)}{i}" for i in range(int(m.group(2)), int(m.group(3)) + 1)]
    return [code]

order = []
def number_citations(text):
    def repl(m):
        inner = m.group(1)
        codes = []
        for part in inner.split(","):
            for c in expand(part.strip()):
                c = ALIAS.get(c, c)
                codes.append(c)
        nums = []
        for c in codes:
            if c not in order:
                order.append(c)
            nums.append(order.index(c) + 1)
        nums = sorted(set(nums))
        # compress runs
        out, i = [], 0
        while i < len(nums):
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            out.append(f"{nums[i]}" if j == i else (f"{nums[i]},{nums[j]}" if j == i + 1 else f"{nums[i]}-{nums[j]}"))
            i = j + 1
        return "[" + ",".join(out) + "]"
    return re.sub(r"\[((?:[RLD]\d+(?:-[RLD]?\d+)?)(?:\s*,\s*[RLD]\d+(?:-[RLD]?\d+)?)*)\]", repl, text)

doc = Document()
st = doc.styles["Normal"]
st.font.name = "Arial"; st.font.size = Pt(11)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Arial")
pf = st.paragraph_format
pf.line_spacing_rule = WD_LINE_SPACING.DOUBLE
pf.space_after = Pt(0)
for lvl, size in (("Heading 1", 14), ("Heading 2", 12), ("Heading 3", 11)):
    hs = doc.styles[lvl]; hs.font.name = "Arial"; hs.font.size = Pt(size); hs.font.bold = True
    hs.font.color.rgb = None
    hs.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    hs.paragraph_format.space_before = Pt(12); hs.paragraph_format.space_after = Pt(0)
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
    setattr(sec, side, Cm(2.5))
ln = OxmlElement("w:lnNumType"); ln.set(qn("w:countBy"), "1"); ln.set(qn("w:restart"), "newPage"); ln.set(qn("w:distance"), "360")
sec._sectPr.append(ln)
# page numbers in footer
fp = sec.footer.paragraphs[0]
r = fp.add_run()
for t, txt in (("begin", None), (None, "PAGE"), ("end", None)):
    if t:
        e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), t); r._r.append(e)
    else:
        e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt; r._r.append(e)

def para(text, bold=False, italic=False, style=None):
    p = doc.add_paragraph(style=style)
    # simple highlight of [TO CONFIRM]/[CHECK] markers in bold
    parts = re.split(r"(\[(?:TO CONFIRM|CHECK)[^\]]*\])", text)
    for part in parts:
        if not part:
            continue
        run = p.add_run(part)
        run.bold = bold or part.startswith("[TO CONFIRM") or part.startswith("[CHECK")
        run.italic = italic
    return p

def block(items, cite=True):
    for kind, text in items:
        t = number_citations(text) if cite else text
        if kind == "h2":
            doc.add_heading(t, level=2)
        elif kind == "bp":
            lead, rest = t.split("|", 1)
            p = doc.add_paragraph(); p.add_run(lead + ": ").bold = True; p.add_run(rest)
        else:
            para(t)

def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)

# ---------- Title page ----------
SUBMIT = os.environ.get("SUBMIT") == "1"   # submission version: no draft line, no notes for authors
if not SUBMIT:
    para("DRAFT 5, submission candidate (29 October 2026): pre-submission check done (decision 177). Bold items in square brackets need the authors.", italic=True)
doc.add_heading(M.TITLE, level=1)
para("Short title: " + M.SHORT_TITLE)
para("Authors: Hamidreza Soleimani (1), Parham Dastjerdi (2), Pouya Ebrahimi (3), Maria Comanici (1), Cha Rajakaruna (1), Eltayeb Mohamed Ahmed (1), Gianni Angelini (1), Daniel Fudulu (1)")
para("(1) Bristol Heart Institute, University of Bristol, Bristol, United Kingdom. (2) [TO CONFIRM: affiliation of Parham Dastjerdi]. (3) [TO CONFIRM: affiliation of Pouya Ebrahimi].")
para("Corresponding author: Hamidreza Soleimani. Bristol Heart Institute, University of Bristol. Email: ec26229@bristol.ac.uk. ORCID: https://orcid.org/0000-0003-2802-4613")
WORDS = sum(len(re.sub(r"\[[^\]]*\]", "", t).split()) for sect in (M.INTRO, M.METHODS, M.RESULTS, M.DISCUSSION) for k, t in sect if k != "h2")
para(f"Word count (main text, Introduction to Discussion, excluding citations): {WORDS:,}. Abstract: {len(' '.join(t.split('|', 1)[1] for k, t in M.ABSTRACT).split())} words.")
para("Keywords: " + M.KEYWORDS)
page_break()

doc.add_heading("Abstract", level=1); block(M.ABSTRACT, cite=False)
doc.add_heading("Lay summary", level=1); block(M.LAY, cite=False)
page_break()
doc.add_heading("Introduction", level=1); block(M.INTRO)
doc.add_heading("Methods", level=1); block(M.METHODS)
doc.add_heading("Results", level=1); block(M.RESULTS)
doc.add_heading("Discussion", level=1); block(M.DISCUSSION)
page_break()
block(M.BACK, cite=False)

# ---------- References ----------
page_break()
doc.add_heading("References", level=1)
missing = []
for i, c in enumerate(order, 1):
    txt = R.get(c) or L.get(c)
    if txt is None:
        missing.append(c); txt = f"[{c}: citation missing]"
    para(f"{i}. {txt}")

# ---------- Tables ----------
def table(title, header, rows, note, widths=None):
    page_break()
    p = doc.add_paragraph(); p.add_run(title).bold = True
    t = doc.add_table(rows=1, cols=len(header)); t.style = "Table Grid"
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]; c.text = ""; c.paragraphs[0].add_run(h).bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = v
    for row in t.rows:
        for c in row.cells:
            for pp in c.paragraphs:
                pp.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
                for rr in pp.runs:
                    rr.font.size = Pt(8)
    if widths:
        t.autofit = False
        for i, w in enumerate(widths):
            t.columns[i].width = Cm(w)
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    para(note)

T1 = [
 ("Patients, n", "75,814", "32,104", "", "", ""),
 ("Age, years, mean (SD)", "66.5 (9.8)", "66.5 (9.6)", "0.00", "0.0", "0.0"),
 ("Female sex, n (%)", "13,409 (17.7)", "5,456 (17.0)", "0.02", "0.1", "<10 records"),
 ("Body mass index, kg/m2, mean (SD)", "28.9 (4.8)", "28.9 (4.8)", "0.01", "5.2", "2.2"),
 ("Diabetes (any treatment), n (%)", "23,111 (30.5)", "10,864 (33.8)", "0.07", "0.0", "0.0"),
 ("Hypertension, n (%)", "57,251 (75.5)", "24,533 (76.4)", "0.02", "0.0", "0.0"),
 ("Ever smoked, n (%)", "48,683 (64.2)", "19,989 (62.3)", "0.04", "0.0", "0.0"),
 ("Chronic pulmonary disease, n (%)", "8,446 (11.1)", "3,802 (11.8)", "0.02", "0.0", "0.0"),
 ("Extracardiac arteriopathy, n (%)", "9,164 (12.1)", "3,395 (10.6)", "0.05", "0.0", "0.0"),
 ("Previous TIA or stroke, n (%)", "5,036 (6.9)", "2,155 (6.8)", "0.00", "3.1", "1.2"),
 ("Neurological dysfunction, n (%)", "1,950 (2.6)", "790 (2.5)", "0.01", "0.0", "0.0"),
 ("Previous cardiac surgery, n (%)", "873 (1.2)", "333 (1.0)", "0.01", "0.0", "0.0"),
 ("Creatinine clearance, mL/min, mean (SD)", "88.1 (32.3)", "89.3 (32.3)", "0.04", "10.4", "5.4"),
 ("Renal failure and dialysis, n (%)", "", "", "0.01", "1.8", "2.4"),
 ("   None", "73,624 (98.9)", "30,959 (98.9)", "", "", ""),
 ("   Dialysis, acute renal failure", "132 (0.2)", "62 (0.2)", "", "", ""),
 ("   Dialysis, chronic renal failure", "513 (0.7)", "212 (0.7)", "", "", ""),
 ("   Acute renal failure, no dialysis", "198 (0.3)", "86 (0.3)", "", "", ""),
 ("Angina (CCS class), n (%)", "", "", "0.07", "0.0", "0.0"),
 ("   No angina", "7,679 (10.1)", "3,182 (9.9)", "", "", ""),
 ("   CCS 1", "6,620 (8.7)", "2,258 (7.0)", "", "", ""),
 ("   CCS 2", "28,673 (37.8)", "12,193 (38.0)", "", "", ""),
 ("   CCS 3", "22,615 (29.8)", "9,894 (30.8)", "", "", ""),
 ("   CCS 4", "10,227 (13.5)", "4,577 (14.3)", "", "", ""),
 ("Dyspnoea (NYHA class), n (%)", "", "", "0.02", "0.0", "0.0"),
 ("   NYHA I", "21,027 (27.7)", "9,101 (28.3)", "", "", ""),
 ("   NYHA II", "37,516 (49.5)", "15,929 (49.6)", "", "", ""),
 ("   NYHA III", "15,179 (20.0)", "6,239 (19.4)", "", "", ""),
 ("   NYHA IV", "2,092 (2.8)", "835 (2.6)", "", "", ""),
 ("Myocardial infarction within 90 days, n (%)", "24,775 (32.7)", "11,362 (35.4)", "0.06", "0.0", "0.0"),
 ("Impaired LV function (EF below 50%), n (%)", "19,853 (26.3)", "8,922 (28.0)", "0.04", "0.6", "0.7"),
 ("Pulmonary artery pressure, n (%)", "", "", "0.06", "0.0", "0.0"),
 ("   Below 31 mmHg", "73,719 (97.2)", "31,503 (98.1)", "", "", ""),
 ("   31-55 mmHg", "1,885 (2.5)", "514 (1.6)", "", "", ""),
 ("   Above 55 mmHg", "210 (0.3)", "87 (0.3)", "", "", ""),
 ("Critical pre-operative state, n (%)", "2,440 (3.2)", "965 (3.0)", "0.01", "0.0", "0.0"),
 ("Operative urgency, n (%)", "", "", "0.11", "0.0", "0.0"),
 ("   Elective", "43,016 (56.7)", "16,413 (51.1)", "", "", ""),
 ("   Urgent", "31,358 (41.4)", "15,002 (46.8)", "", "", ""),
 ("   Emergency", "1,352 (1.8)", "640 (2.0)", "", "", ""),
 ("   Salvage", "74 (0.1)", "33 (0.1)", "", "", ""),
 ("Diseased coronary vessels, n (%)", "", "", "0.03", "7.9", "6.4"),
 ("   None recorded (treated as missing in the models)", "599 (0.9)", "193 (0.6)", "", "", ""),
 ("   One", "3,103 (4.4)", "1,249 (4.2)", "", "", ""),
 ("   Two", "14,888 (21.3)", "6,320 (21.0)", "", "", ""),
 ("   Three", "51,224 (73.4)", "22,276 (74.2)", "", "", ""),
 ("Previous PCI, n (%)", "", "", "0.05", "0.0", "0.0"),
 ("   None", "63,742 (84.1)", "26,480 (82.5)", "", "", ""),
 ("   Within 24 hours", "421 (0.6)", "129 (0.4)", "", "", ""),
 ("   Same admission, over 24 hours", "1,066 (1.4)", "490 (1.5)", "", "", ""),
 ("   Previous admission", "10,585 (14.0)", "5,005 (15.6)", "", "", ""),
 ("On-pump (cardiopulmonary bypass), n (%)", "62,476 (86.6)", "27,038 (91.8)", "0.17", "4.8", "8.2"),
 ("Distal anastomoses, n, mean (SD)", "2.8 (1.0)", "2.9 (0.9)", "0.06", "3.3", "0.6"),
 ("EuroSCORE II, %, median [IQR]", "1.2 [0.8-2.1]", "1.3 [0.9-2.1]", "0.01", "0.1", "<10 records"),
]
table("Table 1. Baseline characteristics of patients discharged alive: development (2012-2016) and temporal evaluation (2017-2019) cohorts",
      ["Characteristic", "Development 2012-2016", "Evaluation 2017-2019", "SMD", "% missing, development", "% missing, evaluation"],
      T1, "SMD, standardized mean difference; CCS, Canadian Cardiovascular Society; NYHA, New York Heart Association; LV, left ventricular; EF, ejection fraction; PCI, percutaneous coronary intervention; TIA, transient ischaemic attack; IQR, interquartile range. Counts of 1-9 are shown as '<10', with a second count suppressed where needed. Percentages missing are before imputation.",
      [5.6, 2.6, 2.6, 1.2, 1.9, 1.9])

import os
import pandas as pd
TAB = os.path.join(os.environ.get("NACSA_DIR", "/home/user/nacsa_data"), "outputs", "18_final_exhibits", "tables")
t2 = pd.read_csv(os.path.join(TAB, "Table2.csv"), keep_default_na=False)
table("Table 2. Performance of every model at each prediction moment on 2017-2019 patients (temporal evaluation)",
      ["Prediction moment", "Model", "AUC (95% CI)", "Calibration-in-the-large (95% CI)", "Calibration slope (95% CI)", "Scaled Brier score (95% CI)"],
      t2.astype(str).values.tolist(),
      "AUC, area under the receiver operating characteristic curve; CI, confidence interval; MLP, multilayer perceptron. 95% CIs from 2,000 paired bootstrap resamples. Calibration-in-the-large: 0 is ideal (negative values mean risks too high on average); calibration slope: 1 is ideal. The headline model at each moment was chosen on 2012-2016 data before this evaluation.",
      [2.6, 3.4, 2.6, 3.2, 2.6, 2.6])

t3 = pd.read_csv(os.path.join(TAB, "Table3.csv"), keep_default_na=False)
table("Table 3. Does each later prediction moment add value? Headline model (XGBoost) on 2017-2019 patients",
      ["Comparison", "AUC difference (95% CI)", "Net benefit difference at 10%", "at 15%", "at 20%", "at 30%", "Adds value"],
      t3.astype(str).values.tolist(),
      "Differences are later moment minus earlier moment, or headline model minus recalibrated EuroSCORE II. Net benefit differences are per 100 patients (net extra prolonged stays correctly flagged, after penalising false flags) at threshold risks of 10%, 15%, 20% and 30%. Pre-specified rule: a later moment adds value when the 95% CI of the AUC difference lies above zero and net benefit is higher at all four thresholds than the earlier model's and than flagging everyone or no one; net benefit was judged on point estimates, and its CIs are descriptive. 95% CIs from the same 2,000 paired bootstrap resamples. Same-algorithm comparisons are in Supplementary Table ST11.",
      [3.6, 2.6, 2.1, 2.1, 2.1, 2.1, 1.4])

# ---------- Figure legends ----------
page_break()
doc.add_heading("Figure legends", level=1)
FIGS = [
 ("Figure 1. Study design and patient flow.", "Patients who died in hospital, or whose survival status or length of stay was not recorded, are excluded from the primary analysis (they are included in sensitivity analyses S1-S2). Counts of 1-9 would be shown as '<10'.", None),
 ("Figure 2. Receiver operating characteristic curves on 2017-2019 patients at each prediction moment.", "(A) Pre-operative, (B) intra-operative, (C) early post-operative. Ridge regression with splines, XGBoost (headline model) and recalibrated EuroSCORE II; AUC with 95% CI in each legend. All six models: Supplementary Figure SF5.", "Three panels of receiver operating characteristic curves, one per prediction moment, each showing three models as lines rising from the bottom left to the top right above the diagonal of no discrimination."),
 ("Figure 3. Calibration curves on 2017-2019 patients at each prediction moment.", "(A) Pre-operative, (B) intra-operative, (C) early post-operative. Lowess-smoothed observed proportion against predicted risk; the diagonal is perfect calibration; histograms of predicted risks below. All models: Supplementary Figure SF7.", "Three panels plotting observed against predicted risk of prolonged stay for three models, with a diagonal reference line and a histogram of predicted risks under each panel."),
 ("Figure 4. Decision curves on 2017-2019 patients at each prediction moment.", "(A) Pre-operative, (B) intra-operative, (C) early post-operative. Net benefit across threshold risks; flagging everyone and no one shown for reference; key thresholds 10-30%. All models: Supplementary Figure SF8.", "Three panels of net benefit against threshold risk for three models and for the flag-everyone and flag-no-one strategies."),
 ("Figure 5. What the headline model (XGBoost) uses at each prediction moment.", "(A-C) Mean absolute SHAP value of the 12 most important predictors at the pre-operative, intra-operative and early post-operative moments (log-odds scale). (D-F) SHAP value of each of 5,000 sampled 2017-2019 patients for the same predictors, coloured by the predictor's value (blue low, red high). Contributions to prediction, not causal effects.", "Six panels: three horizontal bar charts of predictor importance, one per prediction moment, above three dot plots in which each dot is one patient's contribution for each predictor, coloured from blue to red."),
 ("Figure 6. Two methods, one shape: ridge regression spline curves and XGBoost SHAP dependence for age, creatinine and bypass time (intra-operative models, applied at the end of surgery).", "(A-C) Ridge regression with restricted cubic splines: log odds ratio relative to the development median. (D-F) XGBoost: SHAP value of each of 5,000 sampled 2017-2019 patients. Both on the log-odds scale; each column covers the 1st-99th percentile of the predictor.", "Six panels in two rows: smooth rising curves from ridge regression for age, creatinine and bypass time above scatter plots of XGBoost contributions for the same three predictors, which follow similar rising patterns."),
 ("Graphical abstract.", "", "Left: data and design. Centre: the headline model's AUC rising across three prediction moments, above a EuroSCORE II line and below a dashed ceiling line. Right: four take-home messages."),
]
# REVIEW=1 builds the co-author review version with every figure placed above its legend (decision 177)
REVIEW = os.environ.get("REVIEW") == "1"
FIGDIR = os.path.join(os.path.dirname(TAB), "figures")
for title, legend, alt in FIGS:
    if REVIEW:
        fid = "GraphicalAbstract" if title.startswith("Graphical") else title.split(".")[0].replace(" ", "")
        page_break()
        doc.add_picture(os.path.join(FIGDIR, f"{fid}.png"), width=Cm(16))
    p = doc.add_paragraph(); p.add_run(title + " ").bold = True; p.add_run(legend)
    if alt:
        p = doc.add_paragraph(); p.add_run("Alt text: ").italic = True; p.add_run(alt).italic = True

words_pre = sum(len(re.sub(r"\[[^\]]*\]", "", t).split()) for sect in (M.INTRO, M.METHODS, M.RESULTS, M.DISCUSSION) for k, t in sect if k != "h2")
# ---------- Notes for authors ----------
for q in ([] if SUBMIT else ["__HEAD__"]) + [
 "Authors to supply: affiliations of Parham Dastjerdi and Pouya Ebrahimi; Claude model versions and the approving persons (AI paragraph); OSF identifier (Methods; abstract registration line if the journal asks); Zenodo DOI (Data availability; docs/submission/CODE_RELEASE.md); CRediT roles; ICMJE forms; lead-author photo; date and any earlier presentation in the cover letter.",
 f"Word count {WORDS:,} for Introduction to Discussion (limit 5,000; no room for additions). Abstract {len(' '.join(t.split('|', 1)[1] for k, t in M.ABSTRACT).split())} words (limit 250, no abbreviations).",
 f"References: {len(order)} (journal limit 50). To confirm on the journal page: whether supplementary references count toward the limit (decision 162).",
 "Graphical abstract: use the scripted version (journal rules on AI-generated figures; decision 163).",
]:
    if SUBMIT:
        break
    if q == "__HEAD__":
        page_break(); doc.add_heading("Notes for authors (delete before submission)", level=1)
        para("Pending decisions and checks:", bold=True); continue
    para("- " + q)

doc.save("NACSA_manuscript_submission.docx" if SUBMIT else "NACSA_manuscript_draft5_2026-10-29" + ("_review" if REVIEW else "") + ".docx")
print("refs:", len(order), "missing:", missing)
words = 0
for sect in (M.INTRO, M.METHODS, M.RESULTS, M.DISCUSSION):
    for k, t in sect:
        if k != "h2":
            words += len(re.sub(r"\[[^\]]*\]", "", t).split())
print("main text words (approx, excl. headings and citations):", words)
