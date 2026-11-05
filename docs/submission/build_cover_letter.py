"""Cover letter for EHJ Digital Health (decision 177). Run from this folder: python3 build_cover_letter.py"""
import sys
from docx import Document
from docx.shared import Pt
sys.path.insert(0, "../manuscript")
import ms_text as M

doc = Document()
st = doc.styles["Normal"]; st.font.name = "Arial"; st.font.size = Pt(11)
P = lambda t, b=False: doc.add_paragraph().add_run(t).__setattr__("bold", b) if b else doc.add_paragraph(t)
P("[TO CONFIRM: date of submission]")
P("The Editors\nEuropean Heart Journal - Digital Health")
P("Dear Editors,")
P(f"We submit our manuscript entitled “{M.TITLE}” for consideration as an Original Article in European Heart "
  "Journal - Digital Health.")
P("Prolonged hospital stay after coronary artery bypass grafting strains intensive care and ward capacity, yet the risk "
  "scores in routine use were built to predict death. Using the United Kingdom National Adult Cardiac Surgery Audit, we "
  "developed models at three moments of the surgical pathway (pre-operative, intra-operative and early post-operative) "
  "in 75,814 patients and evaluated the locked models once in 32,104 later patients. We found that:")
for t in ["machine learning (XGBoost) performed best but gave no meaningful benefit over penalised regression;",
          "most of the achievable prediction is available before surgery, with intra-operative and early post-operative "
          "information adding little, and a ceiling set by what registries do not record;",
          "age, kidney function and urgency of surgery drive the predictions;",
          "discrimination is lower in women and in patients aged 80 years or over, a potential source of inequity; and",
          "calibration drifts over time, so models need regular updating."]:
    doc.add_paragraph(t, style="List Bullet")
P("We believe these findings suit the journal's readership because they address questions that matter for deploying "
  "artificial intelligence in cardiovascular care: when machine learning is worth its added complexity, how much "
  "information each stage of the pathway adds, whether models perform equally across patient groups, and how quickly "
  "they drift once in use.")
P("The work was designed for transparency. The protocol and statistical analysis plan were frozen before any model was "
  "fitted, every later change is listed with its reason, the 2017-2019 data were opened only after the models were "
  "saved, and all analysis code is openly available. The study is reported following TRIPOD+AI, RECORD and the CODE-EHR "
  "minimum standards, with the checklists in the supplement.")
P("In line with the journal's policy, we disclose that Claude (Anthropic) was used, under the authors' direction, to "
  "write and run analysis code and to draft text, as described in the Methods and Acknowledgements; all analytic "
  "decisions were made by the authors, who checked every number and reference and take full responsibility for the work.")
P("The study was approved by the Health Research Authority and Health and Care Research Wales (IRAS project ID 257758), "
  "and the requirement for individual consent was waived. This manuscript has not been published and is not under "
  "consideration elsewhere. All authors have read and approved the submission. The authors declare no conflicts of "
  "interest, and the study received no specific funding. [TO CONFIRM: any earlier presentation of these results, e.g. "
  "a congress abstract, to be declared here.]")
P("Thank you for considering our manuscript.")
P("Yours sincerely,")
P("Hamidreza Soleimani, on behalf of all authors\nBristol Heart Institute, University of Bristol, Bristol, United Kingdom\n"
  "ec26229@bristol.ac.uk | ORCID 0000-0003-2802-4613")
doc.save("NACSA_cover_letter.docx")
print("saved")
