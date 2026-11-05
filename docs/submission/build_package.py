"""Submission package for EHJ Digital Health (decision 177).
Usage: NACSA_DIR=<data dir> python3 build_package.py <output folder>
Builds the separate files the journal asks for from ms_text.py and the Step 18 outputs:
title page, abstract (separate file, no abbreviations), manuscript, figures, graphical abstract, supplement,
checklists (inside the supplement), protocol and cover letter."""
import os, shutil, subprocess, sys
from pathlib import Path
from docx import Document
from docx.shared import Pt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "docs" / "manuscript"))
import ms_text as M

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
EX = Path(os.environ.get("NACSA_DIR", "/home/user/nacsa_data")) / "outputs" / "18_final_exhibits"
env = {**os.environ, "SUBMIT": "1"}


def new_doc():
    d = Document(); st = d.styles["Normal"]; st.font.name = "Arial"; st.font.size = Pt(11); return d


# 1. Title page
d = new_doc()
d.add_heading(M.TITLE, level=1)
d.add_paragraph("Short title: " + M.SHORT_TITLE)
d.add_paragraph("Authors: Hamidreza Soleimani (1), Parham Dastjerdi (2), Pouya Ebrahimi (3), Maria Comanici (1), "
                "Cha Rajakaruna (1), Eltayeb Mohamed Ahmed (1), Gianni Angelini (1), Daniel Fudulu (1)")
d.add_paragraph("(1) Bristol Heart Institute, University of Bristol, Bristol, United Kingdom. (2) [TO CONFIRM]. (3) [TO CONFIRM].")
d.add_paragraph("Corresponding author: Hamidreza Soleimani, Bristol Heart Institute, University of Bristol, Bristol, "
                "United Kingdom. Email: ec26229@bristol.ac.uk. ORCID: https://orcid.org/0000-0003-2802-4613")
import re
words = sum(len(re.sub(r"\[[^\]]*\]", "", t).split()) for s in (M.INTRO, M.METHODS, M.RESULTS, M.DISCUSSION)
            for k, t in s if k != "h2")
d.add_paragraph(f"Word count (Introduction to Discussion, excluding references, tables and legends): {words:,}")
d.add_paragraph("Tables: 3. Figures: 6. Supplementary material: one file.")
d.add_paragraph("Keywords: " + M.KEYWORDS)
d.save(OUT / "02_title_page.docx")

# 2. Abstract and lay summary (separate file)
d = new_doc()
d.add_heading("Abstract", level=1)
for k, t in M.ABSTRACT:
    lead, rest = t.split("|", 1); p = d.add_paragraph(); p.add_run(lead + ": ").bold = True; p.add_run(rest)
d.add_heading("Lay summary", level=1)
for k, t in M.LAY:
    d.add_paragraph(t)
d.save(OUT / "03_abstract_and_lay_summary.docx")

# 3. Manuscript (submission version: no draft line, no notes)
subprocess.run([sys.executable, "build_ms.py"], cwd=REPO / "docs" / "manuscript", env=env, check=True,
               stdout=subprocess.DEVNULL)
shutil.move(REPO / "docs" / "manuscript" / "NACSA_manuscript_submission.docx", OUT / "04_manuscript.docx")

# 4. Figures and graphical abstract (vector PDF and 300 dpi PNG)
fig = OUT / "05_figures"; fig.mkdir(exist_ok=True)
for f in [f"Figure{i}" for i in range(1, 7)] + ["GraphicalAbstract"]:
    for ext in ("pdf", "png"):
        shutil.copy(EX / "figures" / f"{f}.{ext}", fig / f"{f}.{ext}")

# 5. Supplement (includes TRIPOD+AI, RECORD and CODE-EHR checklists: ST18, ST22, ST23)
for ext in ("docx", "pdf"):
    shutil.copy(EX / f"supplementary_material.{ext}", OUT / f"06_supplementary_material.{ext}")

# 6. Protocol and cover letter
shutil.copy(REPO / "docs" / "protocol" / "NACSA_protocol_v2.1.docx", OUT / "07_protocol_v2.1.docx")
subprocess.run([sys.executable, "build_cover_letter.py"], cwd=HERE, check=True, stdout=subprocess.DEVNULL)
shutil.move(HERE / "NACSA_cover_letter.docx", OUT / "01_cover_letter.docx")

(OUT / "00_README.txt").write_text(
    "Submission package, EHJ Digital Health (built by docs/submission/build_package.py)\n"
    "01 cover letter | 02 title page | 03 abstract and lay summary | 04 manuscript (tables at the end) |\n"
    "05 figures 1-6 and graphical abstract (PDF vector, PNG 300 dpi) | 06 supplement (checklists ST18, ST22, ST23) |\n"
    "07 protocol\n"
    "Before upload, fill every bold [TO CONFIRM] in 01, 02 and 04 (affiliations, model versions and approvers,\n"
    "OSF identifier, Zenodo DOI, CRediT roles, date), add the ICMJE forms and the lead-author photo.\n")
print("package written to", OUT)
