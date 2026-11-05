import sys
from docx import Document
src, out = sys.argv[1], sys.argv[2]
d = Document(src)
P = d.paragraphs
def runs_info(i): return [(r.text, r.bold) for r in P[i].runs]
def set_runs(p, texts):
    rs = p.runs
    for k, r in enumerate(rs):
        r.text = texts[k] if k < len(texts) else ""
    assert len(texts) <= len(rs)
def delete(p):
    p._element.getparent().remove(p._element)

TITLE = ("Prolonged length of stay after coronary artery bypass grafting: staged, interpretable prediction model "
         "development and temporal evaluation in a national registry")
assert P[0].text.startswith("Staged prediction"); set_runs(P[0], [TITLE])
assert P[1].text.startswith("Study protocol"); set_runs(P[1], ["Study protocol and statistical analysis plan (OSF Secondary Data Preregistration)"])
assert P[2].text.startswith("Version 1.0"); set_runs(P[2], ["Version 2.1, 10 October 2026. The methods are unchanged from version 1.0, frozen on 7 October 2026 "
    "(repository commit e89dfa1); versions 2.0 and 2.1 update the title, authors and administrative details only. Changes to the "
    "analysis after the freeze are reported as deviations (Section 14)."])
print(runs_info(3), runs_info(4), runs_info(5), runs_info(19), runs_info(20), runs_info(116))
set_runs(P[3], ["Authors: Hamidreza Soleimani, Parham Dastjerdi, Pouya Ebrahimi, Maria Comanici, Cha Rajakaruna, Eltayeb Mohamed Ahmed, Gianni Angelini, Daniel Fudulu"])
set_runs(P[4], ["Corresponding author: Hamidreza Soleimani. Bristol Heart Institute, University of Bristol. Email: ec26229@bristol.ac.uk. ORCID: https://orcid.org/0000-0003-2802-4613"])
assert P[19].text.startswith("2019 is a partial year")
set_runs(P[19], ["2019 is a partial year (3,264 analysed patients versus about 14,000-15,000 in other years). Last operation date in the extract: March 2019."])
assert P[20].text.startswith("Date the data were received")
set_runs(P[20], ["Date the data were received by the study team: April 2026. Date of access for this analysis: 6 October 2026 (clean analysis file, SHA-256 fingerprint recorded in the code repository)."])
assert P[116].text.startswith("Funding:")
set_runs(P[116], ["Funding: none; this study received no specific grant. Conflicts of interest: none declared."])
assert P[111].text.startswith("Earlier work on this extract"); delete(P[111])
assert P[5].text.startswith("Registration:"); delete(P[5])
for s in d.sections:
    for p in s.footer.paragraphs:
        for r in p.runs:
            r.text = r.text.replace("NACSA PLoS protocol v1.0 draft", "NACSA CABG length-of-stay protocol v2.1")
d.save(out)
