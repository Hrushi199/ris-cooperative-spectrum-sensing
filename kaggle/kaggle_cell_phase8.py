# ===================== PASTE THIS WHOLE CELL INTO KAGGLE =====================
# Phase 8 — minimum number of REs (Fig. 8, Fig. 9, analytic-vs-MC validation)
# Dataset must contain: system_model.py, channel_model.py, energy_detection.py,
#                       benchmark_psm.py, proposed_psm.py (NEW low-rank version),
#                       re_analysis.py
#                       + scsi_results.npz (optional: for the validation plot)
QUICK = False          # True = ~2-3 min smoke test

import glob, os, subprocess, sys

hits = glob.glob("/kaggle/input/**/re_analysis.py", recursive=True)
assert hits, "re_analysis.py not found — did you attach the dataset (Add Input)?"
code_dir = os.path.dirname(hits[0])
print("Code dir:", code_dir, "->", sorted(os.listdir(code_dir)))
assert "def _lowrank" in open(os.path.join(code_dir, "proposed_psm.py")).read(), \
    "OLD proposed_psm.py in dataset — upload scsi/proposed_psm.py from this repository!"
if not glob.glob("/kaggle/input/**/scsi_results.npz", recursive=True):
    print("NOTE: scsi_results.npz not uploaded -> validation plot shows analytic curves only")

cmd = [sys.executable, "-u", os.path.join(code_dir, "re_analysis.py")]
if QUICK:
    cmd.append("--quick")
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end="")
proc.wait()
print("exit code:", proc.returncode)

from IPython.display import Image, display
out = "/kaggle/working/re_results" + ("_quick" if QUICK else "")
for f in ["fig_validation_analytic_vs_mc.png", "fig8_min_M_vs_T.png", "fig8_min_M_vs_T_paper_style.png",
          "fig9_min_M_vs_T_kappa.png", "fig9_min_M_vs_T_kappa_paper_style.png"]:
    p = os.path.join(out, f)
    if os.path.exists(p):
        print(f); display(Image(p))
print("Outputs in", out, ":", os.listdir(out))
