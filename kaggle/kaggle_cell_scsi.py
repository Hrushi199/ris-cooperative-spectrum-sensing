# ===================== PASTE THIS WHOLE CELL INTO KAGGLE =====================
# Phase 7 — S-CSI run (Fig. 3, Fig. 4, Fig. 5 + raw numbers)
# Dataset must contain: system_model.py, channel_model.py, energy_detection.py,
#                       benchmark_psm.py, proposed_psm.py, scsi_simulations.py
QUICK = False          # True = 2-3 min smoke test, False = full run (~1.5-2.5 h)

import glob, os, subprocess, sys

# 1) cvxpy for the SDR candidate (optional — code falls back if missing)
try:
    import cvxpy  # noqa: F401
    print("cvxpy", cvxpy.__version__)
except ImportError:
    print("Installing cvxpy (needs Internet ON)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cvxpy"])

# 2) locate the uploaded code
hits = glob.glob("/kaggle/input/**/scsi_simulations.py", recursive=True)
assert hits, "scsi_simulations.py not found — did you attach the dataset (Add Input)?"
code_dir = os.path.dirname(hits[0])
print("Code dir:", code_dir, "->", sorted(os.listdir(code_dir)))
assert "def _span_search_maxmin" in open(os.path.join(code_dir, "proposed_psm.py")).read(), \
    "OLD proposed_psm.py in dataset — upload scsi/proposed_psm.py from this repository!"

# 3) run (as a subprocess so multiprocessing works and output streams live)
cmd = [sys.executable, "-u", os.path.join(code_dir, "scsi_simulations.py")]
if QUICK:
    cmd.append("--quick")
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end="")
proc.wait()
print("exit code:", proc.returncode)

# 4) show the figures
from IPython.display import Image, display
out = "/kaggle/working/scsi_results" + ("_quick" if QUICK else "")
for f in ["fig3_gamma_validation.png", "fig4_scsi_pd_vs_M.png", "fig5_scsi_css_comparison.png"]:
    p = os.path.join(out, f)
    if os.path.exists(p):
        print(f); display(Image(p))
print("Outputs in", out, ":", os.listdir(out))
