"""
combined_fig4.py — FINAL Fig. 4 and Fig. 5: I-CSI vs S-CSI vs benchmarks
=========================================================================
Reads saved numbers only (no simulation):
  results/icsi/icsi_results.json   I-CSI Kaggle run (AND at M >= 132 extrapolated,
                                   drawn with hollow markers + dotted line)
  results/scsi/scsi_results.npz    S-CSI Kaggle run (Proposed, Partitioned, Random, No RIS)

    python plots/combined_fig4.py      (~3 s)
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(ROOT, 'results', 'combined')
os.makedirs(OUT, exist_ok=True)

icsi = json.load(open(os.path.join(ROOT, 'results', 'icsi', 'icsi_results.json')))
scsi = np.load(os.path.join(ROOT, 'results', 'scsi', 'scsi_results.npz'))
M_i, M_s = np.array(icsi['M']), scsi['M']

plt.rcParams.update({'font.family': 'serif', 'font.size': 12,
                     'axes.grid': True, 'grid.alpha': 0.3})
TITLES = {'EGC': '(a) Data Fusion, EGC', 'OR': '(b) Decision Fusion, OR Rule ($K=1$)',
          'Majority': '(c) Decision Fusion, Majority Rule ($K=2$)',
          'AND': '(d) Decision Fusion, AND Rule ($K=4$)'}


def plot_icsi(ax, s, **kw):
    y = np.array(icsi['results'][s]['proposed_icsi'])
    ext = np.array(icsi.get('extrapolated', {}).get(s, [False] * len(y)))
    sim = ~ext
    ax.plot(M_i[sim], y[sim], '-^', **kw)
    if ext.any():                                    # extrapolated tail: hollow + dotted
        j = np.where(sim)[0][-1]
        ax.plot(np.r_[M_i[j], M_i[ext]], np.r_[y[j], y[ext]], ':', color=kw['color'], lw=1.6)
        ax.plot(M_i[ext], y[ext], '^', color=kw['color'], mfc='white', ms=kw.get('ms', 6))


def axes_fmt(ax):
    ax.set_xlabel('Number of REs per RIS, $M$')
    ax.set_ylabel('Cooperative Detection Probability, $P_d$')
    ax.set_xlim(0, 205); ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0, 201, 50)); ax.set_yticks(np.arange(0, 1.01, 0.1))


# ------------------------------ Fig. 4 ------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, s in zip(axes.flat, ['EGC', 'OR', 'Majority', 'AND']):
    plot_icsi(ax, s, color='r', lw=1.8, ms=6, label='Proposed opt. $\\Phi_l$, I-CSI')
    ax.plot(M_s, scsi[f'{s}__proposed_scsi'], 'g-o', lw=1.8, ms=5,
            label='Proposed opt. $\\Phi_l$, S-CSI')
    ax.plot(M_s, scsi[f'{s}__partitioned'], 'b-s', lw=1.8, ms=5,
            label='Partitioned opt. $\\Phi_l$, S-CSI')
    ax.plot(M_s, scsi[f'{s}__random'], 'k--o', lw=1.4, ms=4, mfc='none', label='Random $\\Phi_l$')
    ax.plot(M_s, scsi[f'{s}__no_ris'], ':', color='gray', lw=1.5, label='No RIS')
    if s == 'AND':
        ax.plot([], [], ':^', color='r', mfc='white', label='I-CSI, extrapolated')
    ax.set_title(TITLES[s])
    axes_fmt(ax)
    ax.legend(fontsize=8.5, loc='lower right', bbox_to_anchor=(1.0, 0.25))
plt.tight_layout()
p = os.path.join(OUT, 'fig4_combined_pd_vs_M.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print('Saved', p)

# ------------------------------ Fig. 5 ------------------------------
fig, ax = plt.subplots(figsize=(9, 6.5))
for s, col, mk, lab in [('EGC', 'b', 's', 'EGC'), ('OR', 'r', 'o', 'OR ($K$=1)'),
                        ('Majority', 'g', '^', 'Majority ($K$=2)'), ('AND', 'k', 'D', 'AND ($K$=4)')]:
    ax.plot(M_s, scsi[f'{s}__proposed_scsi'], '-', marker=mk, color=col, lw=1.7, ms=5,
            label=f'{lab}, S-CSI')
    y = np.array(icsi['results'][s]['proposed_icsi'])
    ext = np.array(icsi.get('extrapolated', {}).get(s, [False] * len(y)))
    ax.plot(M_i[~ext], y[~ext], '--', marker=mk, color=col, lw=1.3, ms=5, mfc='none',
            label=f'{lab}, I-CSI')
ax.set_title('Comparison of CSS schemes (solid: S-CSI, dashed: I-CSI)')
axes_fmt(ax)
ax.legend(fontsize=9, ncol=2, loc='lower right')
plt.tight_layout()
p = os.path.join(OUT, 'fig5_combined_css_comparison.png')
plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
print('Saved', p)

# ------------------------------ key numbers ------------------------------
print('\nPd at M = 100:')
i100 = list(M_i).index(100); s100 = list(M_s).index(100)
for s in ['EGC', 'OR', 'Majority', 'AND']:
    print(f"  {s:9s} I-CSI {icsi['results'][s]['proposed_icsi'][i100]:.3f}  "
          f"S-CSI {scsi[s+'__proposed_scsi'][s100]:.3f}  "
          f"Partitioned {scsi[s+'__partitioned'][s100]:.3f}  Random {scsi[s+'__random'][s100]:.3f}")
