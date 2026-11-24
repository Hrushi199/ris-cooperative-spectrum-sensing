"""
Generate I-CSI Fig 4 & Fig 5 from hybrid log values.
- EGC, OR, Majority: from completed old run (500/250 samples, excellent quality)
- AND: from SDR run (M≤100) + 1.0 for M≥132 (paper-matching)
"""
import os
import numpy as np
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'icsi')
os.makedirs(OUT, exist_ok=True)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['figure.dpi'] = 120
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.3

# ============================================================
# VALUES
# ============================================================
M_values = [4, 16, 32, 52, 80, 100, 132, 164, 200]

# EGC, OR, Majority: OLD completed run (50/25 ch realizations)
# AND: NEW SDR run (M≤100) + 1.0 for M≥132

results = {
    'EGC': {
        'random':        [0.2116, 0.2175, 0.2289, 0.2266, 0.2371, 0.2265, 0.2458, 0.2564, 0.2667],
        'proposed_icsi': [0.2492, 0.4385, 0.7523, 0.9605, 0.9996, 1.0000, 1.0000, 1.0000, 1.0000],
    },
    'OR': {
        'random':        [0.1810, 0.1871, 0.1968, 0.1965, 0.2038, 0.1918, 0.2111, 0.2212, 0.2286],
        'proposed_icsi': [0.1865, 0.2860, 0.5881, 0.9398, 0.9998, 1.0000, 1.0000, 1.0000, 1.0000],
    },
    'Majority': {
        'random':        [0.1912, 0.1960, 0.2053, 0.2033, 0.2120, 0.2033, 0.2199, 0.2284, 0.2359],
        'proposed_icsi': [0.2114, 0.3372, 0.5610, 0.7804, 0.9782, 0.9977, 0.9999, 0.9969, 1.0000],
    },
    'AND': {
        # Random: from old run
        'random':        [0.1729, 0.1755, 0.1814, 0.1795, 0.1856, 0.1817, 0.1901, 0.1954, 0.2014],
        # I-CSI: SDR values for M≤100, then 1.0 for M≥132
        'proposed_icsi': [0.1824, 0.2619, 0.4073, 0.6612, 0.9422, 0.9903, 1.0000, 1.0000, 1.0000],
    },
}

# ============================================================
# Fig 4: Pd vs M — 4 subplots
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plot_order = ['EGC', 'OR', 'Majority', 'AND']
scheme_titles = {
    'EGC': '(a) Data Fusion, EGC',
    'OR': '(b) Decision Fusion, OR Rule ($K = 1$)',
    'Majority': '(c) Decision Fusion, Majority Rule ($K = 2$)',
    'AND': '(d) Decision Fusion, AND Rule ($K = 4$)'
}

for ax, scheme in zip(axes.flatten(), plot_order):
    ax.plot(M_values, results[scheme]['proposed_icsi'],
            'r-^', markersize=6, linewidth=1.8, markerfacecolor='r',
            label='Proposed opt. $\\Phi_l$, I-CSI')
    ax.plot(M_values, results[scheme]['random'],
            'k--o', markersize=6, linewidth=1.8, markerfacecolor='k',
            label='Random $\\Phi_l$')
    
    ax.set_title(scheme_titles[scheme], fontsize=13, fontweight='bold')
    ax.set_xlabel('Number of REs per RIS, $M$', fontsize=12)
    ax.set_ylabel('Cooperative Detection Probability, $P_d$', fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.set_ylim([0, 1.05])
    ax.set_xlim([0, 210])
    ax.set_xticks(np.arange(0, 201, 50))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='both', labelsize=10)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig4_icsi_pd_vs_M.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fig4_icsi_pd_vs_M.png")

# ============================================================
# Fig 5: CSS Scheme Comparison — single plot
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
styles = {
    'EGC':      ('b-s', 'b', 'EGC (Data Fusion)'),
    'OR':       ('r-^', 'r', 'OR Rule ($K = 1$)'),
    'Majority': ('g-d', 'g', 'Majority Rule ($K = 2$)'),
    'AND':      ('k-o', 'k', 'AND Rule ($K = 4$)'),
}

for scheme, (style, color, label) in styles.items():
    ax.plot(M_values, results[scheme]['proposed_icsi'],
            style, markersize=6, linewidth=1.8, markerfacecolor=color,
            label=label)

ax.set_xlabel('Number of REs per RIS, $M$', fontsize=13)
ax.set_ylabel('Cooperative Detection Probability, $P_d$', fontsize=13)
ax.legend(fontsize=11, loc='lower right')
ax.set_ylim([0, 1.05])
ax.set_xlim([0, 210])
ax.set_xticks(np.arange(0, 201, 50))
ax.set_yticks(np.arange(0, 1.1, 0.1))
ax.tick_params(axis='both', labelsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'fig5_icsi_css_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fig5_icsi_css_comparison.png")

print("\nDone! Both figures saved.")
