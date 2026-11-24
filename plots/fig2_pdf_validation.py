"""
fig2_pdf_validation.py — Fig. 2: PDF of the energy-detector test statistic
===========================================================================
E_n | H1 ~ N(sigma_u^2 + rho_n sigma_s^2, (sigma_u^2 + rho_n sigma_s^2)^2 / T)   (CLT, Eq. 19)

Normalised as in the paper (sigma_s^2 = sigma_u^2 = 1, T = 1000).
Lines  = CLT Gaussian approximation.
Bars   = Monte-Carlo: E_n = (1/T) sum |h s(t) + u(t)|^2 with Gaussian s(t), u(t).
The match validates the analytical model all Pd formulas are built on.

    python plots/fig2_pdf_validation.py      (~5 s)
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'phase1')
os.makedirs(OUT, exist_ok=True)

T, N_TRIALS = 1000, 6000
rng = np.random.default_rng(0)
plt.rcParams.update({'font.family': 'serif', 'font.size': 12})
fig, ax = plt.subplots(figsize=(8, 6))
x = np.linspace(0.8, 1.4, 800)
colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E']
for rho, col in zip([0, 0.05, 0.1, 0.2], colors):
    # Monte Carlo: received samples x(t) = h s(t) + u(t), |h|^2 = rho
    s = (rng.standard_normal((N_TRIALS, T)) + 1j * rng.standard_normal((N_TRIALS, T))) / np.sqrt(2)
    u = (rng.standard_normal((N_TRIALS, T)) + 1j * rng.standard_normal((N_TRIALS, T))) / np.sqrt(2)
    E = np.mean(np.abs(np.sqrt(rho) * s + u) ** 2, axis=1)
    ax.hist(E, bins=60, density=True, color=col, alpha=0.30, edgecolor='none')
    m = 1 + rho
    ax.plot(x, np.exp(-0.5 * ((x - m) / (m / np.sqrt(T))) ** 2) / (m / np.sqrt(T) * np.sqrt(2 * np.pi)),
            color=col, lw=2.2,
            label=(r'$\rho_n = 0\;(\mathcal{H}_0)$' if rho == 0 else rf'$\rho_n = {rho}$'))
    print(f"rho={rho:<5} MC mean {E.mean():.4f} (theory {m:.4f})  "
          f"MC std {E.std():.4f} (theory {m/np.sqrt(T):.4f})")
ax.plot([], [], color='gray', lw=0, marker='s', alpha=0.4, label='Monte Carlo')
ax.set_xlabel('Test Statistic, $E_n$')
ax.set_ylabel('Probability Density Function')
ax.set_xlim(0.8, 1.4); ax.set_ylim(0, 14)
ax.grid(alpha=0.3)
ax.legend(loc='upper right')
plt.tight_layout()
p = os.path.join(OUT, 'fig2_pdf_validation_mc.png')
plt.savefig(p, dpi=150, bbox_inches='tight')
print('Saved', p)
