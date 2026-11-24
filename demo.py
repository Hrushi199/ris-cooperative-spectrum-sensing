"""
demo.py — LIVE CODE DEMO (runs in ~1 minute)
=============================================
RIS-Assisted Cooperative Spectrum Sensing (Ge et al., IEEE TWC 2024)

Walks through the whole pipeline on ONE SU drop, printing what happens at
each stage. The full simulations (hours, run on Kaggle) produced the figures
in results/; this script shows the same code working live.

    python demo.py

Steps
  1. System model        topology, path losses              (core/system_model.py)
  2. Channel model       Rayleigh / LOS / Rician channels   (core/channel_model.py)
  3. Energy detection    thresholds + Monte-Carlo Pfa check (core/energy_detection.py)
  4. PSM optimisation    Random, Partitioned, S-CSI, I-CSI  (scsi/, icsi/)
  5. Detection prob.     cooperative Pd for every scheme
  6. Gamma approximation analytic vs. Monte-Carlo moments   (Prop. 3)
  7. Phase 8             bisection: minimum M for Pd = 0.9  (scsi/re_analysis.py)
"""
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')        # SCS "solution may be inaccurate" notices

ROOT = os.path.dirname(os.path.abspath(__file__))
for d in ('core', 'scsi', 'icsi'):
    sys.path.insert(0, os.path.join(ROOT, d))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from system_model import (generate_su_positions, get_all_path_losses, plot_topology,
                          N_SU, L_RIS, KAPPA, T_SAMPLES, P_FA_BAR, SIGMA2_S, SIGMA2_U,
                          RIS_POSITIONS, PU_POS)
from channel_model import generate_all_channels, compute_equivalent_channel
from energy_detection import (compute_threshold_egc, compute_threshold_and,
                              compute_threshold_or, compute_threshold_krank,
                              compute_pd_individual, compute_pd_egc, compute_pd_and,
                              compute_pd_or, compute_pd_krank, q_inv)
from benchmark_psm import random_psm, benchmark_partitioned_psm
from proposed_psm import optimize_psm_scsi
from proposed_psm_icsi import optimize_psm_icsi

OUT = os.path.join(ROOT, 'results', 'demo')
os.makedirs(OUT, exist_ok=True)
SCHEMES = ['EGC', 'OR', 'Majority', 'AND']
K_OF = {'EGC': None, 'OR': 1, 'Majority': 2, 'AND': N_SU}
db = lambda x: 10 * np.log10(x)


def header(i, title):
    print('\n' + '=' * 72 + f'\n[{i}] {title}\n' + '=' * 72, flush=True)


T0 = time.time()
M, MH, MV = 64, 8, 8

# ---------------------------------------------------------------- 1
header(1, 'SYSTEM MODEL  (Sec. V of the paper)')
su = generate_su_positions(seed=7)
eta, beta, zeta = get_all_path_losses(su)
print(f"PU at {tuple(PU_POS)}, RISs at {[tuple(r) for r in RIS_POSITIONS]}, "
      f"N = {N_SU} SUs, L = {L_RIS} RISs, M = {M} ({MH}x{MV}) REs per RIS")
print(f"PU power {SIGMA2_S:.0f} W (30 dBm), noise {SIGMA2_U:.0e} W (-80 dBm), "
      f"T = {T_SAMPLES}, max Pfa = {P_FA_BAR}, kappa = {KAPPA:g}")
for n in range(N_SU):
    print(f"  SU{n+1} at ({su[n,0]:6.1f},{su[n,1]:5.1f}) m   direct-link SNR "
          f"{db(eta[n]*SIGMA2_S/SIGMA2_U):6.1f} dB")
plt.ioff()
plt.show = lambda *a, **k: None          # keep plot_topology non-blocking
plot_topology(su, save_path=os.path.join(OUT, 'topology.png'))
plt.close('all')
print(f"  -> topology plot: results/demo/topology.png")

# ---------------------------------------------------------------- 2
header(2, 'CHANNEL MODEL  d_n: Rayleigh,  f_l: LOS,  g_n,l: Rician')
rng = np.random.default_rng(1)
d, f, g, g_bar = generate_all_channels(su, eta, beta, zeta, MH, MV, KAPPA, rng)
print(f"d: {d.shape}   f: {L_RIS} x {f[0].shape}   g: {N_SU} x {L_RIS} x {g[0][0].shape}")
p_los = sum(np.linalg.norm(g_bar[n][l]) ** 2 for n in range(N_SU) for l in range(L_RIS))
p_nlos = sum(np.linalg.norm(g[n][l] - g_bar[n][l]) ** 2 for n in range(N_SU) for l in range(L_RIS))
print(f"Rician RIS-SU channel: LOS power share {p_los/(p_los+p_nlos):.3f}"
      f"  (theory kappa/(1+kappa) = {KAPPA/(1+KAPPA):.3f})")

# ---------------------------------------------------------------- 3
header(3, 'ENERGY DETECTION  thresholds (CLT) + Monte-Carlo false-alarm check')
gam = {'EGC': compute_threshold_egc(), 'OR': compute_threshold_or(),
       'Majority': compute_threshold_krank(K=2)[0], 'AND': compute_threshold_and()}
mc = np.random.default_rng(0)
n_trials = 4000
E0 = np.mean(np.abs(np.sqrt(SIGMA2_U / 2) * (mc.standard_normal((n_trials, N_SU, T_SAMPLES))
             + 1j * mc.standard_normal((n_trials, N_SU, T_SAMPLES)))) ** 2, axis=2)
dec = {s: E0 > gam[s] for s in ('OR', 'Majority', 'AND')}
pfa = {'EGC': np.mean(E0.mean(axis=1) > gam['EGC']),
       'OR': np.mean(dec['OR'].sum(1) >= 1), 'Majority': np.mean(dec['Majority'].sum(1) >= 2),
       'AND': np.mean(dec['AND'].sum(1) >= 4)}
for s in SCHEMES:
    print(f"  {s:9s} gamma = {gam[s]/SIGMA2_U:.5f} x sigma_u^2    "
          f"simulated cooperative Pfa = {pfa[s]:.3f}   (target {P_FA_BAR})")

# ---------------------------------------------------------------- 4
header(4, f'PSM OPTIMISATION at M = {M}   (gain |h_n|^2 as SNR in dB, per SU)')
phis = {'Random': random_psm(M, L_RIS, np.random.default_rng(5)),
        'Partitioned': benchmark_partitioned_psm(f, g_bar, M, N_SU, L_RIS)}
for s in SCHEMES:
    t = time.time()
    phis[f'S-CSI {s}'] = optimize_psm_scsi(s, f, g_bar, N_SU, L_RIS, M, K=K_OF[s],
                                           rng=np.random.default_rng(0), use_sdr=False)
    print(f"  optimised S-CSI {s:9s} in {time.time()-t:4.1f} s")
for s in SCHEMES:
    t = time.time()
    phis[f'I-CSI {s}'] = optimize_psm_icsi(s, d, f, g, g_bar, N_SU, L_RIS, M, K=K_OF[s] or 1,
                                           rng=np.random.default_rng(0))
    print(f"  optimised I-CSI {s:9s} in {time.time()-t:4.1f} s"
          + ('   (SDR + Gaussian randomisation)' if s == 'AND' else ''))
print(f"\n  {'PSM':<18}" + ''.join(f'{"SU"+str(n+1):>8}' for n in range(N_SU))
      + f"{'sum':>8}{'min':>8}{'max':>8}")
print(f"  {'No RIS':<18}" + ''.join(f'{db(r*SIGMA2_S/SIGMA2_U):8.1f}' for r in np.abs(d)**2))
for name, phi in phis.items():
    rho = compute_equivalent_channel(d, f, g, phi)[1] * SIGMA2_S / SIGMA2_U
    print(f"  {name:<18}" + ''.join(f'{db(r):8.1f}' for r in rho)
          + f"{db(rho.sum()):8.1f}{db(rho.min()):8.1f}{db(rho.max()):8.1f}")
print("  -> each optimiser wins on its own objective: EGC on 'sum', AND on 'min', OR on 'max'")

# ---------------------------------------------------------------- 5
header(5, 'COOPERATIVE DETECTION PROBABILITY  (S-CSI PSM fixed, 300 fading draws)')


def coop_pd(rho, s):
    if s == 'EGC':
        return compute_pd_egc(rho, gam['EGC'])
    pd = compute_pd_individual(rho, gam[s])
    return {'OR': compute_pd_or, 'AND': compute_pd_and}.get(s, lambda p: compute_pd_krank(p, 2))(pd)


acc = {c: {s: 0.0 for s in SCHEMES} for c in ['No RIS', 'Random', 'Partitioned', 'Proposed S-CSI']}
n_fade = 300
for k in range(n_fade):
    d2, f2, g2, _ = generate_all_channels(su, eta, beta, zeta, MH, MV, KAPPA,
                                          np.random.default_rng(100 + k))
    rr = {'No RIS': np.abs(d2) ** 2,
          'Random': compute_equivalent_channel(d2, f2, g2, random_psm(M, L_RIS, np.random.default_rng(k)))[1],
          'Partitioned': compute_equivalent_channel(d2, f2, g2, phis['Partitioned'])[1]}
    for s in SCHEMES:
        for c in ['No RIS', 'Random', 'Partitioned']:
            acc[c][s] += coop_pd(rr[c], s) / n_fade
        acc['Proposed S-CSI'][s] += coop_pd(
            compute_equivalent_channel(d2, f2, g2, phis[f'S-CSI {s}'])[1], s) / n_fade
print(f"  {'':<16}" + ''.join(f'{s:>10}' for s in SCHEMES))
for c, row in acc.items():
    print(f"  {c:<16}" + ''.join(f'{row[s]:10.3f}' for s in SCHEMES))

# ---------------------------------------------------------------- 6
header(6, 'GAMMA APPROXIMATION of rho = ||h||^2  (Proposition 3)')
phi = phis['S-CSI EGC']
pf = [phi[l] * f[l] for l in range(L_RIS)]
c = np.array([sum(np.conj(g_bar[n][l]) @ pf[l] for l in range(L_RIS)) for n in range(N_SU)])
b = M * np.array([sum(zeta[n, l] * beta[l] / (1 + KAPPA) for l in range(L_RIS)) for n in range(N_SU)])
a = eta + b
mu, var = np.sum(np.abs(c) ** 2 + a), np.sum(a ** 2 + 2 * a * np.abs(c) ** 2)
samples = np.array([np.sum(compute_equivalent_channel(*generate_all_channels(
    su, eta, beta, zeta, MH, MV, KAPPA, np.random.default_rng(5000 + k))[:3], phi)[1])
    for k in range(3000)])
print(f"  mean     analytic {mu:.4e}   Monte-Carlo {samples.mean():.4e}")
print(f"  variance analytic {var:.4e}   Monte-Carlo {samples.var():.4e}")
print(f"  Gamma shape k = {mu**2/var:.1f}, scale theta = {var/mu:.3e}")

# ---------------------------------------------------------------- 7
header(7, 'PHASE 8: minimum M for Pd >= 0.9 by bisection  (EGC, T = 1000, 2 SU drops)')
sys.argv = [sys.argv[0]]
import re_analysis as ra
ra.N_DROPS = 2
for cfg in ['proposed', 'partitioned']:
    t = time.time()
    ev = ra.Evaluator(cfg, 'EGC', KAPPA)
    m, m_grid = ev.min_M(1000)
    print(f"  {cfg:<12} needs M = {m:5.1f} REs per RIS (smallest array on the grid: {m_grid}, "
          f"Pd there {ev.pd(m_grid, 1000):.3f}; bisection evaluated {ev.n_opt} sizes, {time.time()-t:.1f} s)")

print('\n' + '=' * 72)
print(f"Demo finished in {time.time()-T0:.0f} s.  Full-scale figures: results/")
print('=' * 72)
