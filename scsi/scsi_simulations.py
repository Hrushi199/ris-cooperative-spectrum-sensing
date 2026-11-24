"""
scsi_simulations.py — Phase 7: S-CSI results (Fig. 3, Fig. 4, Fig. 5)
======================================================================
Ge et al., "RIS-Assisted Cooperative Spectrum Sensing for CRNs", IEEE TWC 2024

Produces (in OUT_DIR):
  fig3_gamma_validation.png   PDF of rho = ||h||^2 vs Gamma / Gaussian approx.
  fig4_scsi_pd_vs_M.png       Pd vs M, 4 subplots: Proposed S-CSI, Partitioned,
                              Random, No RIS
  fig5_scsi_css_comparison.png  All CSS schemes, Proposed S-CSI
  scsi_results.npz / .csv     RAW NUMBERS (use these for the combined
                              I-CSI + S-CSI figure later; no hard-coding!)
  checkpoints/                one file per finished (M, SU-position) task.
                              If the Kaggle session dies, just run again —
                              finished tasks are skipped.

Usage
  python scsi_simulations.py            full run
  python scsi_simulations.py --quick    tiny smoke test (~1-2 min)
  python scsi_simulations.py --plot     only redraw figures from scsi_results.npz
"""

import os
# One BLAS thread per worker process (we parallelise with multiprocessing).
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys
import glob
import json
import time


# ======================= LOCATE CODE + OUTPUT DIR =======================
def _find_code_dir():
    here = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    if os.path.exists(os.path.join(here, 'proposed_psm.py')):
        return here
    hits = glob.glob('/kaggle/input/**/proposed_psm.py', recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    raise FileNotFoundError("Could not find proposed_psm.py — attach the dataset "
                            "with the SCSI .py files to the notebook.")


CODE_DIR = _find_code_dir()
sys.path.insert(0, CODE_DIR)
# Codes/ layout: shared modules live in ../core (on Kaggle everything is flat)
sys.path.insert(0, os.path.join(CODE_DIR, '..', 'core'))
ON_KAGGLE = os.path.isdir('/kaggle/working')
OUT_DIR = '/kaggle/working/scsi_results' if ON_KAGGLE else os.path.join(CODE_DIR, '..', 'results', 'scsi')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats

from system_model import (generate_su_positions, get_all_path_losses,
                          N_SU, L_RIS, SIGMA2_S, SIGMA2_U, KAPPA)
from channel_model import generate_all_channels, compute_equivalent_channel
from energy_detection import (compute_threshold_egc, compute_threshold_and,
                              compute_threshold_or, compute_threshold_krank,
                              compute_pd_individual, compute_pd_egc,
                              compute_pd_and, compute_pd_or, compute_pd_krank)
from benchmark_psm import random_psm, benchmark_partitioned_psm
from proposed_psm import optimize_psm_scsi

# ============================== CONFIG ==============================
QUICK = '--quick' in sys.argv

# Near-square arrays only (Mh x Mv). Skewed shapes such as 52 = 4x13 or
# 148 = 4x37 beam ~25-30% worse and caused the kinks in the first run.
M_VALUES = [4, 16, 36, 48, 64, 80, 100, 120, 144, 168, 196]
N_SU_REALIZATIONS = 10      # SU drops (PSM is optimised once per drop)
N_CH_REALIZATIONS = 100     # fading realisations per drop (PSM fixed = S-CSI)
USE_SDR = True              # paper's SDR as an extra candidate (only for L*M <= 200)
N_RANDOMIZATION = 100       # Gaussian randomization candidates

if QUICK:
    M_VALUES = [16, 64]
    N_SU_REALIZATIONS = 2
    N_CH_REALIZATIONS = 10
    OUT_DIR = OUT_DIR + '_quick'

SCHEMES = ['EGC', 'OR', 'Majority', 'AND']
CONFIGS = ['proposed_scsi', 'partitioned', 'random', 'no_ris']
K_OF = {'EGC': None, 'OR': 1, 'Majority': 2, 'AND': N_SU}
CKPT_DIR = os.path.join(OUT_DIR, 'checkpoints')


def array_shape(M):
    Mh = int(np.sqrt(M))
    while M % Mh:
        Mh -= 1
    return Mh, M // Mh


def thresholds():
    return {'EGC': compute_threshold_egc(), 'AND': compute_threshold_and(),
            'OR': compute_threshold_or(), 'Majority': compute_threshold_krank(K=2)[0]}


def coop_pd(rho, scheme, gam):
    if scheme == 'EGC':
        return compute_pd_egc(rho, gam['EGC'])
    pd = compute_pd_individual(rho, gam[scheme])
    if scheme == 'AND':
        return compute_pd_and(pd)
    if scheme == 'OR':
        return compute_pd_or(pd)
    return compute_pd_krank(pd, K=2)


# ============================== WORKER ==============================
def run_task(task):
    """One (M, SU drop): optimise S-CSI PSMs once, average Pd over fading."""
    m_idx, M, su_real = task
    ckpt = os.path.join(CKPT_DIR, f'M{M:03d}_su{su_real:02d}.json')
    if os.path.exists(ckpt):
        with open(ckpt) as fh:
            return json.load(fh)

    t0 = time.time()
    gam = thresholds()
    Mh, Mv = array_shape(M)
    su_pos = generate_su_positions(seed=su_real * 100 + 7)
    eta, beta, zeta = get_all_path_losses(su_pos)

    # LOS parts (f, g_bar) are deterministic given positions -> S-CSI input
    _, f_los, _, g_bar = generate_all_channels(su_pos, eta, beta, zeta, Mh, Mv,
                                               KAPPA, np.random.default_rng(0))
    phi_prop = {}
    for s in SCHEMES:
        phi_prop[s] = optimize_psm_scsi(s, f_los, g_bar, N_SU, L_RIS, M, K=K_OF[s],
                                        n_randomization=N_RANDOMIZATION,
                                        rng=np.random.default_rng(su_real + 777),
                                        use_sdr=USE_SDR)
    phi_part = benchmark_partitioned_psm(f_los, g_bar, M, N_SU, L_RIS)
    t_opt = time.time() - t0

    acc = {s: {c: 0.0 for c in CONFIGS} for s in SCHEMES}
    for ch in range(N_CH_REALIZATIONS):
        seed = su_real * 10000 + ch * 50 + m_idx + 31415
        d, f, g, _ = generate_all_channels(su_pos, eta, beta, zeta, Mh, Mv, KAPPA,
                                           np.random.default_rng(seed))
        rho_rand = compute_equivalent_channel(
            d, f, g, random_psm(M, L_RIS, np.random.default_rng(seed + 99999)))[1]
        rho_part = compute_equivalent_channel(d, f, g, phi_part)[1]
        rho_none = np.abs(d) ** 2
        for s in SCHEMES:
            rho_prop = compute_equivalent_channel(d, f, g, phi_prop[s])[1]
            acc[s]['proposed_scsi'] += coop_pd(rho_prop, s, gam)
            acc[s]['partitioned'] += coop_pd(rho_part, s, gam)
            acc[s]['random'] += coop_pd(rho_rand, s, gam)
            acc[s]['no_ris'] += coop_pd(rho_none, s, gam)

    out = {'M': M, 'su_real': su_real, 'n_ch': N_CH_REALIZATIONS,
           't_opt': t_opt, 't_total': time.time() - t0,
           'pd': {s: {c: acc[s][c] / N_CH_REALIZATIONS for c in CONFIGS} for s in SCHEMES}}
    tmp = ckpt + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(out, fh)
    os.replace(tmp, ckpt)
    return out


# ============================== FIG. 4 / 5 DATA ==============================
def simulate_pd_vs_M():
    import multiprocessing as mp
    os.makedirs(CKPT_DIR, exist_ok=True)
    tasks = [(i, M, s) for i, M in enumerate(M_VALUES) for s in range(N_SU_REALIZATIONS)]
    tasks.sort(key=lambda t: -t[1])          # big M first -> better load balance
    n_workers = max(1, min(os.cpu_count() or 1, len(tasks)))
    done_before = sum(os.path.exists(os.path.join(CKPT_DIR, f'M{M:03d}_su{s:02d}.json'))
                      for _, M, s in tasks)
    print(f"\n[Fig 4/5] {len(tasks)} tasks ({done_before} already checkpointed), "
          f"{n_workers} workers, M = {M_VALUES}", flush=True)

    results = []
    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        for k, r in enumerate(pool.imap_unordered(run_task, tasks), 1):
            results.append(r)
            p = r['pd']
            print(f"  [{k:3d}/{len(tasks)}] M={r['M']:3d} su={r['su_real']:2d} "
                  f"opt={r['t_opt']:6.1f}s | " +
                  "  ".join(f"{s}:{p[s]['proposed_scsi']:.3f}/{p[s]['partitioned']:.3f}"
                            for s in SCHEMES) +
                  f"   (elapsed {(time.time()-t0)/60:.1f} min)", flush=True)

    pd = {s: {c: np.zeros(len(M_VALUES)) for c in CONFIGS} for s in SCHEMES}
    cnt = np.zeros(len(M_VALUES))
    for r in results:
        i = M_VALUES.index(r['M'])
        cnt[i] += 1
        for s in SCHEMES:
            for c in CONFIGS:
                pd[s][c][i] += r['pd'][s][c]
    for s in SCHEMES:
        for c in CONFIGS:
            pd[s][c] /= cnt
    save_results(pd)
    return pd


def save_results(pd):
    os.makedirs(OUT_DIR, exist_ok=True)
    flat = {f'{s}__{c}': pd[s][c] for s in SCHEMES for c in CONFIGS}
    np.savez(os.path.join(OUT_DIR, 'scsi_results.npz'), M=np.array(M_VALUES),
             n_su=N_SU_REALIZATIONS, n_ch=N_CH_REALIZATIONS, **flat)
    with open(os.path.join(OUT_DIR, 'scsi_results.csv'), 'w') as fh:
        cols = list(flat)
        fh.write('M,' + ','.join(cols) + '\n')
        for i, M in enumerate(M_VALUES):
            fh.write(f'{M},' + ','.join(f'{flat[c][i]:.6f}' for c in cols) + '\n')
    print(f"Saved raw numbers: {OUT_DIR}/scsi_results.npz and .csv")


def load_results():
    z = np.load(os.path.join(OUT_DIR, 'scsi_results.npz'))
    global M_VALUES
    M_VALUES = [int(m) for m in z['M']]
    return {s: {c: z[f'{s}__{c}'] for c in CONFIGS} for s in SCHEMES}


# ============================== PLOTS ==============================
def _style():
    rcParams.update({'font.family': 'serif', 'font.size': 12, 'axes.grid': True,
                     'grid.alpha': 0.3, 'figure.dpi': 120})


def _axes(ax, legend_loc='lower right', anchor=None):
    ax.set_xlabel('Number of REs per RIS, $M$')
    ax.set_ylabel('Cooperative Detection Probability, $P_d$')
    ax.set_xlim([0, 200]); ax.set_ylim([0, 1.05])
    ax.set_xticks(np.arange(0, 201, 50)); ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.legend(fontsize=9, loc=legend_loc, bbox_to_anchor=anchor)


def plot_fig4(pd):
    _style()
    titles = {'EGC': '(a) Data Fusion, EGC',
              'OR': '(b) Decision Fusion, OR Rule ($K=1$)',
              'Majority': '(c) Decision Fusion, Majority Rule ($K=2$)',
              'AND': '(d) Decision Fusion, AND Rule ($K=4$)'}
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, s in zip(axes.flat, SCHEMES):
        ax.plot(M_VALUES, pd[s]['proposed_scsi'], 'g-^', ms=6, lw=1.8, mfc='g',
                label='Proposed opt. $\\Phi_l$, S-CSI')
        ax.plot(M_VALUES, pd[s]['partitioned'], 'b-s', ms=6, lw=1.8, mfc='b',
                label='Partitioned opt. $\\Phi_l$, S-CSI')
        ax.plot(M_VALUES, pd[s]['random'], 'k--o', ms=6, lw=1.5, mfc='k',
                label='Random $\\Phi_l$')
        ax.plot(M_VALUES, pd[s]['no_ris'], ':', color='gray', lw=1.5, label='No RIS')
        ax.set_title(titles[s])
        # lifted above the Random / No-RIS curves (which sit at Pd ~ 0.2)
        _axes(ax, 'lower right', anchor=(1.0, 0.25))
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'fig4_scsi_pd_vs_M.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('Saved', p)


def plot_fig5(pd):
    _style()
    fig, ax = plt.subplots(figsize=(8, 6))
    for s, st, lab in [('EGC', 'b-s', 'EGC'), ('OR', 'r-o', 'OR Rule, $K=1$'),
                       ('Majority', 'g-^', 'Majority Rule, $K=2$'),
                       ('AND', 'k-D', 'AND Rule, $K=4$')]:
        ax.plot(M_VALUES, pd[s]['proposed_scsi'], st, ms=6, lw=1.6,
                label=lab + ' (Proposed S-CSI)')
    _axes(ax)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'fig5_scsi_css_comparison.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('Saved', p)


# ============================== FIG. 3: GAMMA APPROX ==============================
def simulate_gamma_validation(M_list=(16, 64, 144), n_samples=20000, su_seed=7):
    """
    rho = ||h||^2 = sum_n |h_n|^2 with the S-CSI (EGC) PSM held fixed.
    Under S-CSI, h_n = mu_n + w_n, where
        mu_n    = sum_l g_bar_{n,l}^H Phi_l f_l                (deterministic)
        w_n     ~ CN(0, s_n),  s_n = eta_n + sum_l zeta_{n,l}/(1+kappa) ||f_l||^2
    (d_n and the NLOS parts are independent, zero-mean Gaussian).
    Moments (Proposition 3):
        E[rho]   = sum_n (|mu_n|^2 + s_n)
        Var[rho] = sum_n (s_n^2 + 2 s_n |mu_n|^2)
    Gamma fit by moment matching: k = E^2/Var, theta = Var/E.
    x-axis shown as received SNR  rho * sigma_s^2 / sigma_u^2.
    """
    print("\n[Fig 3] Gamma approximation of ||h||^2", flush=True)
    _style()
    rng = np.random.default_rng(2024)
    su_pos = generate_su_positions(seed=su_seed)
    eta, beta, zeta = get_all_path_losses(su_pos)
    snr = SIGMA2_S / SIGMA2_U
    fig, axes = plt.subplots(1, len(M_list), figsize=(5.2 * len(M_list), 4.4))
    for ax, M in zip(np.atleast_1d(axes), M_list):
        Mh, Mv = array_shape(M)
        _, f, _, gb = generate_all_channels(su_pos, eta, beta, zeta, Mh, Mv, KAPPA,
                                            np.random.default_rng(0))
        phi = optimize_psm_scsi('EGC', f, gb, N_SU, L_RIS, M, use_sdr=False,
                                rng=np.random.default_rng(1))
        pf = [phi[l] * f[l] for l in range(L_RIS)]            # Phi_l f_l
        rho = np.zeros(n_samples)
        mu = np.zeros(N_SU, complex); s = np.zeros(N_SU)
        for n in range(N_SU):
            mu[n] = sum(np.conj(gb[n][l]) @ pf[l] for l in range(L_RIS))
            s[n] = eta[n] + sum(zeta[n, l] / (1 + KAPPA) * np.linalg.norm(f[l]) ** 2
                                for l in range(L_RIS))
            # empirical: draw the actual random channels (d_n and NLOS vectors)
            h = np.sqrt(eta[n] / 2) * (rng.standard_normal(n_samples)
                                       + 1j * rng.standard_normal(n_samples)) + mu[n]
            for l in range(L_RIS):
                var = zeta[n, l] / (1 + KAPPA)
                G = np.sqrt(var / 2) * (rng.standard_normal((n_samples, M))
                                        + 1j * rng.standard_normal((n_samples, M)))
                h += G.conj() @ pf[l]
            rho += np.abs(h) ** 2
        mean = np.sum(np.abs(mu) ** 2 + s)
        var = np.sum(s ** 2 + 2 * s * np.abs(mu) ** 2)
        k, theta = mean ** 2 / var, var / mean

        x_emp = rho * snr
        xs = np.linspace(x_emp.min() * 0.9, x_emp.max() * 1.05, 400)
        ax.hist(x_emp, bins=80, density=True, color='#9ecae1', edgecolor='white',
                label='Empirical (Monte Carlo)')
        ax.plot(xs, stats.gamma.pdf(xs, a=k, scale=theta * snr), 'r-', lw=2,
                label=f'Gamma approx. ($k$={k:.1f})')
        ax.plot(xs, stats.norm.pdf(xs, mean * snr, np.sqrt(var) * snr), 'k--', lw=1.6,
                label='Gaussian approx.')
        ks_g = stats.kstest(rho, 'gamma', args=(k, 0, theta)).statistic
        ks_n = stats.kstest(rho, 'norm', args=(mean, np.sqrt(var))).statistic
        print(f"  M={M:3d}: E[rho] analytic={mean:.3e} empirical={rho.mean():.3e} | "
              f"Var analytic={var:.3e} empirical={rho.var():.3e} | "
              f"KS gamma={ks_g:.4f} gaussian={ks_n:.4f}", flush=True)
        ax.set_title(f'$M$ = {M} ({Mh}$\\times${Mv}),  KS: $\\Gamma$={ks_g:.3f}, '
                     f'$\\mathcal{{N}}$={ks_n:.3f}', fontsize=11)
        ax.set_xlabel('$\\|\\mathbf{h}\\|^2\\sigma_s^2/\\sigma_u^2$')
        ax.set_ylabel('PDF')
        ax.legend(fontsize=9)
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, 'fig3_gamma_validation.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('Saved', p)


# ============================== MAIN ==============================
if __name__ == '__main__':
    print("=" * 70)
    print(f"S-CSI simulations | code: {CODE_DIR}\n                    | output: {OUT_DIR}")
    print(f"quick={QUICK}  M={M_VALUES}  drops={N_SU_REALIZATIONS}  "
          f"fading/drop={N_CH_REALIZATIONS}  SDR={USE_SDR}  cpus={os.cpu_count()}")
    print("=" * 70, flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    if '--plot' in sys.argv:
        pd = load_results()
    else:
        simulate_gamma_validation(M_list=(16, 64) if QUICK else (16, 64, 144),
                                  n_samples=2000 if QUICK else 20000)
        pd = simulate_pd_vs_M()
    plot_fig4(pd)
    plot_fig5(pd)

    print("\nSummary (Proposed S-CSI Pd):")
    print("   M  " + "".join(f"{s:>10}" for s in SCHEMES))
    for i, M in enumerate(M_VALUES):
        print(f"{M:4d}  " + "".join(f"{pd[s]['proposed_scsi'][i]:10.4f}" for s in SCHEMES))
    print("\nDone. Download the folder:", OUT_DIR)
