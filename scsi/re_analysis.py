"""
re_analysis.py — Phase 8: how many REs are needed for a target Pd? (Sec. IV-C)
===============================================================================
Ge et al., "RIS-Assisted Cooperative Spectrum Sensing for CRNs", IEEE TWC 2024

Method (exactly the paper's Section IV-C, statistical-CSI case):
  1. For a given M, optimise the PSM with S-CSI (LOS only) — or use the
     partitioned benchmark — for each SU drop.
  2. Equivalent channel  h_n ~ CN(c_n, eta_n + b_n)                    (Eq. 36)
         c_n = g_bar_n^H Phi f,     b_n = M * sum_l zeta_{n,l} beta_l / (1+kappa)
  3. Gamma approximation by moment matching
       decision fusion:  rho_n = |h_n|^2,  mu = |c_n|^2 + eta_n + b_n,
                         v = (eta_n+b_n)^2 + 2|c_n|^2 (eta_n+b_n)    (Eq. 37-38)
       EGC:              rho = ||h||^2, moments of Proposition 3     (Eq. 39-41)
  4. Average detection probability by numerical integration
       P_d,n = E_rho_n[ Q((gamma_n - mu_E(rho_n)) / std_E(rho_n)) ]   (Eq. 35)
       P_d^EGC = E_rho[ Pd_EGC(rho) ]   with Proposition 1               (Eq. 46)
     then decision fusion via Table I (K-rank, K = 1..N).
  5. Bisection over M to find the smallest M with  P_d >= 0.9.

Outputs (OUT_DIR):
  fig_validation_analytic_vs_mc.png  analytic (Gamma + integration) vs. the
                                     Monte-Carlo Fig. 4 results -> proves step 3-4
  fig8_min_M_vs_T.png                minimum M per RIS vs. sensing samples T
  fig9_min_M_vs_T_kappa.png          AND rule, effect of the Rician factor
  re_results.json                    all raw numbers

Usage:
  python re_analysis.py            full run (local, ~10-40 min on a multi-core PC)
  python re_analysis.py --quick    smoke test
  python re_analysis.py --plot     redraw figures from re_results.json
"""

import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import sys
import glob
import json
import time


def _find_code_dir():
    here = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    if os.path.exists(os.path.join(here, 'proposed_psm.py')):
        return here
    hits = glob.glob('/kaggle/input/**/proposed_psm.py', recursive=True)
    if hits:
        return os.path.dirname(hits[0])
    raise FileNotFoundError("proposed_psm.py not found")


CODE_DIR = _find_code_dir()
sys.path.insert(0, CODE_DIR)
# Codes/ layout: shared modules live in ../core (on Kaggle everything is flat)
sys.path.insert(0, os.path.join(CODE_DIR, '..', 'core'))
ON_KAGGLE = os.path.isdir('/kaggle/working')
OUT_DIR = '/kaggle/working/re_results' if ON_KAGGLE else os.path.join(CODE_DIR, '..', 'results', 'phase8')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats

from system_model import (generate_su_positions, get_all_path_losses,
                          N_SU, L_RIS, SIGMA2_S, SIGMA2_U, P_FA_BAR, KAPPA)
from channel_model import generate_all_channels
from energy_detection import (compute_threshold_egc, compute_threshold_or,
                              compute_threshold_and, compute_threshold_krank,
                              compute_pd_individual, compute_pd_krank)
from benchmark_psm import benchmark_partitioned_psm
from proposed_psm import optimize_psm_scsi

# ============================== CONFIG ==============================
QUICK = '--quick' in sys.argv
TARGET_PD = 0.9
N_DROPS = 10                         # same SU drops as the Monte-Carlo run
# short sensing times (100-2000) + the paper's Fig. 8/9 range (up to 5e4)
T_VALUES = [100, 200, 300, 500, 700, 1000, 2000, 5000, 10000, 15000, 20000,
            25000, 30000, 35000, 40000, 45000, 50000]
KAPPAS_FIG9 = [1, 2, 5, 10]          # same Rician factors as the paper's Fig. 9
M_MAX = 1600
N_QUAD = 2000                        # quantile points for the outer integral
VALID_T = 1000
VALID_M = [4, 16, 36, 48, 64, 80, 100, 120, 144, 168, 196]   # = Fig. 4 grid

# scheme -> K of the K-rank rule (None = EGC data fusion)
SCHEME_K = {'EGC': None, 'OR': 1, 'Majority': 2, 'K3': 3, 'AND': 4}
SCHEME_LABEL = {'EGC': 'EGC', 'OR': 'OR ($K$=1)', 'Majority': 'Majority ($K$=2)',
                'K3': 'Krank ($K$=3)', 'AND': 'AND ($K$=4)'}
SCHEME_COLOR = {'EGC': 'tab:blue', 'OR': 'tab:red', 'Majority': 'tab:green',
                'K3': 'tab:purple', 'AND': 'k'}

if QUICK:
    N_DROPS = 2
    T_VALUES = [300, 1000, 20000]
    KAPPAS_FIG9 = [5]
    M_MAX = 400
    N_QUAD = 500
    VALID_M = [16, 64, 144]
    OUT_DIR += '_quick'


def build_shapes(m_max):
    """
    Allowed RIS sizes: M divisible by N=4 (partitioned benchmark) with a
    reasonably square Mh x Mv factorisation (Mv/Mh <= MAX_ASPECT). Very
    skewed arrays (e.g. 4x37) beam much worse, which would break the
    monotonicity bisection relies on. MAX_ASPECT = 2 keeps the grid fine
    at small M (8, 28, 32, 40, ... are allowed), where Fig. 8's tail lives.
    """
    shapes = []
    for M in range(4, m_max + 1, 4):
        best = None
        for Mh in range(int(np.sqrt(M)), 0, -1):
            if M % Mh == 0:
                Mv = M // Mh
                if Mv / Mh <= MAX_ASPECT:
                    best = (Mh, Mv)
                break
        if best:
            shapes.append((M,) + best)
    return shapes


MAX_ASPECT = 2.0
SHAPES = build_shapes(M_MAX)
SHAPE_OF = {M: (Mh, Mv) for M, Mh, Mv in SHAPES}


def _shape_any(M):
    if M in SHAPE_OF:
        return SHAPE_OF[M]
    Mh = int(np.sqrt(M))
    while M % Mh:
        Mh -= 1
    return Mh, M // Mh


# ============================== DETECTION FORMULAS ==============================
def threshold(scheme, T):
    K = SCHEME_K[scheme]
    if K is None:
        return compute_threshold_egc(T=T)
    if K == 1:
        return compute_threshold_or(T=T)
    if K == N_SU:
        return compute_threshold_and(T=T)
    return compute_threshold_krank(K, T=T)[0]


def pd_egc_given_rho(rho, gamma, T, N=N_SU):
    """Proposition 1 (vectorised over rho = ||h||^2)."""
    mean = SIGMA2_U + rho / N * SIGMA2_S
    var = (SIGMA2_U ** 2 / (N * T)
           + (rho ** 2 * SIGMA2_S ** 2 + 2 * rho * SIGMA2_S * SIGMA2_U) / (N ** 2 * T))
    return stats.norm.sf((gamma - mean) / np.sqrt(var))


def gamma_expectation(fun, mean, var, n_q=None):
    """E[fun(rho)] for rho ~ Gamma(k = mean^2/var, theta = var/mean),
    by the midpoint rule in probability space (quantile integration)."""
    n_q = n_q or N_QUAD
    u = (np.arange(n_q) + 0.5) / n_q
    rho = stats.gamma.ppf(u, a=mean ** 2 / var, scale=var / mean)
    return float(np.mean(fun(rho)))


def avg_pd(scheme, c, eta, b, T):
    """
    Average cooperative Pd for ONE SU drop under S-CSI.
    c: (N,) LOS cascaded gains c_n; eta: (N,) direct path loss; b: (N,) NLOS power.
    """
    gam = threshold(scheme, T)
    a = eta + b                                   # variance of the random part of h_n
    c2 = np.abs(c) ** 2
    if SCHEME_K[scheme] is None:                  # EGC — Proposition 3 + Eq. (46)
        mu = np.sum(c2 + a)
        v = np.sum(a ** 2 + 2 * a * c2)
        return gamma_expectation(lambda r: pd_egc_given_rho(r, gam, T), mu, v)
    pdn = np.array([gamma_expectation(lambda r: compute_pd_individual(r, gam, T=T),
                                      c2[n] + a[n], a[n] ** 2 + 2 * a[n] * c2[n])
                    for n in range(N_SU)])        # Eq. (35) with Eq. (37)-(38)
    return float(compute_pd_krank(pdn, SCHEME_K[scheme]))   # Table I


# ============================== EVALUATOR (one job) ==============================
def drop_gains(args):
    """
    Worker: optimise the PSM for ONE SU drop at size M and return the LOS
    cascaded gains c_n, direct path loss eta_n and NLOS power b_n.
    Runs in the process pool, so the 10 drops of an M value run in parallel.
    """
    config, scheme, kappa, M, s = args
    su = generate_su_positions(seed=s * 100 + 7)                # same drops as MC run
    eta, beta, zeta = get_all_path_losses(su)
    Mh, Mv = _shape_any(M)
    _, f, _, gb = generate_all_channels(su, eta, beta, zeta, Mh, Mv,
                                        kappa, np.random.default_rng(0))
    if config == 'proposed':
        K = SCHEME_K[scheme]
        name = {None: 'EGC', 1: 'OR', N_SU: 'AND'}.get(K, 'Majority')
        phi = optimize_psm_scsi(name, f, gb, N_SU, L_RIS, M, K=K,
                                rng=np.random.default_rng(s + 777), use_sdr=False)
    else:
        phi = benchmark_partitioned_psm(f, gb, M, N_SU, L_RIS)
    c = np.array([sum(np.conj(gb[n][l]) @ (phi[l] * f[l]) for l in range(L_RIS))
                  for n in range(N_SU)])
    b = M * np.array([sum(zeta[n, l] * beta[l] / (1 + kappa) for l in range(L_RIS))
                      for n in range(N_SU)])
    return c, eta, b


class Evaluator:
    """Caches the optimised LOS gains c_n for each (M, drop); T is cheap to vary."""

    def __init__(self, config, scheme, kappa, pool=None):
        self.config, self.scheme, self.kappa = config, scheme, kappa
        self.pool = pool
        self.cache = {}
        self.n_opt = 0

    def gains(self, M):
        if M in self.cache:
            return self.cache[M]
        args = [(self.config, self.scheme, self.kappa, M, s) for s in range(N_DROPS)]
        out = self.pool.map(drop_gains, args) if self.pool else list(map(drop_gains, args))
        self.n_opt += 1
        self.cache[M] = out
        return out

    def pd(self, M, T):
        return float(np.mean([avg_pd(self.scheme, c, eta, b, T)
                              for c, eta, b in self.gains(M)]))

    def min_M(self, T, target=TARGET_PD):
        """
        Bisection over the ordered list of allowed sizes (Sec. IV-C), then
        linear interpolation of Pd between the two bracketing sizes to find
        where Pd crosses the target. Both Pd values are already computed by
        the bisection, so this costs nothing; it removes the staircase that
        a coarse size grid would otherwise put on the curves.
        Returns (M_crossing rounded to 0.1, smallest allowed M on the grid).
        """
        Ms = [s[0] for s in SHAPES]
        if self.pd(Ms[-1], T) < target:
            return None, None                            # more than M_MAX needed
        if self.pd(Ms[0], T) >= target:
            return float(Ms[0]), Ms[0]
        lo, hi = 0, len(Ms) - 1                          # pd(lo) < target <= pd(hi)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if self.pd(Ms[mid], T) >= target:
                hi = mid
            else:
                lo = mid
        p_lo, p_hi = self.pd(Ms[lo], T), self.pd(Ms[hi], T)
        frac = (target - p_lo) / (p_hi - p_lo) if p_hi > p_lo else 1.0
        return round(Ms[lo] + frac * (Ms[hi] - Ms[lo]), 1), Ms[hi]


def run_job(job, pool=None):
    config, scheme, kappa, do_valid = job
    t0 = time.time()
    ev = Evaluator(config, scheme, kappa, pool)
    res = {'config': config, 'scheme': scheme, 'kappa': kappa,
           'T': T_VALUES}
    mm = [ev.min_M(T) for T in T_VALUES]
    res['min_M'] = [m[0] for m in mm]              # interpolated crossing (plotted)
    res['min_M_grid'] = [m[1] for m in mm]         # smallest allowed array size
    res['max_aspect'] = MAX_ASPECT
    if do_valid:
        res['valid_M'] = VALID_M
        res['valid_pd'] = [ev.pd(M, VALID_T) for M in VALID_M]
    res['n_M_evaluated'] = ev.n_opt
    res['time_s'] = time.time() - t0
    print(f"  done {config:11s} {scheme:8s} kappa={kappa:<4} "
          f"min_M={res['min_M']}  ({res['time_s']/60:.1f} min, {ev.n_opt} M values)",
          flush=True)
    return res


# ============================== PLOTS ==============================
def _style():
    rcParams.update({'font.family': 'serif', 'font.size': 12, 'axes.grid': True,
                     'grid.alpha': 0.3, 'figure.dpi': 120})


def _find_mc():
    for p in [os.path.join(CODE_DIR, '..', 'results', 'scsi', 'scsi_results.npz'),
              os.path.join(CODE_DIR, 'scsi_results', 'scsi_results', 'scsi_results.npz'),
              os.path.join(CODE_DIR, 'results_scsi', 'scsi_results.npz'),
              '/kaggle/working/scsi_results/scsi_results.npz']:
        if os.path.exists(p):
            return np.load(p)
    hits = glob.glob('/kaggle/input/**/scsi_results.npz', recursive=True)
    return np.load(hits[0]) if hits else None


def plot_validation(results):
    _style()
    mc = _find_mc()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    cfg_key = {'proposed': 'proposed_scsi', 'partitioned': 'partitioned'}
    sty = {'proposed': ('g', '^', 'Proposed S-CSI'), 'partitioned': ('b', 's', 'Partitioned')}
    for ax, s in zip(axes.flat, ['EGC', 'OR', 'Majority', 'AND']):
        for r in results:
            if r['scheme'] == s and r['kappa'] == KAPPA and 'valid_pd' in r:
                col, mk, lab = sty[r['config']]
                ax.plot(r['valid_M'], r['valid_pd'], '-', color=col, lw=1.8,
                        label=f'{lab}: analytic (Gamma + num. integration)')
                if mc is not None:
                    ax.plot(mc['M'], mc[f"{s}__{cfg_key[r['config']]}"], mk, color=col,
                            mfc='none', ms=8, mew=1.6, label=f'{lab}: Monte Carlo')
        ax.set_title(f"{SCHEME_LABEL[s]},  $T$ = {VALID_T}")
        ax.set_xlabel('Number of REs per RIS, $M$')
        ax.set_ylabel('Cooperative $P_d$')
        ax.set_xlim(0, 200); ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8.5, loc='lower right')
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'fig_validation_analytic_vs_mc.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('Saved', p)


def _series(r, paper_style):
    T = np.array(r['T'], float)
    y = np.array([np.nan if m is None else m for m in r['min_M']], float)
    if paper_style:          # the paper's sample points: T = 500, 5000, 10000, ..., 50000
        keep = (T == 500) | ((T >= 5000) & (T % 5000 == 0))
    else:
        keep = np.ones_like(T, bool)
    return T[keep], y[keep]


# ---------- log-scale versions (all T values, incl. short sensing times) ----------
def _log_axes(ax):
    ax.set_xscale('log')
    ticks = [t for t in [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
             if min(T_VALUES) <= t <= max(T_VALUES)]
    ax.set_xticks(ticks); ax.set_xticklabels([f'{t:,}' for t in ticks])
    ax.set_xlabel('Number of sensing signal samples, $T$ (log scale)')
    ax.set_ylabel(rf'Number of required REs per RIS for $\bar P_d$ = {TARGET_PD}')


# ---------- paper-style versions: same look as Figs. 8/9 of Ge et al. ----------
# MATLAB-like colours/markers, hollow markers, solid = proposed, dashed = benchmark
PAPER_FIG8 = [('OR', 'Krank-1', 'r', 'o'), ('Majority', 'Krank-2', '#00cc00', 'd'),
              ('K3', 'Krank-3', 'm', 's'), ('AND', 'Krank-4', 'b', '*'),
              ('EGC', 'EGC', 'c', '^')]
PAPER_FIG9 = {1: ('r', 'o'), 2: ('#00cc00', 'd'), 5: ('m', 's'), 10: ('b', '*')}


def _paper_axes(ax, y_lo, y_hi):
    ax.set_xlim(0, 50000)
    ax.set_xticks(np.arange(0, 50001, 5000))
    ax.set_xticklabels([f'{t / 1e4:g}' for t in np.arange(0, 50001, 5000)])
    ax.text(1.0, -0.085, r'$\times 10^4$', transform=ax.transAxes, ha='right', fontsize=10)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel('Number of Sensing Signal Samples, $T$')
    ax.set_ylabel('Number of Required REs per RIS, $M$')
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=9, loc='upper right', framealpha=1, edgecolor='0.3')


def _paper_line(ax, r, col, mk, label, prop):
    ax.plot(*_series(r, True), '-' if prop else '--', marker=mk, color=col, lw=1.1,
            ms=7 if mk == '*' else 5.5, mfc='none', mew=1.1, label=label)


def _ceil50(v):
    return int(np.ceil(np.nanmax(v) / 50.0) * 50)


def plot_fig8(results):
    _style()
    # log-scale version
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for s in SCHEME_K:
        for r in results:
            if r['scheme'] == s and r['kappa'] == KAPPA:
                prop = r['config'] == 'proposed'
                ax.plot(*_series(r, False), '-' if prop else '--',
                        marker='o' if prop else 's', color=SCHEME_COLOR[s], ms=5, lw=1.7,
                        mfc=SCHEME_COLOR[s] if prop else 'none',
                        label=f"{SCHEME_LABEL[s]}, {'proposed' if prop else 'partitioned'}")
    _log_axes(ax)
    ax.legend(fontsize=8.5, ncol=2)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'fig8_min_M_vs_T.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('Saved', p)

    # paper-style version
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ymax = 0
    for s, name, col, mk in PAPER_FIG8:
        for cfg, word in [('proposed', 'proposed'), ('partitioned', 'benchmark')]:
            for r in results:
                if r['scheme'] == s and r['config'] == cfg and r['kappa'] == KAPPA:
                    _paper_line(ax, r, col, mk, f'{name}, {word}', cfg == 'proposed')
                    ymax = max(ymax, np.nanmax(_series(r, True)[1]))
    _paper_axes(ax, 0, max(250, _ceil50([ymax])))
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'fig8_min_M_vs_T_paper_style.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('Saved', p)


def plot_fig9(results):
    _style()
    ks = sorted({r['kappa'] for r in results if r['scheme'] == 'AND'})
    # log-scale version
    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    for i, k in enumerate(ks):
        col = cmap(i / max(1, len(ks) - 1) * 0.9)
        for r in results:
            if r['scheme'] == 'AND' and r['kappa'] == k:
                prop = r['config'] == 'proposed'
                ax.plot(*_series(r, False), '-' if prop else '--',
                        marker='o' if prop else 's', color=col, ms=5, lw=1.7,
                        mfc=col if prop else 'none',
                        label=rf"$\kappa$ = {k:g}, {'proposed' if prop else 'partitioned'}")
    _log_axes(ax)
    ax.set_title(r'Logic-AND fusion: impact of the Rician factor $\kappa$')
    ax.legend(fontsize=8.5, ncol=2)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'fig9_min_M_vs_T_kappa.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('Saved', p)

    # paper-style version (kappa = 1, 2, 5, 10 like the paper)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ys = []
    for k in [k for k in ks if k in PAPER_FIG9]:
        col, mk = PAPER_FIG9[k]
        for cfg, word in [('proposed', 'proposed'), ('partitioned', 'benchmark')]:
            for r in results:
                if r['scheme'] == 'AND' and r['config'] == cfg and r['kappa'] == k:
                    _paper_line(ax, r, col, mk, rf'$\kappa$ = {k:g}, {word}', cfg == 'proposed')
                    ys.append(_series(r, True)[1])
    if ys:
        allv = np.concatenate(ys)
        _paper_axes(ax, min(50, int(np.nanmin(allv) // 50 * 50)), max(350, _ceil50(allv)))
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'fig9_min_M_vs_T_kappa_paper_style.png')
    plt.savefig(p, dpi=150, bbox_inches='tight'); plt.close()
    print('Saved', p)


def print_tables(results):
    print(f"\nMinimum M per RIS for Pd >= {TARGET_PD}  (kappa = {KAPPA:g})")
    print(f"{'scheme':<10}{'config':<13}" + ''.join(f'{t:>7}' for t in T_VALUES))
    for s in SCHEME_K:
        for cfg in ['proposed', 'partitioned']:
            for r in results:
                if r['scheme'] == s and r['config'] == cfg and r['kappa'] == KAPPA:
                    print(f"{s:<10}{cfg:<13}" + ''.join(
                        f"{(f'{m:.0f}' if m else '>' + str(M_MAX)):>7}" for m in r['min_M']))


# ============================== MAIN ==============================
if __name__ == '__main__':
    import multiprocessing as mp
    os.makedirs(OUT_DIR, exist_ok=True)
    jpath = os.path.join(OUT_DIR, 're_results.json')
    print(f"RE analysis | out: {OUT_DIR} | quick={QUICK} | drops={N_DROPS} | "
          f"T={T_VALUES} | {len(SHAPES)} allowed M values up to {M_MAX}", flush=True)

    if '--plot' in sys.argv:
        with open(jpath) as fh:
            results = json.load(fh)
    else:
        jobs = [(cfg, s, KAPPA, s in ('EGC', 'OR', 'Majority', 'AND'))
                for s in SCHEME_K for cfg in ('proposed', 'partitioned')]
        jobs += [(cfg, 'AND', k, False) for k in KAPPAS_FIG9 if k != KAPPA
                 for cfg in ('proposed', 'partitioned')]
        # Jobs run one after another; inside each job the 10 SU drops of every
        # M value are optimised in parallel, so all cores stay busy even for
        # the slow proposed K-rank jobs (one sub-problem per K-subset).
        # Checkpoint: every finished job is saved immediately; a rerun skips it.
        ck_dir = os.path.join(OUT_DIR, 'checkpoints')
        os.makedirs(ck_dir, exist_ok=True)

        def _ck(j):
            return os.path.join(ck_dir, f'{j[0]}_{j[1]}_k{j[2]:g}.json')

        t0 = time.time()
        results = []
        with mp.Pool(min(N_DROPS, os.cpu_count() or 1)) as pool:
            for j in jobs:
                if os.path.exists(_ck(j)):
                    with open(_ck(j)) as fh:
                        old = json.load(fh)
                    if old.get('T') == T_VALUES and old.get('max_aspect') == MAX_ASPECT:
                        results.append(old)
                        print(f"  skip (checkpoint) {j[0]} {j[1]} kappa={j[2]:g}", flush=True)
                        continue
                r = run_job(j, pool)
                with open(_ck(j), 'w') as fh:
                    json.dump(r, fh)
                results.append(r)
        print(f"All jobs done in {(time.time()-t0)/60:.1f} min")
        with open(jpath, 'w') as fh:
            json.dump(results, fh, indent=1)
        print('Saved', jpath)

    plot_validation(results)
    plot_fig8(results)
    plot_fig9(results)
    print_tables(results)
