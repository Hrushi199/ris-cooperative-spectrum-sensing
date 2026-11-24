"""
proposed_psm.py  (v3 — SCSI final run)
=======================================
Proposed PSM optimization with Statistical CSI (S-CSI), Section IV-B of
Ge et al., IEEE TWC 2024.

  - P-S-EGC   (Eq. 42): max ||G_bar^H Phi f||^2                 (sum of gains)
  - P-S-AND   (Eq. 43): max min_n |g_bar_n^H Phi f|^2            (worst SU)
  - P-S-OR    (Eq. 44): max max_n |g_bar_n^H Phi f|^2            (best SU)
  - P-S-Krank (Eq. 45): max over K-subsets of min within subset

Under S-CSI only the LOS parts g_bar are known and d_n is treated as unknown
(zero-mean), so u_n = [F^H g_bar_n ; 0].

Solver for every scheme = best of several candidates, each scored on the
ORIGINAL objective:
  1. SDR (the paper's method: SDP relaxation + Gaussian randomization)
     - used when L*M <= SDR_MAX_LM (SDP cost explodes for large L*M)
  2. Dominant / weighted eigenvector projected to unit modulus
  3. Span search for max-min (AND / K-rank)                    <-- new in v3
  4. Gauss-Seidel element-wise refinement (monotone ascent)    <-- new in v3

Changes vs v2 (the version used for the first S-CSI Kaggle run):
  * v2's AND solver relied on SDR + weighted eigenvector. The weighted
    eigenvector alone is 3-10x worse than SDR on the min-gain, and v2's
    K-rank (Majority) solver used ONLY that heuristic -> Majority was
    under-optimised. The new span search matches/beats SDR in < 1 s.
  * EGC now gets Gauss-Seidel refinement (SDR adds nothing for EGC: the
    eigenvector + refinement already reaches the same value).
  * SCS tolerances relaxed 1e-7 -> 1e-4 (randomization follows anyway).
"""

import numpy as np
from itertools import combinations

# SDR is only run when L*M <= this value. Above it, eigenvector + refinement
# is used (it matches SDR quality in our tests and is ~100x faster).
# Set to a huge number to force SDR everywhere (very slow for M > 100).
SDR_MAX_LM = 200
SDR_MAX_ITERS = 2000
SDR_EPS = 1e-4


# ======================== CORE UTILITIES ========================

def _build_u_vectors(f_channels, g_bar_channels, N, L, M):
    """u_n = [conj(f) * g_bar_n ; 0]  so that  |h_n|^2 = |u_n^H v_bar|^2."""
    LM = L * M
    f_stacked = np.concatenate(f_channels)
    u_list = []
    for n in range(N):
        g_bar_n = np.concatenate([g_bar_channels[n][l] for l in range(L)])
        u_n = np.zeros(LM + 1, dtype=complex)
        u_n[:LM] = np.conj(f_stacked) * g_bar_n
        u_list.append(u_n)
    return u_list


def _gains(v, U):
    """|u_n^H v_bar|^2 for all rows of U (shape N x (LM+1))."""
    v_bar = np.append(v, 1.0)
    return np.abs(U.conj() @ v_bar) ** 2


def _score(v, U, mode):
    g = _gains(v, U)
    return {'sum': g.sum(), 'min': g.min(), 'max': g.max()}[mode]


def _v_to_phi_list(v, M, L):
    return [v[l * M:(l + 1) * M].copy() for l in range(L)]


def _unit(x):
    return np.exp(1j * np.angle(x))


# ======================== REFINEMENT (NEW) ========================

# All objectives here have the low-rank form  R = B B^H  with
# B = [sqrt(w_1) u_1, ..., sqrt(w_K) u_K]  of shape (LM+1, K), K <= N.
# Working with B instead of the (LM+1)x(LM+1) matrix R gives identical
# results but costs O(LM*K) instead of O(LM^2) — needed for M up to ~1000
# in the RE-number analysis (re_analysis.py).

def _lowrank(Us, w=None):
    """B such that B B^H = sum_k w_k u_k u_k^H  (Us has the u_k as rows)."""
    B = Us.T.copy()
    if w is not None:
        B = B * np.sqrt(w)
    return B


def _gs_refine_sum(B, v, n_sweeps=10):
    """
    Gauss-Seidel ascent for max ||B^H v_bar||^2 = v_bar^H R v_bar, |v_i| = 1.
    Each element update v_i <- exp(j arg(sum_{k!=i} R_ik v_k)) never
    decreases the objective, so this is a monotone ascent.
    (R v)_i is computed as B_i . (B^H v), with B^H v updated incrementally.
    """
    LM = len(v)
    v_bar = np.append(v.copy(), 1.0)
    Bc = B.conj()
    s = Bc.T @ v_bar                                  # B^H v_bar  (K,)
    diag = np.sum(np.abs(B) ** 2, axis=1)             # R_ii
    for _ in range(n_sweeps):
        changed = 0.0
        for i in range(LM):
            c = B[i] @ s - diag[i] * v_bar[i]
            a = np.abs(c)
            if a > 0:
                delta = c / a - v_bar[i]
                ad = np.abs(delta)
                if ad > 0:
                    s += Bc[i] * delta
                    v_bar[i] += delta
                    changed = max(changed, ad)
        if changed < 1e-6:
            break
    return v_bar[:LM]


def _refine_maxmin(U, idx, v, n_outer=15, n_sweeps=3):
    """
    Max-min refinement: repeatedly re-weight SUs (weak SUs get more weight)
    and do Gauss-Seidel ascent on the weighted sum. Keeps the best iterate
    measured by the true min objective.
    """
    Us = U[idx]
    best_v, best = v.copy(), _gains(v, Us).min()
    cur = v.copy()
    for _ in range(n_outer):
        g = _gains(cur, Us)
        w = 1.0 / (g + 1e-300)
        w /= w.sum()
        cur = _gs_refine_sum(_lowrank(Us, w), cur, n_sweeps=n_sweeps)
        val = _gains(cur, Us).min()
        if val > best:
            best, best_v = val, cur.copy()
    return best_v, best


# ======================== EIGENVECTOR METHODS ========================

def _dominant_unit(B, LM):
    """Dominant eigenvector of R = B B^H via the small K x K matrix B^H B."""
    G = B.conj().T @ B
    G = (G + G.conj().T) / 2
    _, Y = np.linalg.eigh(G)
    x = B @ Y[:, -1]
    return _unit(x[:LM])


def _weighted_eig_maxmin(U, idx, LM, n_iter=20):
    Us = U[idx]
    w = np.ones(len(idx)) / len(idx)
    best_v, best = None, -np.inf
    for _ in range(n_iter):
        v = _dominant_unit(_lowrank(Us, w), LM)
        g = _gains(v, Us)
        if g.min() > best:
            best, best_v = g.min(), v
        w = 1.0 / (g + 1e-300)
        w /= w.sum()
    return best_v, best


# ======================== SDR (paper's method) ========================

def _sdr(Q_list, LM, mode, n_rand, rng):
    """
    mode='sum': max Tr(R V)           s.t. V_ii = 1, V >= 0
    mode='min': max t, Tr(Q_n V) >= t s.t. V_ii = 1, V >= 0
    Returns V* or None.
    """
    import cvxpy as cp
    dim = LM + 1
    scale = max(np.max(np.abs(Q)) for Q in Q_list)
    if scale <= 0:
        return None
    Qs = [(Q + Q.conj().T) / (2 * scale) for Q in Q_list]

    V = cp.Variable((dim, dim), hermitian=True)
    cons = [V >> 0, cp.diag(V) == 1]
    if mode == 'sum':
        obj = cp.Maximize(cp.real(cp.trace(sum(Qs) @ V)))
    else:
        t = cp.Variable()
        cons += [cp.real(cp.trace(Q @ V)) >= t for Q in Qs]
        obj = cp.Maximize(t)
    prob = cp.Problem(obj, cons)
    prob.solve(solver=cp.SCS, max_iters=SDR_MAX_ITERS,
               eps_abs=SDR_EPS, eps_rel=SDR_EPS, verbose=False)
    return V.value


def _gaussian_randomization(V_opt, LM, U, idx, mode, n_rand, rng):
    dim = LM + 1
    V_h = (V_opt + V_opt.conj().T) / 2
    vals, vecs = np.linalg.eigh(V_h)
    B = vecs * np.sqrt(np.maximum(vals, 0))
    Us = U[idx]
    best_v, best = None, -np.inf
    for _ in range(n_rand):
        w = (rng.standard_normal(dim) + 1j * rng.standard_normal(dim)) / np.sqrt(2)
        z = B @ w
        z = z * np.exp(-1j * np.angle(z[-1]))
        v = _unit(z[:LM])
        val = _score(v, Us, mode)
        if val > best:
            best, best_v = val, v
    return best_v, best


def _try_sdr(U, idx, LM, mode, n_rand, rng):
    if LM > SDR_MAX_LM:
        return None
    try:
        Q_list = [np.outer(U[n], U[n].conj()) for n in idx]
        V_opt = _sdr(Q_list, LM, mode, n_rand, rng)
        if V_opt is None:
            return None
        return _gaussian_randomization(V_opt, LM, U, idx, mode, n_rand, rng)
    except Exception:
        return None


# ======================== PUBLIC API ========================

def optimize_psm_egc_scsi(f_channels, g_bar_channels, N, L, M,
                          n_randomization=100, rng=None, use_sdr=True):
    rng = rng or np.random.default_rng()
    LM = L * M
    U = np.array(_build_u_vectors(f_channels, g_bar_channels, N, L, M))
    idx = list(range(N))
    B = _lowrank(U)                       # B B^H = sum_n u_n u_n^H

    cands = []
    v_eig = _dominant_unit(B, LM)
    cands.append(v_eig)
    if use_sdr:
        res = _try_sdr(U, idx, LM, 'sum', n_randomization, rng)
        if res is not None:
            cands.append(res[0])
    # Refine every candidate with monotone Gauss-Seidel, keep the best
    best_v, best = None, -np.inf
    for v in cands:
        v = _gs_refine_sum(B, v, n_sweeps=15)
        val = _score(v, U, 'sum')
        if val > best:
            best, best_v = val, v
    return _v_to_phi_list(best_v, M, L)


def optimize_psm_or_scsi(f_channels, g_bar_channels, N, L, M, **_):
    """Closed form (optimal): align every element to the best single SU."""
    LM = L * M
    U = np.array(_build_u_vectors(f_channels, g_bar_channels, N, L, M))
    best_v, best = None, -np.inf
    for n in range(N):
        v = _unit(U[n][:LM])
        val = _gains(v, U)[n]
        if val > best:
            best, best_v = val, v
    return _v_to_phi_list(best_v, M, L)


def _span_search_maxmin(U, idx, LM, rng, n_starts=12, maxfev=600):
    """
    Low-dimensional search for max-min problems.
    The gain of SU n is maximised by v = unit(u_n), so good max-min solutions
    lie (approximately) at v = unit(sum_n c_n u_n) for complex weights c_n.
    With c_1 = 1 fixed there are only 2(K-1) real parameters, which
    Nelder-Mead searches directly on the TRUE min-gain objective.
    In tests this matches or beats SDR (within ~5%, usually better) while
    being ~100x faster, so it also scales to M = 200.
    """
    from scipy.optimize import minimize
    Us = U[idx]
    K = len(idx)
    A = Us[:, :LM]
    An = A / np.linalg.norm(A, axis=1)[:, None]
    if K == 1:
        v = _unit(An[0])
        return v, _gains(v, Us).min()

    def v_from(x):
        c = np.concatenate([[1.0], x[0::2] + 1j * x[1::2]])
        return _unit(c @ An)

    def neg_min(x):
        return -_gains(v_from(x), Us).min()

    best_v, best = None, -np.inf
    for s in range(n_starts):
        x0 = np.zeros(2 * (K - 1)) if s == 0 else rng.standard_normal(2 * (K - 1))
        r = minimize(neg_min, x0, method='Nelder-Mead',
                     options={'maxfev': maxfev, 'xatol': 1e-4, 'fatol': 0.0})
        if -r.fun > best:
            best, best_v = -r.fun, v_from(r.x)
    return best_v, best


def _maxmin_solve(U, idx, LM, n_randomization, rng, use_sdr):
    rng = rng or np.random.default_rng()
    cands = []
    v_w, _ = _weighted_eig_maxmin(U, idx, LM)
    cands.append(v_w)
    v_s, _ = _span_search_maxmin(U, idx, LM, rng)
    cands.append(v_s)
    if use_sdr:
        res = _try_sdr(U, idx, LM, 'min', n_randomization, rng)
        if res is not None:
            cands.append(res[0])
    best_v, best = None, -np.inf
    for v in cands:
        v_r, val = _refine_maxmin(U, idx, v)
        if val > best:
            best, best_v = val, v_r
    return best_v, best


def optimize_psm_and_scsi(f_channels, g_bar_channels, N, L, M,
                          n_randomization=100, rng=None, use_sdr=True):
    rng = rng or np.random.default_rng()
    LM = L * M
    U = np.array(_build_u_vectors(f_channels, g_bar_channels, N, L, M))
    v, _ = _maxmin_solve(U, list(range(N)), LM, n_randomization, rng, use_sdr)
    return _v_to_phi_list(v, M, L)


def optimize_psm_krank_scsi(f_channels, g_bar_channels, N, L, M, K,
                            n_randomization=100, rng=None, use_sdr=False):
    """
    P-S-Krank: solve the max-min sub-problem for every K-subset and keep the
    subset/PSM with the largest min-gain. SDR is off by default here because
    there are C(N,K) sub-problems (6 for N=4, K=2).
    """
    rng = rng or np.random.default_rng()
    LM = L * M
    U = np.array(_build_u_vectors(f_channels, g_bar_channels, N, L, M))
    best_v, best = None, -np.inf
    for subset in combinations(range(N), K):
        v, val = _maxmin_solve(U, list(subset), LM, n_randomization, rng, use_sdr)
        if val > best:
            best, best_v = val, v
    return _v_to_phi_list(best_v, M, L)


def optimize_psm_scsi(scheme, f_channels, g_bar_channels, N, L, M,
                      K=None, n_randomization=100, rng=None, use_sdr=True):
    if scheme == 'EGC':
        return optimize_psm_egc_scsi(f_channels, g_bar_channels, N, L, M,
                                     n_randomization, rng, use_sdr)
    if scheme == 'AND':
        return optimize_psm_and_scsi(f_channels, g_bar_channels, N, L, M,
                                     n_randomization, rng, use_sdr)
    if scheme == 'OR':
        return optimize_psm_or_scsi(f_channels, g_bar_channels, N, L, M)
    if scheme == 'Majority':
        K = K or int(np.ceil(N / 2))
        return optimize_psm_krank_scsi(f_channels, g_bar_channels, N, L, M, K,
                                       n_randomization, rng)
    raise ValueError(f"Unknown scheme: {scheme}")


# ======================== SELF-TEST ========================

if __name__ == '__main__':
    import time
    from system_model import generate_su_positions, get_all_path_losses, N_SU, L_RIS, KAPPA
    from channel_model import generate_all_channels
    from benchmark_psm import random_psm, benchmark_partitioned_psm

    su_pos = generate_su_positions(seed=42)
    eta, beta, zeta = get_all_path_losses(su_pos)
    for (Mh, Mv) in [(4, 4), (8, 8), (10, 10)]:
        M = Mh * Mv
        _, f, _, gb = generate_all_channels(su_pos, eta, beta, zeta, Mh, Mv, KAPPA,
                                            np.random.default_rng(1))
        U = np.array(_build_u_vectors(f, gb, N_SU, L_RIS, M))
        print(f"\n=== M = {M} (LOS gains, what S-CSI optimises) ===")
        print(f"  {'Config':<14}{'sum':>12}{'min':>12}{'max':>12}{'time':>8}")
        rows = [('Random', lambda: random_psm(M, L_RIS, np.random.default_rng(0))),
                ('Partitioned', lambda: benchmark_partitioned_psm(f, gb, M, N_SU, L_RIS))]
        for s in ['EGC', 'OR', 'AND', 'Majority']:
            rows.append((f'Prop. {s}', lambda s=s: optimize_psm_scsi(
                s, f, gb, N_SU, L_RIS, M, K=2, rng=np.random.default_rng(0))))
        for name, fn in rows:
            t0 = time.time(); phi = fn(); dt = time.time() - t0
            g = _gains(np.concatenate(phi), U)
            print(f"  {name:<14}{g.sum():>12.3e}{g.min():>12.3e}{g.max():>12.3e}{dt:>7.1f}s")
