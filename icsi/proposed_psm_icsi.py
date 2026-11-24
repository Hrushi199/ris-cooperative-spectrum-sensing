"""
proposed_psm_icsi.py — I-CSI PSM Optimization
===================================================================
Key design:
  - AND: SDR (SDP relaxation + Gaussian randomization) — matches paper
  - EGC: Eigenvector + Jacobi coord ascent + perturbations
  - OR:  Closed-form phase alignment (optimal)
  - Majority: Fast weighted eigenvector per subset
"""

import numpy as np
from itertools import combinations


# ======================== CORE ========================

def _build_u_vectors(d, f_channels, g_channels, N, L, M):
    LM = L * M
    f_stacked = np.concatenate(f_channels)
    u_list = []
    for n in range(N):
        g_n = np.concatenate([g_channels[n][l] for l in range(L)])
        u_n = np.zeros(LM + 1, dtype=complex)
        u_n[:LM] = np.conj(f_stacked) * g_n
        u_n[LM] = np.conj(d[n])
        u_list.append(u_n)
    return u_list


def _v_to_phi_list(v, M, L):
    return [v[l * M:(l + 1) * M] for l in range(L)]


def _evaluate_min(v, u_list, indices):
    """Evaluate min_n |u_n^H v_bar|^2."""
    v_bar = np.append(v, 1.0)
    return min(np.abs(np.conj(u_list[n]) @ v_bar) ** 2 for n in indices)


# ======================== COORD ASCENT ========================

def _coord_jacobi(R, LM, v, n_iter=6):
    """Jacobi vectorized coord ascent for sum objective."""
    v_bar = np.append(v.copy(), 1.0)
    diag = np.diag(R)[:LM].copy()
    for _ in range(n_iter):
        c = (R[:LM, :] @ v_bar) - diag * v_bar[:LM]
        v_bar[:LM] = np.exp(1j * np.angle(c))
    return v_bar[:LM].copy()


def _gs_refine_maxmin(Q_list, indices, LM, v, n_iter=8):
    """
    Gauss-Seidel coord ascent for max-min objective.
    Each iteration: find weakest SU, optimize element by element for it.
    """
    v_bar = np.append(v.copy(), 1.0)
    for _ in range(n_iter):
        objs = {n: np.real(np.conj(v_bar) @ Q_list[n] @ v_bar) for n in indices}
        weak = min(objs, key=objs.get)
        Q_w = Q_list[weak]
        for i in range(LM):
            c_i = Q_w[i, :] @ v_bar - Q_w[i, i] * v_bar[i]
            if np.abs(c_i) > 1e-30:
                v_bar[i] = np.exp(1j * np.angle(c_i))
    return v_bar[:LM].copy()


# ======================== SOLVERS ========================

def _solve_egc(u_list, N, LM, rng=None):
    """EGC: max sum |u_n^H v_bar|^2."""
    R = sum(np.outer(u_list[n], np.conj(u_list[n])) for n in range(N))
    R = (R + R.conj().T) / 2
    _, vecs = np.linalg.eigh(R)
    v = np.exp(1j * np.angle(vecs[:, -1][:LM]))
    v = _coord_jacobi(R, LM, v, n_iter=8)
    
    # Light perturbations
    if rng is not None:
        best_v, best_obj = v.copy(), sum(
            np.abs(np.conj(u_list[n]) @ np.append(v, 1.0))**2 for n in range(N))
        for sigma in [0.3, 0.1]:
            for _ in range(15):
                v_c = np.exp(1j * (np.angle(v) + rng.standard_normal(LM) * sigma))
                obj = sum(np.abs(np.conj(u_list[n]) @ np.append(v_c, 1.0))**2
                          for n in range(N))
                if obj > best_obj:
                    best_obj = obj; best_v = v_c
        v = best_v
    return v


def _solve_or(u_list, N, LM):
    """OR: max max_n |u_n^H v_bar|^2. Closed-form."""
    best_v, best_obj = None, -1
    for n in range(N):
        v = np.exp(1j * np.angle(u_list[n][:LM]))
        v_bar = np.append(v, 1.0)
        obj = np.abs(np.conj(u_list[n]) @ v_bar) ** 2
        if obj > best_obj:
            best_obj = obj; best_v = v
    return best_v


def _solve_and_sdr(u_list, indices, LM, rng=None, max_scs_iters=150):
    """
    AND: max min_n |u_n^H v_bar|^2.
    SDR (SDP relaxation) + Gaussian randomization + eigenvector backup.
    This is the paper's method — gives provably near-optimal solutions.
    """
    import cvxpy as cp
    if rng is None:
        rng = np.random.default_rng()
    
    N_sub = len(indices)
    Q_list = {n: np.outer(u_list[n], np.conj(u_list[n])) for n in indices}
    dim = LM + 1
    best_v, best_obj = None, -1
    
    # === Method 1: SDR + Gaussian randomization ===
    try:
        # Scale for numerical stability
        scale = max(np.max(np.abs(Q_list[n])) for n in indices)
        if scale < 1e-30:
            raise ValueError("Zero matrices")
        Q_scaled = {n: (Q_list[n] + Q_list[n].conj().T) / (2 * scale) for n in indices}
        
        # SDP: max t s.t. Tr(Q_n V) >= t, V_ii = 1, V >= 0
        V = cp.Variable((dim, dim), hermitian=True)
        t = cp.Variable()
        constraints = [V >> 0]
        for i in range(dim):
            constraints.append(V[i, i] == 1)
        for n in indices:
            constraints.append(cp.real(cp.trace(Q_scaled[n] @ V)) >= t)
        
        prob = cp.Problem(cp.Maximize(t), constraints)
        prob.solve(solver=cp.SCS, verbose=False,
                   max_iters=max_scs_iters, eps_abs=1e-3, eps_rel=1e-3)
        
        V_opt = V.value
        if V_opt is not None:
            # Gaussian randomization
            V_h = (V_opt + V_opt.conj().T) / 2
            eigvals, eigvecs = np.linalg.eigh(V_h)
            eigvals = np.maximum(eigvals, 0)
            sqrt_L = np.diag(np.sqrt(eigvals))
            
            for _ in range(200):
                w = (rng.standard_normal(dim) + 1j * rng.standard_normal(dim)) / np.sqrt(2)
                z = eigvecs @ sqrt_L @ w
                z_bar = z * np.exp(-1j * np.angle(z[-1]))
                v = np.exp(1j * np.angle(z_bar[:LM]))
                obj = _evaluate_min(v, u_list, indices)
                if obj > best_obj:
                    best_obj = obj; best_v = v
    except Exception:
        pass  # Fall through to eigenvector backup
    
    # === Method 2: Eigenvector backup (fast, always works) ===
    # Weighted eigenvector with multiple inits
    weight_inits = [np.ones(N_sub) / N_sub]
    for k in range(N_sub):
        w = np.ones(N_sub) * 0.05; w[k] = 0.85; w /= w.sum()
        weight_inits.append(w)
    
    for w_init in weight_inits:
        w = w_init.copy()
        for _ in range(10):
            R_w = sum(w[k] * Q_list[indices[k]] for k in range(N_sub))
            R_w = (R_w + R_w.conj().T) / 2
            _, vecs = np.linalg.eigh(R_w)
            v = np.exp(1j * np.angle(vecs[:, -1][:LM]))
            obj = _evaluate_min(v, u_list, indices)
            if obj > best_obj:
                best_obj = obj; best_v = v
            
            v_bar = np.append(v, 1.0)
            objs = np.array([np.abs(np.conj(u_list[indices[k]]) @ v_bar)**2
                             for k in range(N_sub)])
            w = 1.0 / (objs + 1e-30); w /= w.sum()
    
    # === GS refinement on best ===
    if best_v is not None:
        v = _gs_refine_maxmin(Q_list, indices, LM, best_v, n_iter=10)
        obj = _evaluate_min(v, u_list, indices)
        if obj > best_obj:
            best_obj = obj; best_v = v
    
    # === Perturbations on best ===
    if best_v is not None:
        for sigma in [0.3, 0.15, 0.05]:
            for _ in range(30):
                v_c = np.exp(1j * (np.angle(best_v) + rng.standard_normal(LM) * sigma))
                obj = _evaluate_min(v_c, u_list, indices)
                if obj > best_obj:
                    best_obj = obj; best_v = v_c
    
    return best_v


def _solve_and_fast(u_list, indices, LM, n_iter=6, rng=None):
    """
    Fast AND solver for Majority subsets.
    Single-start weighted eigenvector + light GS.
    """
    N_sub = len(indices)
    Q_list = {n: np.outer(u_list[n], np.conj(u_list[n])) for n in indices}
    
    w = np.ones(N_sub) / N_sub
    best_v, best_obj = None, -1
    
    for _ in range(n_iter):
        R_w = sum(w[k] * Q_list[indices[k]] for k in range(N_sub))
        R_w = (R_w + R_w.conj().T) / 2
        _, vecs = np.linalg.eigh(R_w)
        v = np.exp(1j * np.angle(vecs[:, -1][:LM]))
        
        v_bar = np.append(v, 1.0)
        objs = np.array([np.abs(np.conj(u_list[indices[k]]) @ v_bar) ** 2
                         for k in range(N_sub)])
        obj = objs.min()
        if obj > best_obj:
            best_obj = obj; best_v = v
        
        w = 1.0 / (objs + 1e-30)
        w /= w.sum()
    
    # Light GS
    if best_v is not None:
        v = _gs_refine_maxmin(Q_list, indices, LM, best_v, n_iter=3)
        obj = _evaluate_min(v, u_list, indices)
        if obj > best_obj:
            best_v = v
    
    return best_v


# ======================== PUBLIC API ========================

def optimize_psm_icsi(scheme, d, f_channels, g_channels, g_bar_channels,
                       N, L, M, K=None, n_randomization=50, rng=None):
    LM = L * M
    u_list = _build_u_vectors(d, f_channels, g_channels, N, L, M)
    
    if rng is None:
        rng = np.random.default_rng()
    
    if scheme == 'EGC':
        v = _solve_egc(u_list, N, LM, rng=rng)
    
    elif scheme == 'OR':
        v = _solve_or(u_list, N, LM)
    
    elif scheme == 'AND':
        v = _solve_and_sdr(u_list, list(range(N)), LM, rng=rng)
    
    elif scheme == 'Majority':
        if K is None:
            K = int(np.ceil(N / 2))
        best_v, best_obj = None, -1
        for subset in combinations(range(N), K):
            v_sub = _solve_and_fast(u_list, list(subset), LM,
                                     n_iter=6, rng=rng)
            if v_sub is not None:
                obj = _evaluate_min(v_sub, u_list, list(subset))
                if obj > best_obj:
                    best_obj = obj; best_v = v_sub
        v = best_v if best_v is not None else np.exp(1j * rng.uniform(0, 2*np.pi, LM))
    
    else:
        raise ValueError(f"Unknown scheme: {scheme}")
    
    return _v_to_phi_list(v, M, L)
