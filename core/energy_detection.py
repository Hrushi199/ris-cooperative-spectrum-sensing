"""
energy_detection.py
====================
Energy detection and individual/cooperative detection probability computation.
Implements:
  - Test statistic computation - Eq. (8)
  - Detection threshold calculation via CLT - Eqs. (16), (17)
  - Individual Pd and Pfa - Eqs. (10), (19)
  - Scheme-specific thresholds (EGC, AND, OR, K-rank)

All equations and parameters match the paper exactly.
"""

import numpy as np
from scipy.stats import norm
from scipy.special import comb
from system_model import SIGMA2_S, SIGMA2_U, T_SAMPLES, P_FA_BAR, N_SU


def q_function(x):
    """
    Q-function: Q(x) = integral from x to inf of (1/sqrt(2*pi)) * exp(-t^2/2) dt.
    Equivalent to 1 - Phi(x) where Phi is the standard normal CDF.
    """
    return norm.sf(x)  # survival function = 1 - CDF


def q_inv(p):
    """
    Inverse Q-function: Q^{-1}(p).
    Returns x such that Q(x) = p.
    """
    return norm.isf(p)  # inverse survival function


# ============================== THRESHOLDS ==============================

def compute_threshold_individual(pfa_n, T=T_SAMPLES, sigma2_u=SIGMA2_U):
    """
    Compute individual detection threshold for SU n - Eq. (17):
        gamma_n = sqrt(1/T) * sigma2_u * Q^{-1}(P_fa_n) + sigma2_u
    
    Parameters
    ----------
    pfa_n : float
        Maximum tolerable false alarm probability for SU n.
    T : int
        Number of signal samples.
    sigma2_u : float
        Noise power.
    
    Returns
    -------
    gamma_n : float
        Detection threshold.
    """
    return np.sqrt(1.0 / T) * sigma2_u * q_inv(pfa_n) + sigma2_u


def compute_threshold_egc(pfa_bar=P_FA_BAR, N=N_SU, T=T_SAMPLES,
                           sigma2_u=SIGMA2_U):
    """
    Compute detection threshold for EGC-based CSS - from Section V:
        gamma_EGC = sqrt(1/(N*T)) * sigma2_u * Q^{-1}(P_fa_bar) + sigma2_u
    
    Parameters
    ----------
    pfa_bar : float
        Maximum tolerable cooperative false alarm probability.
    N : int
        Number of SUs.
    T : int
        Number of signal samples.
    sigma2_u : float
        Noise power.
    
    Returns
    -------
    gamma_egc : float
        EGC detection threshold.
    """
    return np.sqrt(1.0 / (N * T)) * sigma2_u * q_inv(pfa_bar) + sigma2_u


def compute_threshold_and(pfa_bar=P_FA_BAR, N=N_SU, T=T_SAMPLES,
                           sigma2_u=SIGMA2_U):
    """
    Compute detection threshold for Logic-AND scheme - from Section V:
        gamma_AND = sqrt(1/T) * sigma2_u * Q^{-1}(P_fa_bar^{1/N}) + sigma2_u
    
    For Logic-AND: cooperative Pfa = prod(Pfa_n) = Pfa_n^N (equal thresholds)
    So individual Pfa_n = P_fa_bar^{1/N}.
    """
    pfa_n = pfa_bar ** (1.0 / N)
    return compute_threshold_individual(pfa_n, T, sigma2_u)


def compute_threshold_or(pfa_bar=P_FA_BAR, N=N_SU, T=T_SAMPLES,
                          sigma2_u=SIGMA2_U):
    """
    Compute detection threshold for Logic-OR scheme - from Section V:
        gamma_OR = sqrt(1/T) * sigma2_u * Q^{-1}(1 - (1-P_fa_bar)^{1/N}) + sigma2_u
    
    For Logic-OR: cooperative Pfa = 1 - prod(1-Pfa_n) = 1 - (1-Pfa_n)^N
    So individual Pfa_n = 1 - (1-P_fa_bar)^{1/N}.
    """
    pfa_n = 1.0 - (1.0 - pfa_bar) ** (1.0 / N)
    return compute_threshold_individual(pfa_n, T, sigma2_u)


def compute_threshold_krank(K, pfa_bar=P_FA_BAR, N=N_SU, T=T_SAMPLES,
                             sigma2_u=SIGMA2_U):
    """
    Compute detection threshold for K-rank scheme - from Section V.
    Need to find individual Pfa_n satisfying:
        sum_{i=K}^{N} C(N,i) * Pfa_n^i * (1-Pfa_n)^{N-i} = P_fa_bar
    
    This is solved numerically via bisection.
    
    Parameters
    ----------
    K : int
        K-rank parameter (number of SUs required to declare PU present).
    """
    from scipy.optimize import brentq
    
    def equation(pfa_n):
        total = 0.0
        for i in range(K, N + 1):
            total += comb(N, i, exact=True) * (pfa_n ** i) * ((1 - pfa_n) ** (N - i))
        return total - pfa_bar
    
    # Solve for pfa_n in (0, 1)
    pfa_n = brentq(equation, 1e-15, 1.0 - 1e-15)
    return compute_threshold_individual(pfa_n, T, sigma2_u), pfa_n


# ========================= DETECTION PROBABILITIES =========================

def compute_pd_individual(rho_n, gamma_n, T=T_SAMPLES, sigma2_s=SIGMA2_S,
                           sigma2_u=SIGMA2_U):
    """
    Compute individual detection probability for SU n - Eq. (19):
        En|H1 ~ N(sigma2_u + rho_n*sigma2_s, (sigma2_u + rho_n*sigma2_s)^2 / T)
        Pd,n = Q((gamma_n - mean) / std)
    
    Parameters
    ----------
    rho_n : float or ndarray
        Equivalent channel gain |hn|^2.
    gamma_n : float
        Detection threshold.
    T : int
        Number of signal samples.
    sigma2_s : float
        PU transmit power.
    sigma2_u : float
        Noise power.
    
    Returns
    -------
    pd_n : float or ndarray
        Detection probability.
    """
    mean_h1 = sigma2_u + rho_n * sigma2_s
    std_h1 = mean_h1 / np.sqrt(T)
    pd_n = q_function((gamma_n - mean_h1) / std_h1)
    return pd_n


def compute_pfa_individual(gamma_n, T=T_SAMPLES, sigma2_u=SIGMA2_U):
    """
    Compute individual false alarm probability for SU n - Eq. (16):
        En|H0 ~ N(sigma2_u, sigma2_u^2 / T)
        Pfa,n = Q((gamma_n - sigma2_u) / (sigma2_u / sqrt(T)))
    """
    mean_h0 = sigma2_u
    std_h0 = sigma2_u / np.sqrt(T)
    return q_function((gamma_n - mean_h0) / std_h0)


# =================== COOPERATIVE DETECTION PROBABILITIES ===================

def compute_pd_egc(rho_array, gamma_egc, N=N_SU, T=T_SAMPLES,
                    sigma2_s=SIGMA2_S, sigma2_u=SIGMA2_U):
    """
    Compute cooperative detection probability for EGC scheme - Proposition 1.
    
    E_EGC|H1 ~ N(sigma2_u + rho/N * sigma2_s,
                  (sigma_u^4/(NT) + rho^2*sigma_s^4 + 2*rho*sigma_s^2*sigma_u^2) / (N^2*T))
    
    NOTE: In the paper, sigma_u^4 = (sigma^2_u)^2. Since our variable sigma2_u
    represents sigma^2_u, we compute sigma2_u**2 to get sigma_u^4.
    
    where rho = ||h||^2 = sum(rho_n).
    
    Parameters
    ----------
    rho_array : ndarray, shape (N,)
        Individual channel gains |hn|^2.
    gamma_egc : float
        EGC detection threshold.
    
    Returns
    -------
    pd_egc : float
        Cooperative detection probability (EGC).
    """
    rho = np.sum(rho_array)
    
    # Mean from Proposition 1
    mean_h1 = sigma2_u + (rho / N) * sigma2_s
    
    # Variance from Proposition 1
    # sigma_u^4 = (sigma2_u)^2, sigma_s^4 = (sigma2_s)^2
    # sigma_s^2 * sigma_u^2 = sigma2_s * sigma2_u
    var_h1 = (sigma2_u ** 2 / (N * T)
              + rho ** 2 * sigma2_s ** 2 / (N ** 2 * T)
              + 2 * rho * sigma2_s * sigma2_u / (N ** 2 * T))
    
    std_h1 = np.sqrt(var_h1)
    pd_egc = q_function((gamma_egc - mean_h1) / std_h1)
    return pd_egc


def compute_pd_and(pd_array):
    """
    Cooperative detection probability for Logic-AND - Eq. (12), Table I:
        Pd_AND = prod(Pd,n)
    """
    return np.prod(pd_array)


def compute_pd_or(pd_array):
    """
    Cooperative detection probability for Logic-OR - Eq. (13), Table I:
        Pd_OR = 1 - prod(1 - Pd,n)
    """
    return 1.0 - np.prod(1.0 - pd_array)


def compute_pd_krank(pd_array, K):
    """
    Cooperative detection probability for K-rank - Eq. (14), Table I:
        Pd_Krank = sum_{i=K}^{N} C(N,i) * prod_combinations
    
    For the case where all SUs have equal Pd (same threshold):
        Pd_Krank = sum_{i=K}^{N} C(N,i) * Pd^i * (1-Pd)^{N-i}
    
    For general (unequal) Pd's, we use the exact formula.
    Since in the paper all SUs use the same threshold but have different
    channel conditions, Pd's are generally unequal. We compute the exact
    cooperative Pd.
    """
    N = len(pd_array)
    
    # For unequal Pd's, we need to enumerate all subsets of size >= K
    # and compute the probability that at least K out of N detect.
    # This is done using the inclusion-exclusion or dynamic programming.
    
    # DP approach: dp[j] = probability that exactly j SUs detect
    # Start with dp[0] = 1 (no SUs considered yet)
    dp = np.zeros(N + 1)
    dp[0] = 1.0
    
    for n in range(N):
        # Process SU n (from right to left to avoid overwriting)
        new_dp = np.zeros(N + 1)
        for j in range(n + 1, -1, -1):
            if j > 0:
                new_dp[j] = dp[j] * (1 - pd_array[n]) + dp[j - 1] * pd_array[n]
            else:
                new_dp[j] = dp[j] * (1 - pd_array[n])
        dp = new_dp
    
    # Pd_Krank = P(sum(Dn) >= K) = sum_{i=K}^{N} dp[i]
    return np.sum(dp[K:])


def compute_pfa_and(pfa_n, N=N_SU):
    """Cooperative Pfa for Logic-AND: prod(Pfa_n)."""
    return pfa_n ** N


def compute_pfa_or(pfa_n, N=N_SU):
    """Cooperative Pfa for Logic-OR: 1 - (1-Pfa_n)^N."""
    return 1.0 - (1.0 - pfa_n) ** N


def compute_pfa_krank(pfa_n, K, N=N_SU):
    """Cooperative Pfa for K-rank."""
    total = 0.0
    for i in range(K, N + 1):
        total += comb(N, i, exact=True) * (pfa_n ** i) * ((1 - pfa_n) ** (N - i))
    return total


# ========================= TEST STATISTIC ================================

def generate_test_statistic_h0(T=T_SAMPLES, sigma2_u=SIGMA2_U, rng=None):
    """
    Generate test statistic under H0 for one SU.
    En|H0 = (1/T) * sum_{t=1}^{T} |un(t)|^2
    """
    if rng is None:
        rng = np.random.default_rng()
    # un(t) ~ CN(0, sigma2_u)
    noise = np.sqrt(sigma2_u / 2) * (rng.standard_normal(T) +
                                      1j * rng.standard_normal(T))
    return np.mean(np.abs(noise) ** 2)


def generate_test_statistic_h1(hn, T=T_SAMPLES, sigma2_s=SIGMA2_S,
                                sigma2_u=SIGMA2_U, rng=None):
    """
    Generate test statistic under H1 for one SU - Eq. (2).
    xn(t) = hn * s(t) + un(t)
    En|H1 = (1/T) * sum_{t=1}^{T} |xn(t)|^2
    
    Parameters
    ----------
    hn : complex
        Equivalent channel for this SU.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # s(t) ~ CN(0, sigma2_s)
    signal = np.sqrt(sigma2_s / 2) * (rng.standard_normal(T) +
                                       1j * rng.standard_normal(T))
    # un(t) ~ CN(0, sigma2_u)
    noise = np.sqrt(sigma2_u / 2) * (rng.standard_normal(T) +
                                      1j * rng.standard_normal(T))
    # Received signal
    x = hn * signal + noise
    return np.mean(np.abs(x) ** 2)


if __name__ == '__main__':
    print("=== Detection Threshold Verification ===")
    print(f"Parameters: T={T_SAMPLES}, sigma2_s={SIGMA2_S:.2e}, "
          f"sigma2_u={SIGMA2_U:.2e}, P_fa_bar={P_FA_BAR}")
    
    # Compute thresholds for all schemes
    gamma_egc = compute_threshold_egc()
    gamma_and = compute_threshold_and()
    gamma_or = compute_threshold_or()
    gamma_kr2, pfa_kr2 = compute_threshold_krank(K=2)  # Majority rule with N=4
    gamma_maj, pfa_maj = compute_threshold_krank(K=2)   # K=ceil(N/2)=2 for N=4
    
    print(f"\nThresholds:")
    print(f"  EGC:      gamma = {gamma_egc:.6e}")
    print(f"  AND:      gamma = {gamma_and:.6e}")
    print(f"  OR:       gamma = {gamma_or:.6e}")
    print(f"  K-rank(2):gamma = {gamma_kr2:.6e}, individual Pfa = {pfa_kr2:.6f}")
    
    # Verify Pfa
    print(f"\nFalse alarm probability verification:")
    pfa_egc = compute_pfa_individual(gamma_egc, T=T_SAMPLES * N_SU)
    pfa_and_ind = compute_pfa_individual(gamma_and)
    pfa_or_ind = compute_pfa_individual(gamma_or)
    pfa_kr_ind = compute_pfa_individual(gamma_kr2)
    
    print(f"  EGC:   individual Pfa -> cooperative Pfa = {pfa_egc:.6f} (target: {P_FA_BAR})")
    print(f"  AND:   individual Pfa = {pfa_and_ind:.6f} -> cooperative = {compute_pfa_and(pfa_and_ind):.6f}")
    print(f"  OR:    individual Pfa = {pfa_or_ind:.6f} -> cooperative = {compute_pfa_or(pfa_or_ind):.6f}")
    print(f"  K-rank(2): individual Pfa = {pfa_kr_ind:.6f} -> cooperative = {compute_pfa_krank(pfa_kr_ind, 2):.6f}")
    
    # Test detection probability computation
    print(f"\n=== Detection Probability Test ===")
    test_rho = np.array([1e-10, 1e-9, 1e-8, 1e-7])
    for rho_val in test_rho:
        pd = compute_pd_individual(rho_val, gamma_or)
        print(f"  rho = {rho_val:.1e}: Pd = {pd:.6f}")
