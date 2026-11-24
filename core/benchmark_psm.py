"""
benchmark_psm.py
=================
PSM configuration methods for baseline and benchmark.
Implements:
  1. Random PSM: phi ~ Uniform[0, 2*pi) for all elements
  2. Benchmark (Partitioned Sub-surfaces) - Eq. (47):
     Each RIS is partitioned into N sub-surfaces, one per SU.
     Phase shifts are configured to align the cascaded channel.

All equations match the paper exactly.
"""

import numpy as np


def random_psm(M, L, rng=None):
    """
    Generate random PSM configuration.
    Phase shifts are drawn uniformly from [0, 2*pi).
    
    Parameters
    ----------
    M : int
        Number of REs per RIS.
    L : int
        Number of RISs.
    rng : numpy.random.Generator or None
    
    Returns
    -------
    Phi_list : list of ndarray, each shape (M,)
        Phase shift vectors as complex exponentials.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    Phi_list = []
    for l in range(L):
        phases = rng.uniform(0, 2 * np.pi, M)
        Phi_list.append(np.exp(1j * phases))
    
    return Phi_list


def no_ris_psm(M, L):
    """
    Configuration simulating no RIS (set all phase shifts to cancel out).
    We achieve this by returning zero-vector PSMs.
    
    Actually, "no RIS" means we just don't include the RIS path.
    We handle this in simulation by not adding the reflected paths.
    This function returns None to signal no RIS usage.
    """
    return None


def benchmark_partitioned_psm(f_channels, g_bar_channels, M, N, L):
    """
    Benchmark PSM: Partitioned sub-surfaces method - Eq. (47).
    
    Each RIS (with M elements) is equally partitioned into N sub-surfaces.
    The n-th sub-surface (elements (n-1)*M/N+1 to n*M/N) of the l-th RIS
    is configured to align the cascaded channel from PU to SU n:
    
        phi_{l,(n-1)*M/N + i} = arg(g_bar_{n,(l-1)*M+(n-1)*M/N+i}) 
                                - arg(f_{l,(n-1)*M/N+i})
    
    for i = 1, ..., M/N.
    
    Parameters
    ----------
    f_channels : list of ndarray, each shape (M,)
        PU-RIS channel vectors (LOS).
    g_bar_channels : list of list of ndarray
        g_bar_channels[n][l] is the LOS component of channel from RIS l to SU n.
    M : int
        Number of REs per RIS.
    N : int
        Number of SUs.
    L : int
        Number of RISs.
    
    Returns
    -------
    Phi_list : list of ndarray, each shape (M,)
        Benchmark PSM as complex exponentials.
    """
    assert M % N == 0, f"M={M} must be divisible by N={N} for partitioning."
    M_sub = M // N  # Elements per sub-surface
    
    Phi_list = []
    for l in range(L):
        phases = np.zeros(M, dtype=float)
        for n in range(N):
            start_idx = n * M_sub
            end_idx = (n + 1) * M_sub
            
            # Eq. (47): align phase to cascaded channel
            # phi = arg(g_bar_n,l[i]) - arg(f_l[i])
            g_bar_sub = g_bar_channels[n][l][start_idx:end_idx]
            f_sub = f_channels[l][start_idx:end_idx]
            
            phases[start_idx:end_idx] = np.angle(g_bar_sub) - np.angle(f_sub)
        
        Phi_list.append(np.exp(1j * phases))
    
    return Phi_list


if __name__ == '__main__':
    from system_model import generate_su_positions, get_all_path_losses, N_SU, L_RIS
    from channel_model import generate_all_channels, compute_equivalent_channel
    
    su_pos = generate_su_positions(seed=42)
    eta, beta, zeta = get_all_path_losses(su_pos)
    
    # M must be divisible by N=4
    Mh, Mv = 10, 10  # M = 100
    M = Mh * Mv
    rng = np.random.default_rng(123)
    
    d, f, g, g_bar = generate_all_channels(su_pos, eta, beta, zeta, Mh, Mv, rng=rng)
    
    # --- Random PSM ---
    Phi_rand = random_psm(M, L_RIS, rng=np.random.default_rng(999))
    _, rho_rand = compute_equivalent_channel(d, f, g, Phi_rand)
    
    # --- Benchmark PSM ---
    Phi_bench = benchmark_partitioned_psm(f, g_bar, M, N_SU, L_RIS)
    _, rho_bench = compute_equivalent_channel(d, f, g, Phi_bench)
    
    # --- No RIS ---
    rho_no_ris = np.abs(d) ** 2
    
    print("=== Channel Gain Comparison ===")
    print(f"{'SU':<6} {'No RIS':>12} {'Random PSM':>14} {'Benchmark':>14}")
    for n in range(N_SU):
        print(f"SU{n+1:<3}  {rho_no_ris[n]:>12.4e}  {rho_rand[n]:>14.4e}  {rho_bench[n]:>14.4e}")
    
    print(f"\nSum rho (||h||^2):")
    print(f"  No RIS:    {np.sum(rho_no_ris):.6e}")
    print(f"  Random:    {np.sum(rho_rand):.6e}")
    print(f"  Benchmark: {np.sum(rho_bench):.6e}")
