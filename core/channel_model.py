"""
channel_model.py
=================
Channel generation for the RIS-Assisted CSS system.
Implements all three channel types from Section II-B:
  1. PU-SU direct channels (Rayleigh fading) - Eq. below (3)
  2. PU-RIS channels (LOS) - Eqs. (3), (4), (5)
  3. RIS-SU channels (Rician fading) - Eqs. (6), (7)

All parameters match the paper exactly.
"""

import numpy as np
from system_model import (LAMBDA, D_ELEMENT, KAPPA, PU_POS, RIS_POSITIONS,
                          compute_distance, ALPHA_PU_RIS, ALPHA_RIS_SU,
                          compute_path_loss)


def steering_vector_h(theta, psi, Mh, d=D_ELEMENT, wavelength=LAMBDA):
    """
    Horizontal steering vector - Eq. (4):
        ah(theta, psi) = [1, ..., exp(-j*2*pi*d/lambda * (Mh-1) * sin(theta)*cos(psi))]^T
    
    Parameters
    ----------
    theta : float
        Azimuth angle in radians.
    psi : float
        Elevation angle in radians.
    Mh : int
        Number of elements along horizontal axis.
    d : float
        Element spacing (meters).
    wavelength : float
        Carrier wavelength (meters).
    
    Returns
    -------
    ah : ndarray, shape (Mh,)
        Horizontal steering vector.
    """
    indices = np.arange(Mh)
    phase = -2 * np.pi * d / wavelength * indices * np.sin(theta) * np.cos(psi)
    return np.exp(1j * phase)


def steering_vector_v(theta, psi, Mv, d=D_ELEMENT, wavelength=LAMBDA):
    """
    Vertical steering vector - Eq. (5):
        av(theta, psi) = [1, ..., exp(-j*2*pi*d/lambda * (Mv-1) * cos(theta)*cos(psi))]^T
    
    Parameters
    ----------
    theta : float
        Azimuth angle in radians.
    psi : float
        Elevation angle in radians.
    Mv : int
        Number of elements along vertical axis.
    
    Returns
    -------
    av : ndarray, shape (Mv,)
        Vertical steering vector.
    """
    indices = np.arange(Mv)
    phase = -2 * np.pi * d / wavelength * indices * np.cos(theta) * np.cos(psi)
    return np.exp(1j * phase)


def compute_angles(pos_from, pos_to):
    """
    Compute azimuth and elevation angles between two 2D positions.
    We work in 2D (x, y plane), so elevation angle psi = 0.
    Azimuth theta = atan2(y_diff, x_diff).
    
    Parameters
    ----------
    pos_from : ndarray, shape (2,)
        Source position [x, y].
    pos_to : ndarray, shape (2,)
        Destination position [x, y].
    
    Returns
    -------
    theta : float
        Azimuth angle in radians.
    psi : float
        Elevation angle in radians (0 for 2D case).
    """
    diff = pos_to - pos_from
    theta = np.arctan2(diff[1], diff[0])
    psi = 0.0  # 2D scenario
    return theta, psi


def generate_pu_su_channels(eta, rng=None):
    """
    Generate PU-SU direct channels (Rayleigh fading).
    dn ~ CN(0, eta_n) for each SU n.
    
    Parameters
    ----------
    eta : ndarray, shape (N,)
        PU-SU path losses.
    rng : numpy.random.Generator or None
        Random number generator.
    
    Returns
    -------
    d : ndarray, shape (N,)
        Complex channel gains from PU to each SU.
    """
    if rng is None:
        rng = np.random.default_rng()
    N = len(eta)
    # CN(0, eta_n) = sqrt(eta_n/2) * (randn + j*randn)
    d = np.sqrt(eta / 2) * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    return d


def generate_pu_ris_channels(beta, Mh, Mv, pu_pos=PU_POS,
                              ris_positions=RIS_POSITIONS):
    """
    Generate PU-RIS channels (LOS) - Eq. (3):
        fl = sqrt(beta_l) * ah(theta_AOA, psi_AOA) ⊗ av(theta_AOA, psi_AOA)
    
    Parameters
    ----------
    beta : ndarray, shape (L,)
        PU-RIS path losses.
    Mh : int
        Number of horizontal REs per RIS.
    Mv : int
        Number of vertical REs per RIS.
    pu_pos : ndarray, shape (2,)
        PU position.
    ris_positions : ndarray, shape (L, 2)
        RIS positions.
    
    Returns
    -------
    f_channels : list of ndarray, each shape (M,)
        PU-RIS channel vectors, where M = Mh * Mv.
    """
    L = len(beta)
    M = Mh * Mv
    f_channels = []
    
    for l in range(L):
        # AOA at RIS l from PU
        theta_aoa, psi_aoa = compute_angles(pu_pos, ris_positions[l])
        
        # Steering vectors - Eqs. (4) and (5)
        ah = steering_vector_h(theta_aoa, psi_aoa, Mh)
        av = steering_vector_v(theta_aoa, psi_aoa, Mv)
        
        # Kronecker product: ah ⊗ av - Eq. (3)
        fl = np.sqrt(beta[l]) * np.kron(ah, av)
        f_channels.append(fl)
    
    return f_channels


def generate_ris_su_channels(zeta, Mh, Mv, kappa, su_positions,
                              ris_positions=RIS_POSITIONS, rng=None):
    """
    Generate RIS-SU channels (Rician fading) - Eqs. (6) and (7):
        gn,l = g_bar_n,l + g_tilde_n,l
    where:
        g_bar_n,l = sqrt(kappa*zeta/(1+kappa)) * ah(AOD) ⊗ av(AOD)  [LOS]
        g_tilde_n,l ~ CN(0, zeta/(1+kappa) * I_M)                   [NLOS]
    
    Parameters
    ----------
    zeta : ndarray, shape (N, L)
        RIS-SU path losses.
    Mh : int
        Number of horizontal REs per RIS.
    Mv : int
        Number of vertical REs per RIS.
    kappa : float
        Rician factor (same for all channels).
    su_positions : ndarray, shape (N, 2)
        SU positions.
    ris_positions : ndarray, shape (L, 2)
        RIS positions.
    rng : numpy.random.Generator or None
        Random number generator.
    
    Returns
    -------
    g_channels : list of list of ndarray
        g_channels[n][l] is the channel vector from RIS l to SU n, shape (M,).
    g_bar_channels : list of list of ndarray
        LOS components only (for statistical CSI case in Phase 2).
    """
    if rng is None:
        rng = np.random.default_rng()
    
    N = su_positions.shape[0]
    L = ris_positions.shape[0]
    M = Mh * Mv
    
    g_channels = [[None] * L for _ in range(N)]
    g_bar_channels = [[None] * L for _ in range(N)]
    
    for n in range(N):
        for l in range(L):
            # AOD from RIS l to SU n
            theta_aod, psi_aod = compute_angles(ris_positions[l], su_positions[n])
            
            # LOS component - Eq. (6)
            ah = steering_vector_h(theta_aod, psi_aod, Mh)
            av = steering_vector_v(theta_aod, psi_aod, Mv)
            g_bar = np.sqrt(kappa * zeta[n, l] / (1 + kappa)) * np.kron(ah, av)
            
            # NLOS component - Eq. (7): CN(0, zeta/(1+kappa) * I_M)
            nlos_var = zeta[n, l] / (1 + kappa)
            g_tilde = np.sqrt(nlos_var / 2) * (rng.standard_normal(M) +
                                                1j * rng.standard_normal(M))
            
            g_channels[n][l] = g_bar + g_tilde
            g_bar_channels[n][l] = g_bar
    
    return g_channels, g_bar_channels


def generate_all_channels(su_positions, eta, beta, zeta, Mh, Mv,
                           kappa=KAPPA, rng=None):
    """
    Generate all channels in the system.
    
    Parameters
    ----------
    su_positions : ndarray, shape (N, 2)
    eta : ndarray, shape (N,)
    beta : ndarray, shape (L,)
    zeta : ndarray, shape (N, L)
    Mh, Mv : int
    kappa : float
    rng : numpy.random.Generator or None
    
    Returns
    -------
    d : ndarray, shape (N,) — PU-SU channels
    f : list of ndarray, each shape (M,) — PU-RIS channels
    g : list of list of ndarray — RIS-SU channels (full)
    g_bar : list of list of ndarray — RIS-SU channels (LOS only)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    d = generate_pu_su_channels(eta, rng)
    f = generate_pu_ris_channels(beta, Mh, Mv)
    g, g_bar = generate_ris_su_channels(zeta, Mh, Mv, kappa, su_positions,
                                         rng=rng)
    
    return d, f, g, g_bar


def compute_equivalent_channel(d, f, g, Phi_list):
    """
    Compute the equivalent channel gain for each SU - from Eq. (2):
        hn = dn + sum_l (gH_n,l * Phi_l * fl)
        rho_n = |hn|^2
    
    Parameters
    ----------
    d : ndarray, shape (N,)
        PU-SU direct channels.
    f : list of ndarray, each shape (M,)
        PU-RIS channels.
    g : list of list of ndarray
        g[n][l] is channel from RIS l to SU n, shape (M,).
    Phi_list : list of ndarray, each shape (M,)
        Diagonal elements of PSM for each RIS (phase shifts as complex exponentials).
    
    Returns
    -------
    h : ndarray, shape (N,)
        Equivalent channel for each SU.
    rho : ndarray, shape (N,)
        Equivalent channel gain |hn|^2 for each SU.
    """
    N = len(d)
    L = len(f)
    h = np.copy(d)
    
    for n in range(N):
        for l in range(L):
            # gH_n,l * Phi_l * fl
            # Phi_l is diagonal, so Phi_l * fl = Phi_l .* fl (element-wise)
            h[n] += np.conj(g[n][l]) @ (Phi_list[l] * f[l])
    
    rho = np.abs(h) ** 2
    return h, rho


if __name__ == '__main__':
    from system_model import generate_su_positions, get_all_path_losses, N_SU
    
    # Setup
    su_pos = generate_su_positions(seed=42)
    eta, beta, zeta = get_all_path_losses(su_pos)
    
    # Generate channels with M = 10x10 = 100 REs per RIS
    Mh, Mv = 10, 10
    M = Mh * Mv
    rng = np.random.default_rng(123)
    
    d, f, g, g_bar = generate_all_channels(su_pos, eta, beta, zeta, Mh, Mv, rng=rng)
    
    print(f"--- Channel Dimensions ---")
    print(f"d (PU-SU):    {d.shape}, dtype={d.dtype}")
    print(f"f (PU-RIS):   {len(f)} channels, each shape {f[0].shape}")
    print(f"g (RIS-SU):   {N_SU}x{len(f)} channels, each shape {g[0][0].shape}")
    
    # Test with random PSM
    Phi_list = [np.exp(1j * rng.uniform(0, 2*np.pi, M)) for _ in range(len(f))]
    h, rho = compute_equivalent_channel(d, f, g, Phi_list)
    
    print(f"\n--- Equivalent Channels (Random PSM) ---")
    for n in range(N_SU):
        print(f"  SU{n+1}: |h|^2 = {rho[n]:.6e}")
    
    # Compare with no RIS (direct link only)
    rho_no_ris = np.abs(d) ** 2
    print(f"\n--- Direct Link Only (No RIS) ---")
    for n in range(N_SU):
        print(f"  SU{n+1}: |d|^2 = {rho_no_ris[n]:.6e}")
    
    print(f"\nSNR improvement (dB) with random PSM:")
    for n in range(N_SU):
        if rho_no_ris[n] > 0:
            improvement = 10 * np.log10(rho[n] / rho_no_ris[n])
            print(f"  SU{n+1}: {improvement:.2f} dB")
