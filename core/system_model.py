"""
system_model.py
================
Network topology for the RIS-Assisted CSS system.
All coordinates and parameters match Section V of:
  Ge et al., "RIS-Assisted Cooperative Spectrum Sensing for Cognitive Radio Networks,"
  IEEE Trans. Wireless Commun., vol. 23, no. 9, Sep. 2024.

Simulation Setup (from the paper):
  - PU at (0, 0)
  - N=4 SUs randomly in circle centered at (400, 0) with radius 10m
  - L=2 RISs at (380, 25) and (400, 25)
  - Carrier wavelength lambda = 0.12 m
  - PU transmit power = 30 dBm = 1 W
  - AWGN power at SUs = -80 dBm = 1e-11 W
  - Path loss exponents: alpha1=3.5 (PU-SU), alpha2=2 (PU-RIS), alpha3=2 (RIS-SU)
  - Rician factor kappa = 10 for all RIS-SU channels
  - Element spacing d = lambda/2
  - Number of signal samples T = 1000
  - Maximum tolerable false alarm probability P_fa_bar = 0.1
"""

import numpy as np
import matplotlib.pyplot as plt


# ========================== SYSTEM PARAMETERS ==========================
# --- Topology ---
PU_POS = np.array([0.0, 0.0])                          # PU position (meters)
SU_CENTER = np.array([400.0, 0.0])                      # Center of SU deployment area
SU_RADIUS = 10.0                                        # Radius of SU deployment circle (meters)
N_SU = 4                                                # Number of secondary users
L_RIS = 2                                               # Number of RISs
RIS_POSITIONS = np.array([[380.0, 25.0],                # RIS 1 position
                          [400.0, 25.0]])                # RIS 2 position

# --- Carrier & Antenna ---
LAMBDA = 0.12                                           # Carrier wavelength (meters), ~2.5 GHz
D_ELEMENT = LAMBDA / 2                                  # Element spacing (half-wavelength)

# --- Power ---
PU_POWER_DBM = 30.0                                    # PU transmit power in dBm
NOISE_POWER_DBM = -80.0                                 # AWGN power at each SU in dBm
SIGMA2_S = 10 ** (PU_POWER_DBM / 10) / 1000            # PU transmit power in Watts (1 W)
SIGMA2_U = 10 ** (NOISE_POWER_DBM / 10) / 1000         # Noise power in Watts (1e-11 W)

# --- Path Loss Exponents ---
ALPHA_PU_SU = 3.5                                       # PU-SU path loss exponent
ALPHA_PU_RIS = 2.0                                      # PU-RIS path loss exponent
ALPHA_RIS_SU = 2.0                                      # RIS-SU path loss exponent

# --- Rician Factor ---
KAPPA = 10.0                                            # Rician factor for all RIS-SU channels

# --- Sensing Parameters ---
T_SAMPLES = 1000                                        # Number of signal samples per sensing period
P_FA_BAR = 0.1                                          # Maximum tolerable false alarm probability

# --- RIS Array Defaults ---
# M = Mh * Mv (total REs per RIS), varied in simulations
# Default: Mh = Mv for square arrays, adjusted as needed


def generate_su_positions(n_su=N_SU, center=SU_CENTER, radius=SU_RADIUS,
                          seed=None):
    """
    Generate N SU positions uniformly in a circular area.
    
    Parameters
    ----------
    n_su : int
        Number of secondary users.
    center : ndarray
        Center of the deployment circle [x, y] in meters.
    radius : float
        Radius of the deployment circle in meters.
    seed : int or None
        Random seed for reproducibility.
    
    Returns
    -------
    su_positions : ndarray, shape (n_su, 2)
        Positions of SUs in meters.
    """
    rng = np.random.default_rng(seed)
    # Uniform distribution in a circle: r = radius * sqrt(U), theta = 2*pi*V
    r = radius * np.sqrt(rng.uniform(0, 1, n_su))
    theta = rng.uniform(0, 2 * np.pi, n_su)
    su_positions = np.zeros((n_su, 2))
    su_positions[:, 0] = center[0] + r * np.cos(theta)
    su_positions[:, 1] = center[1] + r * np.sin(theta)
    return su_positions


def compute_distance(pos1, pos2):
    """Compute Euclidean distance between two 2D positions."""
    return np.linalg.norm(pos1 - pos2)


def compute_path_loss(distance, alpha, wavelength=LAMBDA):
    """
    Compute path loss using the free-space model from Eq. (Section V):
        L = lambda^2 / ((4*pi)^2 * d^alpha)
    
    Parameters
    ----------
    distance : float
        Distance in meters.
    alpha : float
        Path loss exponent.
    wavelength : float
        Carrier wavelength in meters.
    
    Returns
    -------
    path_loss : float
        Linear path loss (not in dB).
    """
    return (wavelength ** 2) / ((4 * np.pi) ** 2 * distance ** alpha)


def get_all_path_losses(su_positions, pu_pos=PU_POS, ris_positions=RIS_POSITIONS):
    """
    Compute all path losses in the system.
    
    Returns
    -------
    eta : ndarray, shape (N,)
        PU-SU path losses.
    beta : ndarray, shape (L,)
        PU-RIS path losses.
    zeta : ndarray, shape (N, L)
        RIS-SU path losses.
    """
    N = su_positions.shape[0]
    L = ris_positions.shape[0]
    
    # PU-SU path losses (Rayleigh, alpha1=3.5)
    eta = np.zeros(N)
    for n in range(N):
        d1 = compute_distance(pu_pos, su_positions[n])
        eta[n] = compute_path_loss(d1, ALPHA_PU_SU)
    
    # PU-RIS path losses (LOS, alpha2=2)
    beta = np.zeros(L)
    for l in range(L):
        d2 = compute_distance(pu_pos, ris_positions[l])
        beta[l] = compute_path_loss(d2, ALPHA_PU_RIS)
    
    # RIS-SU path losses (Rician, alpha3=2)
    zeta = np.zeros((N, L))
    for n in range(N):
        for l in range(L):
            d3 = compute_distance(su_positions[n], ris_positions[l])
            zeta[n, l] = compute_path_loss(d3, ALPHA_RIS_SU)
    
    return eta, beta, zeta


def plot_topology(su_positions, pu_pos=PU_POS, ris_positions=RIS_POSITIONS,
                  save_path=None):
    """Plot the network topology showing PU, SUs, and RISs."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot PU
    ax.plot(pu_pos[0], pu_pos[1], 'r^', markersize=14, label='PU', zorder=5)
    
    # Plot SUs
    ax.plot(su_positions[:, 0], su_positions[:, 1], 'bs', markersize=10,
            label='SUs', zorder=5)
    for n in range(su_positions.shape[0]):
        ax.annotate(f'SU{n+1}', (su_positions[n, 0] + 2, su_positions[n, 1] + 2),
                    fontsize=9)
    
    # Plot RISs
    ax.plot(ris_positions[:, 0], ris_positions[:, 1], 'g*', markersize=16,
            label='RISs', zorder=5)
    for l in range(ris_positions.shape[0]):
        ax.annotate(f'RIS{l+1}', (ris_positions[l, 0] + 2, ris_positions[l, 1] + 2),
                    fontsize=9)
    
    # Draw SU deployment circle
    circle = plt.Circle(SU_CENTER, SU_RADIUS, fill=False, linestyle='--',
                        color='blue', alpha=0.5, label='SU deployment area')
    ax.add_patch(circle)
    
    # Draw channel links (PU to SUs, PU to RISs, RISs to SUs)
    for n in range(su_positions.shape[0]):
        ax.plot([pu_pos[0], su_positions[n, 0]], [pu_pos[1], su_positions[n, 1]],
                'r--', alpha=0.2, linewidth=0.8)
    for l in range(ris_positions.shape[0]):
        ax.plot([pu_pos[0], ris_positions[l, 0]], [pu_pos[1], ris_positions[l, 1]],
                'g-', alpha=0.3, linewidth=1.0)
        for n in range(su_positions.shape[0]):
            ax.plot([ris_positions[l, 0], su_positions[n, 0]],
                    [ris_positions[l, 1], su_positions[n, 1]],
                    'b:', alpha=0.3, linewidth=0.8)
    
    ax.set_xlabel('x (m)', fontsize=12)
    ax.set_ylabel('y (m)', fontsize=12)
    ax.set_title('RIS-Assisted CSS System Topology', fontsize=14)
    ax.legend(fontsize=10, loc='upper left')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # Generate SU positions with a fixed seed for reproducibility
    su_pos = generate_su_positions(seed=42)
    print("SU Positions (m):")
    for n in range(N_SU):
        print(f"  SU{n+1}: ({su_pos[n, 0]:.2f}, {su_pos[n, 1]:.2f})")
    
    # Compute path losses
    eta, beta, zeta = get_all_path_losses(su_pos)
    print(f"\nPU-SU path losses (eta): {eta}")
    print(f"PU-RIS path losses (beta): {beta}")
    print(f"RIS-SU path losses (zeta):\n{zeta}")
    
    # Print key parameters for verification
    print(f"\n--- Key Parameters ---")
    print(f"sigma2_s (PU power): {SIGMA2_S:.6e} W = {PU_POWER_DBM} dBm")
    print(f"sigma2_u (Noise):    {SIGMA2_U:.6e} W = {NOISE_POWER_DBM} dBm")
    print(f"Lambda:              {LAMBDA} m")
    print(f"T (samples):         {T_SAMPLES}")
    print(f"P_fa_bar:            {P_FA_BAR}")
    
    # Plot topology
    plot_topology(su_pos)
