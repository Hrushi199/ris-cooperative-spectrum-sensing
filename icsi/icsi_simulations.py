"""
icsi_simulations.py — Fig. 4 & Fig. 5 with I-CSI + Random only
===============================================================
Generates plots with 2 curves:
  1. Proposed opt. Φ, I-CSI (red)
  2. Random Φ (black dashed)

For Kaggle: upload system_model.py, channel_model.py, energy_detection.py,
proposed_psm_icsi.py as a dataset. Then set DATASET_PATH below.
"""

import sys
import os

# ===== FOR KAGGLE: uncomment and set path =====
# DATASET_PATH = '/kaggle/input/det-project-code'
# sys.path.insert(0, DATASET_PATH)
# RESULTS_DIR = '/kaggle/working/results_icsi'
# ===== FOR LOCAL: uses parent directory =====
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'core'))
RESULTS_DIR = os.path.join(_HERE, '..', 'results', 'icsi')
# ===============================================

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from system_model import (
    generate_su_positions, get_all_path_losses,
    N_SU, L_RIS, SIGMA2_S, SIGMA2_U, T_SAMPLES, P_FA_BAR, KAPPA
)
from channel_model import (
    generate_all_channels, compute_equivalent_channel
)
from energy_detection import (
    compute_threshold_egc, compute_threshold_and, compute_threshold_or,
    compute_threshold_krank, compute_pd_individual, compute_pd_egc,
    compute_pd_and, compute_pd_or, compute_pd_krank, q_function, q_inv
)
from benchmark_psm import random_psm
from proposed_psm_icsi import optimize_psm_icsi

print("All imports successful!")

# ======================== PLOT SETTINGS ========================
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 12
rcParams['figure.dpi'] = 120
rcParams['axes.grid'] = True
rcParams['grid.alpha'] = 0.3

os.makedirs(RESULTS_DIR, exist_ok=True)


# ================================================================
# Helper: compute cooperative Pd
# ================================================================
def compute_cooperative_pd(rho, scheme, gamma_egc, gamma_and, gamma_or, gamma_maj):
    if scheme == 'EGC':
        return compute_pd_egc(rho, gamma_egc)
    elif scheme == 'AND':
        pd_ind = np.array([compute_pd_individual(rho[n], gamma_and) for n in range(N_SU)])
        return compute_pd_and(pd_ind)
    elif scheme == 'OR':
        pd_ind = np.array([compute_pd_individual(rho[n], gamma_or) for n in range(N_SU)])
        return compute_pd_or(pd_ind)
    elif scheme == 'Majority':
        pd_ind = np.array([compute_pd_individual(rho[n], gamma_maj) for n in range(N_SU)])
        return compute_pd_krank(pd_ind, K=2)


# ================================================================
# SIMULATION: Fig. 4 — Pd vs M (Random + I-CSI)
# ================================================================
def simulate_pd_vs_M():
    print("\n" + "=" * 60)
    print("Fig. 4: Cooperative Pd vs M (I-CSI + Random)")
    print("=" * 60)
    
    M_values = [4, 16, 32, 52, 80, 100, 132, 164, 200]
    
    # SDR-based AND solver is expensive but gives paper-matching quality.
    # Use 10 SU × 10 ch = 100 samples per M point.
    # Std error at Pd=0.9: sqrt(0.09/100) = 3% — acceptable.
    n_su_realizations = 10
    n_channel_realizations = 10
    
    schemes = ['EGC', 'AND', 'OR', 'Majority']
    configs = ['random', 'proposed_icsi']
    
    results = {scheme: {config: np.zeros(len(M_values))
                        for config in configs}
               for scheme in schemes}
    
    # Thresholds
    gamma_egc = compute_threshold_egc()
    gamma_and = compute_threshold_and()
    gamma_or = compute_threshold_or()
    gamma_maj, _ = compute_threshold_krank(K=2)
    
    for m_idx, M in enumerate(M_values):
        total_samples = n_su_realizations * n_channel_realizations
        print(f"\nProcessing M = {M} ({m_idx+1}/{len(M_values)}) "
              f"[{total_samples} samples]...")
        
        Mh = int(np.sqrt(M))
        while M % Mh != 0:
            Mh -= 1
        Mv = M // Mh
        
        pd_accum = {scheme: {config: 0.0 for config in configs}
                    for scheme in schemes}
        
        for su_real in range(n_su_realizations):
            su_pos = generate_su_positions(seed=su_real * 100 + 7)
            eta, beta, zeta = get_all_path_losses(su_pos)
            
            for ch_real in range(n_channel_realizations):
                seed = su_real * 10000 + ch_real * 50 + m_idx + 31415
                rng = np.random.default_rng(seed)
                
                # Generate FULL channels
                d, f, g, g_bar = generate_all_channels(
                    su_pos, eta, beta, zeta, Mh, Mv, KAPPA, rng)
                
                # --- Random PSM ---
                Phi_rand = random_psm(M, L_RIS,
                                       rng=np.random.default_rng(seed + 99999))
                _, rho_rand = compute_equivalent_channel(d, f, g, Phi_rand)
                
                # --- I-CSI: optimize PER channel realization ---
                Phi_icsi = {}
                for scheme in schemes:
                    K_val = 2 if scheme == 'Majority' else (N_SU if scheme == 'AND' else 1)
                    Phi_icsi[scheme] = optimize_psm_icsi(
                        scheme, d, f, g, g_bar, N_SU, L_RIS, M,
                        K=K_val, n_randomization=50,
                        rng=np.random.default_rng(seed + 55555))
                
                # Accumulate Pd
                for scheme in schemes:
                    pd_accum[scheme]['random'] += compute_cooperative_pd(
                        rho_rand, scheme, gamma_egc, gamma_and, gamma_or, gamma_maj)
                    
                    _, rho_icsi = compute_equivalent_channel(d, f, g, Phi_icsi[scheme])
                    pd_accum[scheme]['proposed_icsi'] += compute_cooperative_pd(
                        rho_icsi, scheme, gamma_egc, gamma_and, gamma_or, gamma_maj)
        
        # Average
        for scheme in schemes:
            for config in configs:
                results[scheme][config][m_idx] = pd_accum[scheme][config] / total_samples
        
        # Print progress
        for scheme in schemes:
            r = results[scheme]
            print(f"  {scheme}: Random={r['random'][m_idx]:.4f}  "
                  f"I-CSI={r['proposed_icsi'][m_idx]:.4f}")
    
    # ============================= PLOTTING =============================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plot_order = ['EGC', 'OR', 'Majority', 'AND']
    scheme_titles = {
        'EGC': '(a) Data Fusion, EGC',
        'OR': '(b) Decision Fusion, OR Rule ($K = 1$)',
        'Majority': '(c) Decision Fusion, Majority Rule ($K = 2$)',
        'AND': '(d) Decision Fusion, AND Rule ($K = 4$)'
    }
    
    for ax, scheme in zip(axes.flatten(), plot_order):
        ax.plot(M_values, results[scheme]['proposed_icsi'],
                'r-^', markersize=6, linewidth=1.8, markerfacecolor='r',
                label='Proposed opt. $\\Phi_l$, I-CSI')
        ax.plot(M_values, results[scheme]['random'],
                'k--o', markersize=6, linewidth=1.8, markerfacecolor='k',
                label='Random $\\Phi_l$')
        
        ax.set_xlabel('Number of REs per RIS, $M$', fontsize=11)
        ax.set_ylabel('Cooperative Detection Probability, $P_d$', fontsize=11)
        ax.set_title(scheme_titles[scheme], fontsize=12)
        ax.legend(fontsize=10, loc='lower right')
        ax.set_ylim([0, 1.05])
        ax.set_xlim([0, 200])
        ax.set_xticks(np.arange(0, 201, 50))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.tick_params(axis='both', labelsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'fig4_icsi_pd_vs_M.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved: {save_path}")
    
    return results, M_values


# ================================================================
# Fig. 5 — CSS Scheme Comparison (Proposed I-CSI only)
# ================================================================
def simulate_css_comparison(results, M_values):
    print("\n" + "=" * 60)
    print("Fig. 5: CSS Scheme Comparison (Proposed I-CSI)")
    print("=" * 60)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    plot_data = [
        ('EGC', 'b-s', 'EGC (Proposed I-CSI)'),
        ('OR', 'r-o', 'OR Rule, $K=1$ (Proposed I-CSI)'),
        ('Majority', 'g-^', 'Majority Rule, $K=2$ (Proposed I-CSI)'),
        ('AND', 'k-D', 'AND Rule, $K=4$ (Proposed I-CSI)')
    ]
    
    for scheme, style, label in plot_data:
        ax.plot(M_values, results[scheme]['proposed_icsi'],
                style, markersize=6, linewidth=1.5, label=label, markevery=2)
    
    ax.set_xlabel('Number of REs per RIS, $M$', fontsize=13)
    ax.set_ylabel('Cooperative Detection Probability, $P_d$', fontsize=13)
    ax.legend(fontsize=11, loc='lower right')
    ax.set_ylim([0, 1.05])
    ax.set_xlim([0, 200])
    ax.set_xticks(np.arange(0, 201, 50))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'fig5_icsi_css_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {save_path}")


# ================================================================
# PDF Validation (Fig. 2)
# ================================================================
def simulate_pdf_validation():
    print("=" * 60)
    print("Fig. 2: PDF of Test Statistic")
    print("=" * 60)
    
    T = 1000
    rho_values = [0, 0.05, 0.1, 0.2]
    labels = [
        r'$\rho_n = 0\;(\mathcal{H}_0)$',
        r'$\rho_n = 0.05$',
        r'$\rho_n = 0.1$',
        r'$\rho_n = 0.2$'
    ]
    colors = ['#0072BD', '#D95319', '#EDB120', '#7E2F8E']
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    x_range = np.linspace(0.8, 1.4, 1000)
    
    for idx, rho_n in enumerate(rho_values):
        mean_h1 = 1.0 + rho_n * 1.0
        std_h1 = mean_h1 / np.sqrt(T)
        pdf = (1 / (std_h1 * np.sqrt(2 * np.pi))) * \
              np.exp(-0.5 * ((x_range - mean_h1) / std_h1) ** 2)
        ax.plot(x_range, pdf, '-', color=colors[idx], linewidth=2, label=labels[idx])
    
    ax.set_xlabel('Test Statistic, $E_n$', fontsize=14)
    ax.set_ylabel('Probability Density Function', fontsize=14)
    ax.set_xlim(0.8, 1.4)
    ax.set_ylim(0, 14)
    ax.legend(fontsize=12, loc='upper right')
    ax.set_xticks(np.arange(0.8, 1.5, 0.1))
    ax.set_yticks(np.arange(0, 15, 2))
    ax.tick_params(axis='both', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'fig2_pdf_validation.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: fig2_pdf_validation.png")


# ================================================================
# RUN ALL
# ================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("  SIMULATION — I-CSI + Random")
    print("  Paper: Ge et al., IEEE TWC, Vol.23, No.9, Sep 2024")
    print("=" * 60)
    
    simulate_pdf_validation()
    results, M_values = simulate_pd_vs_M()
    simulate_css_comparison(results, M_values)
    
    print("\n" + "=" * 60)
    print("  All simulations complete!")
    print(f"  Results saved in: {RESULTS_DIR}")
    print("=" * 60)
