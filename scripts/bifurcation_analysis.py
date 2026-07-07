"""
Bifurcation analysis for the spatially-homogeneous FitzHugh-Nagumo ODE.
Classifies the (a, b) plane by regime and draws the training box on top.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fhn_fno.config import DataConfig


def fixed_points(a: float, b: float):
    """Real roots of b u^3 - 3 (b - 1) u + 3 a = 0 as (u*, v*)."""
    coeffs = [b, 0.0, -3.0 * (b - 1.0), 3.0 * a]
    roots = np.roots(coeffs)
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-8:
            u_star = float(r.real)
            v_star = (u_star + a) / b
            real_roots.append((u_star, v_star))
    return real_roots


def classify(a: float, b: float, tau: float):
    """Regime label and fixed-point count from the Jacobian at each fixed point."""
    fps = fixed_points(a, b)
    n = len(fps)
    if n == 1:
        u_star, _ = fps[0]
        trace = 1.0 - u_star ** 2 - b / tau
        det = (1.0 - b * (1.0 - u_star ** 2)) / tau
        if det <= 0:
            return 'degenerate', n
        if trace < 0:
            return 'excitable', n
        return 'oscillatory', n
    if n == 3:
        return 'bistable', n
    return 'degenerate', n


def main():
    cfg = DataConfig()
    a_lo, a_hi = cfg.a_range
    b_lo, b_hi = cfg.b_range
    tau_lo, tau_hi = cfg.tau_range
    tau_med = 0.5 * (tau_lo + tau_hi)

    a_plot = np.linspace(-0.8, 0.8, 401)
    b_plot = np.linspace(0.02, 1.5, 401)
    AA, BB = np.meshgrid(a_plot, b_plot, indexing='ij')

    regime_map = np.zeros_like(AA, dtype=np.int8)
    label_to_id = {'excitable': 0, 'oscillatory': 1, 'bistable': 2, 'degenerate': 3}
    for i in range(AA.shape[0]):
        for j in range(AA.shape[1]):
            label, _ = classify(AA[i, j], BB[i, j], tau_med)
            regime_map[i, j] = label_to_id[label]

    # Sweep the training box on a finer grid and report regime fractions.
    a_tr = np.linspace(a_lo, a_hi, 60)
    b_tr = np.linspace(b_lo, b_hi, 60)
    tau_tr = np.linspace(tau_lo, tau_hi, 20)
    counts = {k: 0 for k in label_to_id}
    for tau in tau_tr:
        for aa in a_tr:
            for bb in b_tr:
                lbl, _ = classify(aa, bb, tau)
                counts[lbl] += 1
    total = sum(counts.values())
    print('Sweep over training parameter cube (a x b x tau):')
    for k, v in counts.items():
        print(f"  {k:12s}: {v}/{total} ({100*v/total:.1f}%)")
    print(f"Median tau used for 2D map: {tau_med:.2f}")

    cmap = plt.matplotlib.colors.ListedColormap(['#7DCEA0', '#F5B041', '#AF7AC5', '#BFC9CA'])
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    extent = (a_plot.min(), a_plot.max(), b_plot.min(), b_plot.max())
    ax.imshow(regime_map.T, origin='lower', extent=extent, aspect='auto',
              cmap=cmap, vmin=-0.5, vmax=3.5, interpolation='nearest')

    rect = mpatches.Rectangle((a_lo, b_lo), a_hi - a_lo, b_hi - b_lo,
                              linewidth=2.0, edgecolor='black',
                              facecolor='none', linestyle='-',
                              label='Training parameter range')
    ax.add_patch(rect)

    handles = [
        mpatches.Patch(color='#7DCEA0', label='Excitable (stable rest)'),
        mpatches.Patch(color='#F5B041', label='Oscillatory (limit cycle)'),
        mpatches.Patch(color='#AF7AC5', label='Bistable'),
        mpatches.Patch(facecolor='none', edgecolor='black', label='Training range'),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.95)

    ax.set_xlabel(r'$a$', fontsize=11)
    ax.set_ylabel(r'$b$', fontsize=11)
    ax.set_title(
        rf'FHN dynamical regimes at $\tau = {tau_med:.1f}$',
        fontsize=11,
    )
    ax.grid(False)

    out_dirs = [
        Path('/Users/1amaj/Documents/MY RESEARCH/FHN-paper/IJCAI/figures'),
        Path('/Users/1amaj/Documents/MY RESEARCH/FHN-paper/ICML-workshop/figures'),
        REPO_ROOT / 'research_outputs',
    ]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        out = d / 'bifurcation_diagram.png'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Saved {out}')

    plt.close(fig)


if __name__ == '__main__':
    main()
