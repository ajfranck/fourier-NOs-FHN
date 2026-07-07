"""
Two-regime bifurcation map for the spatially-homogeneous FitzHugh-Nagumo ODE.
Classifies the (a, b) plane and overlays both the oscillatory and excitable
training boxes used in the paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Oscillatory training box, from DataConfig if available else hard-coded.
try:
    from fhn_fno.config import DataConfig
    _cfg = DataConfig()
    OSC_A = tuple(_cfg.a_range)
    OSC_B = tuple(_cfg.b_range)
    OSC_TAU = tuple(_cfg.tau_range)
except Exception:
    OSC_A, OSC_B, OSC_TAU = (-0.1, 0.1), (0.1, 0.5), (1.0, 20.0)

# Excitable training box (second surrogate).
EXC_A, EXC_B, EXC_TAU = (0.70, 0.95), (0.60, 1.00), (8.0, 20.0)


def fixed_points(a: float, b: float):
    """Real roots of b u^3 - 3 (b - 1) u + 3 a = 0, returned as (u*, v*)."""
    coeffs = [b, 0.0, -3.0 * (b - 1.0), 3.0 * a]
    out = []
    for r in np.roots(coeffs):
        if abs(r.imag) < 1e-8:
            u_star = float(r.real)
            out.append((u_star, (u_star + a) / b))
    return out


def classify(a: float, b: float, tau: float):
    """Regime label: excitable, oscillatory, bistable, or degenerate."""
    fps = fixed_points(a, b)
    n = len(fps)
    if n == 1:
        u_star, _ = fps[0]
        trace = 1.0 - u_star ** 2 - b / tau
        det = (1.0 - b * (1.0 - u_star ** 2)) / tau
        if det <= 0:
            return 'degenerate'
        return 'excitable' if trace < 0 else 'oscillatory'
    if n == 3:
        return 'bistable'
    return 'degenerate'


def sweep_box(a_rng, b_rng, tau_rng, n=60, ntau=20):
    """Percentage of a box (over its own tau range) in each regime."""
    counts = {'excitable': 0, 'oscillatory': 0, 'bistable': 0, 'degenerate': 0}
    for tau in np.linspace(*tau_rng, ntau):
        for a in np.linspace(*a_rng, n):
            for b in np.linspace(*b_rng, n):
                counts[classify(a, b, tau)] += 1
    total = sum(counts.values())
    return {k: 100.0 * v / total for k, v in counts.items()}


def main():
    tau_med = 0.5 * (OSC_TAU[0] + OSC_TAU[1])

    # Grid extends in a to include the excitable box.
    a_plot = np.linspace(-0.4, 1.1, 501)
    b_plot = np.linspace(0.02, 1.30, 501)
    AA, BB = np.meshgrid(a_plot, b_plot, indexing='ij')
    label_to_id = {'excitable': 0, 'oscillatory': 1, 'bistable': 2, 'degenerate': 3}
    regime = np.empty(AA.shape, dtype=np.int8)
    for i in range(AA.shape[0]):
        for j in range(AA.shape[1]):
            regime[i, j] = label_to_id[classify(AA[i, j], BB[i, j], tau_med)]

    osc = sweep_box(OSC_A, OSC_B, OSC_TAU)
    exc = sweep_box(EXC_A, EXC_B, EXC_TAU)
    print(f"Median tau for 2D map: {tau_med:.1f}")
    print("Oscillatory box:", {k: f"{v:.1f}%" for k, v in osc.items() if v > 0})
    print("Excitable box:  ", {k: f"{v:.1f}%" for k, v in exc.items() if v > 0})

    cmap = plt.matplotlib.colors.ListedColormap(
        ['#7DCEA0', '#F5B041', '#AF7AC5', '#BFC9CA'])
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    extent = (a_plot.min(), a_plot.max(), b_plot.min(), b_plot.max())
    ax.imshow(regime.T, origin='lower', extent=extent, aspect='auto',
              cmap=cmap, vmin=-0.5, vmax=3.5, interpolation='nearest')

    ax.add_patch(mpatches.Rectangle(
        (OSC_A[0], OSC_B[0]), OSC_A[1] - OSC_A[0], OSC_B[1] - OSC_B[0],
        linewidth=2.2, edgecolor='black', facecolor='none'))
    ax.annotate('Oscillatory\ntraining box', xy=(0.0, OSC_B[1]),
                xytext=(-0.34, 0.72), fontsize=8.5, ha='left', va='center',
                color='black',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    ax.add_patch(mpatches.Rectangle(
        (EXC_A[0], EXC_B[0]), EXC_A[1] - EXC_A[0], EXC_B[1] - EXC_B[0],
        linewidth=2.2, edgecolor='#1A5276', facecolor='none', linestyle='-'))
    ax.annotate('Excitable\ntraining box', xy=(EXC_A[1], EXC_B[0]),
                xytext=(0.98, 0.30), fontsize=8.5, ha='left', va='center',
                color='#1A5276',
                arrowprops=dict(arrowstyle='->', color='#1A5276', lw=1.2))

    handles = [
        mpatches.Patch(color='#7DCEA0', label='Excitable (stable rest)'),
        mpatches.Patch(color='#F5B041', label='Oscillatory (limit cycle)'),
        mpatches.Patch(color='#AF7AC5', label='Bistable'),
        mpatches.Patch(facecolor='none', edgecolor='black', label='Oscillatory box'),
        mpatches.Patch(facecolor='none', edgecolor='#1A5276', label='Excitable box'),
    ]
    ax.legend(handles=handles, loc='upper center', ncol=1, fontsize=7.5,
              framealpha=0.95)

    ax.set_xlabel(r'$a$', fontsize=11)
    ax.set_ylabel(r'$b$', fontsize=11)
    ax.set_title(rf'FHN dynamical regimes at $\tau = {tau_med:.1f}$', fontsize=11)
    ax.grid(False)

    out_dirs = [
        Path('/Users/1amaj/Documents/MY RESEARCH/FHN-paper/IJCAI/figures'),
        REPO_ROOT / 'research_outputs',
    ]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        out = d / 'bifurcation_two_regime.png'
        fig.savefig(out, dpi=300, bbox_inches='tight')
        print(f'Saved {out}')
    plt.close(fig)


if __name__ == '__main__':
    main()
