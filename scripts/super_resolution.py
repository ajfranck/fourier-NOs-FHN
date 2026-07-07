"""
Zero-shot super-resolution for the parameter-conditioned FHN-FNO.
Evaluates the frozen model on finer grids than it was trained on and measures
how relative L2 error grows with resolution.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fhn_fno.config import FHNParams
from fhn_fno.data.generate_fhn import FDBackendGPU, sample_initial_conditions
from fhn_fno.models.fno import FNO

CKPT = REPO_ROOT / "checkpoints" / "param_fno_best.pt"
OUT_DIRS = [
    Path("/Users/1amaj/Documents/MY RESEARCH/FHN-paper/IJCAI/figures"),
    REPO_ROOT / "research_outputs",
]


def load_model(device="cpu"):
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    model = FNO(**ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    stats = dict(u_mean=ckpt["u_mean"], u_std=ckpt["u_std"],
                 v_mean=ckpt["v_mean"], v_std=ckpt["v_std"],
                 p_mean=np.asarray(ckpt["p_mean"], dtype=np.float32),
                 p_std=np.asarray(ckpt["p_std"], dtype=np.float32))
    return model, stats, ckpt["data_config"]


def spectral_upsample(arr, nx_hi):
    """Band-limited FFT zero-padding upsample of a periodic 1D signal."""
    nx_lo = arr.shape[-1]
    if nx_hi == nx_lo:
        return arr.copy()
    ft = np.fft.rfft(arr)
    ft_hi = np.zeros(nx_hi // 2 + 1, dtype=complex)
    keep = min(len(ft), len(ft_hi))
    ft_hi[:keep] = ft[:keep]
    return np.fft.irfft(ft_hi, n=nx_hi) * (nx_hi / nx_lo)


def normalize(u, v, stats):
    return np.stack([(u - stats["u_mean"]) / stats["u_std"],
                     (v - stats["v_mean"]) / stats["v_std"]], axis=0)


def denormalize(x, stats):
    u = x[0] * stats["u_std"] + stats["u_mean"]
    v = x[1] * stats["v_std"] + stats["v_mean"]
    return u, v


def rel_l2(pred, true):
    return float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-12))


def evaluate(resolutions=(128, 256, 512, 1024), n_traj=16, rollout_steps=20,
             device="cpu", seed=0):
    model, stats, dc = load_model(device)
    nx_train = int(dc["nx"])
    T, dt, n_save = float(dc["T"]), float(dc["dt"]), int(dc["n_timesteps"])
    ranges = [dc["Du_range"], dc["Dv_range"], dc["a_range"], dc["b_range"], dc["tau_range"]]
    rng = np.random.RandomState(seed)

    # Same IC seeds and parameters at every resolution so only the grid changes.
    ic_seeds = rng.randint(0, 2**31 - 1, size=n_traj)
    params = np.array([[rng.uniform(*r) for r in ranges] for _ in range(n_traj)],
                      dtype=np.float32)

    results = {nx: dict(ss_u=[], ss_v=[], ro_u=[], ro_v=[]) for nx in resolutions}
    qualitative = {}

    for nx in resolutions:
        solver = FDBackendGPU(nx=nx, dx=1.0 / nx, device=device, dtype=torch.float32)
        t0 = time.time()
        for k in range(n_traj):
            u0_lo, v0_lo = sample_initial_conditions(nx_train, None, ic_type="grf",
                                                     alpha=2.0, seed=int(ic_seeds[k]))
            u0 = spectral_upsample(u0_lo, nx)
            v0 = spectral_upsample(v0_lo, nx)
            p = FHNParams(*params[k])
            sol = solver.solve(u0, v0, p, T=T, dt=dt, n_save=n_save)
            u_gt, v_gt = sol["u"], sol["v"]            # (n_save+1, nx)

            p_norm = torch.tensor((params[k] - stats["p_mean"]) / stats["p_std"],
                                  dtype=torch.float32, device=device).unsqueeze(0)

            # Single-step error over all consecutive frames.
            xin = np.stack([normalize(u_gt[t], v_gt[t], stats) for t in range(n_save)])
            xin = torch.tensor(xin, dtype=torch.float32, device=device)
            with torch.no_grad():
                pred = model(xin, p_norm.expand(xin.shape[0], -1)).cpu().numpy()
            pu = pred[:, 0] * stats["u_std"] + stats["u_mean"]
            pv = pred[:, 1] * stats["v_std"] + stats["v_mean"]
            results[nx]["ss_u"].append(rel_l2(pu, u_gt[1:]))
            results[nx]["ss_v"].append(rel_l2(pv, v_gt[1:]))

            # Autoregressive rollout from frame 0.
            x = torch.tensor(normalize(u_gt[0], v_gt[0], stats),
                             dtype=torch.float32, device=device).unsqueeze(0)
            roll_u, roll_v = [], []
            with torch.no_grad():
                for _ in range(rollout_steps):
                    x = model(x, p_norm)
                    u_p, v_p = denormalize(x[0].cpu().numpy(), stats)
                    roll_u.append(u_p); roll_v.append(v_p)
            roll_u, roll_v = np.array(roll_u), np.array(roll_v)
            results[nx]["ro_u"].append(rel_l2(roll_u, u_gt[1:rollout_steps + 1]))
            results[nx]["ro_v"].append(rel_l2(roll_v, v_gt[1:rollout_steps + 1]))

            if k == 0:   # keep one example for the qualitative panel
                qualitative[nx] = (u_gt[1].copy(), pu[0].copy())
        print(f"nx={nx:5d}: {time.time()-t0:5.1f}s  "
              f"single-step relL2 u={np.mean(results[nx]['ss_u']):.4e} "
              f"v={np.mean(results[nx]['ss_v']):.4e}  | "
              f"{rollout_steps}-step rollout u={np.mean(results[nx]['ro_u']):.4e}")

    return results, qualitative, nx_train


def make_figures(results, qualitative, nx_train, rollout_steps):
    res = sorted(results)
    ss_u = [np.mean(results[n]["ss_u"]) for n in res]
    ss_v = [np.mean(results[n]["ss_v"]) for n in res]
    ro_u = [np.mean(results[n]["ro_u"]) for n in res]
    ss_u_sd = [np.std(results[n]["ss_u"]) for n in res]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.4))

    ax1.errorbar(res, ss_u, yerr=ss_u_sd, marker="o", capsize=3, label=r"single-step $u$")
    ax1.plot(res, ss_v, marker="s", label=r"single-step $v$")
    ax1.plot(res, ro_u, marker="^", linestyle="--",
             label=rf"{rollout_steps}-step rollout $u$")
    ax1.axvline(nx_train, color="gray", ls=":", lw=1.2)
    ax1.text(nx_train * 1.05, ax1.get_ylim()[1] * 0.6,
             f"train\n$n_x={nx_train}$", fontsize=8, color="gray")
    ax1.set_xscale("log", base=2); ax1.set_yscale("log")
    ax1.set_xticks(res); ax1.set_xticklabels(res)
    ax1.set_xlabel(r"evaluation resolution $n_x$")
    ax1.set_ylabel(r"relative $L^2$ error")
    ax1.set_title("Zero-shot super-resolution")
    ax1.legend(fontsize=8); ax1.grid(True, which="both", alpha=0.3)

    nx_hi = res[-1]
    u_gt, u_pred = qualitative[nx_hi]
    x = np.linspace(0, 1, nx_hi, endpoint=False)
    ax2.plot(x, u_gt, color="k", lw=1.6, label="solver (ground truth)")
    ax2.plot(x, u_pred, color="tab:red", lw=1.1, ls="--", label="FNO (zero-shot)")
    ax2.set_xlabel(r"$x$"); ax2.set_ylabel(r"$u(x)$")
    ax2.set_title(rf"Single-step prediction at $n_x={nx_hi}$"
                  rf" ({nx_hi // nx_train}$\times$ training)")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        out = d / "super_resolution.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"Saved {out}")
    plt.close(fig)


def main():
    rollout_steps = 20
    results, qualitative, nx_train = evaluate(rollout_steps=rollout_steps)
    make_figures(results, qualitative, nx_train, rollout_steps)

    print("\n=== Summary (mean relative L2) ===")
    print(f"{'nx':>6} {'single-step u':>14} {'single-step v':>14} "
          f"{str(rollout_steps)+'-step u':>14}")
    for nx in sorted(results):
        r = results[nx]
        print(f"{nx:6d} {np.mean(r['ss_u']):14.4e} {np.mean(r['ss_v']):14.4e} "
              f"{np.mean(r['ro_u']):14.4e}")


if __name__ == "__main__":
    main()
