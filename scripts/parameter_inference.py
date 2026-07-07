"""Recover FHN parameters from noisy voltage traces by gradient inversion through the FNO,
with an FD-solver inversion as the slow baseline."""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fhn_fno.config import DataConfig, FHNParams
from fhn_fno.data.generate_fhn import FDBackend, sample_initial_conditions
from fhn_fno.models.fno import FNO


def pick_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_checkpoint(path: Path, device: str):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = FNO(**ck['model_kwargs']).to(device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck


def generate_test_trajectory(cfg: DataConfig, lam_star: np.ndarray, nx: int,
                             T: float, dt: float, n_save: int, seed: int):
    """One ground-truth FD trajectory with known params."""
    rng = np.random.RandomState(seed)
    u0, v0 = sample_initial_conditions(nx, None, ic_type='grf',
                                        alpha=cfg.grf_alpha,
                                        seed=int(rng.randint(0, 2 ** 31 - 1)))
    solver = FDBackend(nx=nx, ny=None, dx=1.0 / nx, dy=1.0)
    p = FHNParams(Du=lam_star[0], Dv=lam_star[1], a=lam_star[2],
                  b=lam_star[3], tau=lam_star[4])
    sol = solver.solve(u0, v0, p, T=T, dt=dt, n_save=n_save, I_ext=None)
    u_arr = sol['u'][:n_save + 1]
    v_arr = sol['v'][:n_save + 1]
    if u_arr.shape[0] < n_save + 1:
        pad = n_save + 1 - u_arr.shape[0]
        u_arr = np.concatenate([u_arr, np.repeat(u_arr[-1:], pad, axis=0)], axis=0)
        v_arr = np.concatenate([v_arr, np.repeat(v_arr[-1:], pad, axis=0)], axis=0)
    return u0, v0, u_arr, v_arr


def fno_rollout(model: FNO, u0: torch.Tensor, v0: torch.Tensor,
                lam_norm: torch.Tensor, n_steps: int,
                norm_stats: dict) -> torch.Tensor:
    """Autoregressive rollout in normalized space, returned in physical units, shape (n_steps+1, 2, nx)."""
    um, us, vm, vs = norm_stats['u_mean'], norm_stats['u_std'], norm_stats['v_mean'], norm_stats['v_std']
    u = (u0 - um) / us
    v = (v0 - vm) / vs
    x = torch.stack([u, v], dim=0).unsqueeze(0)  # (1, 2, nx)
    traj = [x.clone()]
    p = lam_norm.unsqueeze(0)
    for _ in range(n_steps):
        x = model(x, p)
        traj.append(x)
    traj = torch.cat(traj, dim=0)  # (n_steps+1, 2, nx)
    u_phys = traj[:, 0] * us + um
    v_phys = traj[:, 1] * vs + vm
    return torch.stack([u_phys, v_phys], dim=1)


def invert_with_fno(model: FNO, u_obs: torch.Tensor, u0: torch.Tensor,
                    v0: torch.Tensor, n_steps: int, ck: dict,
                    p_lo: torch.Tensor, p_hi: torch.Tensor,
                    n_iters: int = 400, lr: float = 5e-2, device: str = 'cpu'):
    """Recover lambda by gradient descent through the FNO. lambda lives in normalized
    space and is clamped to the training bounds in physical units before each pass."""
    p_mean = torch.tensor(ck['p_mean'], device=device, dtype=torch.float32)
    p_std = torch.tensor(ck['p_std'], device=device, dtype=torch.float32)
    norm_stats = {'u_mean': ck['u_mean'], 'u_std': ck['u_std'],
                  'v_mean': ck['v_mean'], 'v_std': ck['v_std']}

    lam_norm = torch.zeros(5, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([lam_norm], lr=lr)

    history = []
    for it in range(n_iters):
        optimizer.zero_grad()
        lam_phys = lam_norm * p_std + p_mean
        # hard clamp to bounds (differentiable through the interior)
        lam_clamped = torch.maximum(torch.minimum(lam_phys, p_hi), p_lo)
        lam_use_norm = (lam_clamped - p_mean) / p_std

        traj = fno_rollout(model, u0, v0, lam_use_norm, n_steps, norm_stats)
        u_pred = traj[:, 0]
        loss = ((u_pred - u_obs) ** 2).mean()
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))

    with torch.no_grad():
        lam_final = lam_norm * p_std + p_mean
        lam_final = torch.maximum(torch.minimum(lam_final, p_hi), p_lo)
    return lam_final.detach().cpu().numpy(), history


def invert_with_fd(u_obs_np: np.ndarray, u0_np: np.ndarray, v0_np: np.ndarray,
                   nx: int, T: float, dt: float, n_save: int,
                   p_lo: np.ndarray, p_hi: np.ndarray,
                   n_iters: int = 30, eps: float = 1e-3):
    """Gradient-free baseline: random-direction line search over lambda with the FD solver."""
    solver = FDBackend(nx=nx, ny=None, dx=1.0 / nx, dy=1.0)
    p_mid = 0.5 * (p_lo + p_hi)
    best_lam = p_mid.copy()

    def loss_at(lam):
        params = FHNParams(Du=float(lam[0]), Dv=float(lam[1]), a=float(lam[2]),
                           b=float(lam[3]), tau=float(lam[4]))
        sol = solver.solve(u0_np, v0_np, params, T=T, dt=dt, n_save=n_save, I_ext=None)
        u_arr = sol['u'][:n_save + 1]
        if u_arr.shape[0] < n_save + 1:
            pad = n_save + 1 - u_arr.shape[0]
            u_arr = np.concatenate([u_arr, np.repeat(u_arr[-1:], pad, axis=0)], axis=0)
        return float(((u_arr - u_obs_np) ** 2).mean())

    best_loss = loss_at(best_lam)
    history = [best_loss]
    rng = np.random.RandomState(0)
    for it in range(n_iters):
        d = rng.randn(5)
        d /= np.linalg.norm(d) + 1e-12
        scale = (p_hi - p_lo) * 0.1
        for step in [-1.0, -0.3, 0.3, 1.0]:
            cand = np.clip(best_lam + step * scale * d, p_lo, p_hi)
            l = loss_at(cand)
            if l < best_loss:
                best_loss = l
                best_lam = cand
        history.append(best_loss)
    return best_lam, history


def main():
    ck_path = REPO_ROOT / 'checkpoints' / 'param_fno_best.pt'
    if not ck_path.exists():
        print(f'Checkpoint not found at {ck_path}. Run train_param_fno.py first.')
        sys.exit(1)

    device = pick_device()
    print(f'Using device: {device}')

    model, ck = load_checkpoint(ck_path, device)
    cfg = DataConfig()
    dc = ck['data_config']
    nx, T, dt, n_steps = dc['nx'], dc['T'], dc['dt'], dc['n_timesteps']

    p_lo = torch.tensor([cfg.Du_range[0], cfg.Dv_range[0], cfg.a_range[0],
                          cfg.b_range[0], cfg.tau_range[0]], device=device)
    p_hi = torch.tensor([cfg.Du_range[1], cfg.Dv_range[1], cfg.a_range[1],
                          cfg.b_range[1], cfg.tau_range[1]], device=device)
    p_lo_np = p_lo.cpu().numpy()
    p_hi_np = p_hi.cpu().numpy()

    n_trajectories = 8
    snrs_db = [np.inf, 20.0, 10.0]  # inf = noiseless

    rng = np.random.RandomState(7)
    results = {snr: {'true': [], 'recovered': [], 'walltime': []} for snr in snrs_db}

    # FD baseline runs only on the first 3 trajectories at SNR=20 dB
    fd_results = {'true': [], 'recovered': [], 'walltime': []}
    fd_subset_idx = list(range(3))

    partial_json = REPO_ROOT / 'research_outputs' / 'parameter_inference_partial.json'
    partial_json.parent.mkdir(parents=True, exist_ok=True)

    for traj_idx in range(n_trajectories):
        lam_star = np.array([
            rng.uniform(*cfg.Du_range),
            rng.uniform(*cfg.Dv_range),
            rng.uniform(*cfg.a_range),
            rng.uniform(*cfg.b_range),
            rng.uniform(*cfg.tau_range),
        ], dtype=np.float32)
        seed = 1000 + traj_idx
        u0_np, v0_np, u_traj_np, v_traj_np = generate_test_trajectory(
            cfg, lam_star, nx, T, dt, n_steps, seed)
        u_clean = torch.tensor(u_traj_np, device=device, dtype=torch.float32)
        u0 = torch.tensor(u0_np, device=device, dtype=torch.float32)
        v0 = torch.tensor(v0_np, device=device, dtype=torch.float32)

        for snr in snrs_db:
            if np.isfinite(snr):
                sigma_signal = float(np.std(u_traj_np))
                sigma_noise = sigma_signal / (10.0 ** (snr / 20.0))
                noise = torch.randn_like(u_clean) * sigma_noise
                u_obs = u_clean + noise
            else:
                u_obs = u_clean

            t0 = time.time()
            lam_rec, hist = invert_with_fno(model, u_obs, u0, v0, n_steps, ck,
                                            p_lo, p_hi, n_iters=150, lr=8e-2,
                                            device=device)
            wt = time.time() - t0
            results[snr]['true'].append(lam_star.copy())
            results[snr]['recovered'].append(lam_rec.copy())
            results[snr]['walltime'].append(wt)

        if traj_idx in fd_subset_idx:
            sigma_signal = float(np.std(u_traj_np))
            sigma_noise = sigma_signal / (10.0 ** (20.0 / 20.0))
            u_obs_np = u_traj_np + np.random.RandomState(seed).randn(*u_traj_np.shape) * sigma_noise
            t0 = time.time()
            lam_fd, _ = invert_with_fd(u_obs_np, u0_np, v0_np, nx, T, dt, n_steps,
                                        p_lo_np, p_hi_np, n_iters=15)
            wt_fd = time.time() - t0
            fd_results['true'].append(lam_star.copy())
            fd_results['recovered'].append(lam_fd.copy())
            fd_results['walltime'].append(wt_fd)
            print(f'  [traj {traj_idx}] FD inversion: {wt_fd:.1f}s')

        print(f'traj {traj_idx+1}/{n_trajectories} done', flush=True)

        # snapshot to disk after each trajectory so a kill doesn't lose everything
        snapshot = {}
        for snr in snrs_db:
            if results[snr]['true']:
                snapshot[str(snr)] = {
                    'true': [a.tolist() for a in results[snr]['true']],
                    'recovered': [a.tolist() for a in results[snr]['recovered']],
                    'walltime': results[snr]['walltime'],
                }
        if fd_results['true']:
            snapshot['fd'] = {
                'true': [a.tolist() for a in fd_results['true']],
                'recovered': [a.tolist() for a in fd_results['recovered']],
                'walltime': fd_results['walltime'],
            }
        with open(partial_json, 'w') as f:
            json.dump(snapshot, f, indent=2)

    param_names = ['D_u', 'D_v', 'a', 'b', 'tau']
    summary = {}
    print('\n' + '=' * 76)
    print(f'{"SNR (dB)":>10s} | ' + ' '.join(f'{n:>9s}' for n in param_names) + f' | {"walltime":>9s}')
    print('=' * 76)
    for snr in snrs_db:
        T_lam = np.stack(results[snr]['true'])
        R_lam = np.stack(results[snr]['recovered'])
        rel = np.abs(R_lam - T_lam) / (np.abs(T_lam) + 1e-8)
        med = np.median(rel, axis=0)
        wt_med = float(np.median(results[snr]['walltime']))
        label = 'inf' if not np.isfinite(snr) else f'{snr:.0f}'
        print(f'{label:>10s} | ' + ' '.join(f'{x:>9.3%}' for x in med) + f' | {wt_med:>7.2f}s')
        summary[label] = {
            'median_rel_err': med.tolist(),
            'walltime_median': wt_med,
            'true': T_lam.tolist(),
            'recovered': R_lam.tolist(),
        }

    if fd_results['walltime']:
        wt_fd_med = float(np.median(fd_results['walltime']))
        T_fd = np.stack(fd_results['true'])
        R_fd = np.stack(fd_results['recovered'])
        rel_fd = np.median(np.abs(R_fd - T_fd) / (np.abs(T_fd) + 1e-8), axis=0)
        print(f'{"FD (20dB)":>10s} | ' + ' '.join(f'{x:>9.3%}' for x in rel_fd) + f' | {wt_fd_med:>7.2f}s')
        summary['fd_baseline'] = {
            'median_rel_err': rel_fd.tolist(),
            'walltime_median': wt_fd_med,
            'n_inversions': len(fd_results['walltime']),
        }

    out_json = REPO_ROOT / 'research_outputs' / 'parameter_inference_summary.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary written to {out_json}')

    snr_for_fig = 20.0
    T_lam = np.stack(results[snr_for_fig]['true'])
    R_lam = np.stack(results[snr_for_fig]['recovered'])
    fig, axes = plt.subplots(1, 5, figsize=(14, 3))
    for i, name in enumerate(param_names):
        ax = axes[i]
        ax.scatter(T_lam[:, i], R_lam[:, i], s=30, alpha=0.85, color='#1f77b4', edgecolor='black', linewidth=0.5)
        lo, hi = float(p_lo_np[i]), float(p_hi_np[i])
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
        ax.set_xlabel(f'true ${name}$')
        ax.set_ylabel(f'recovered ${name}$' if i == 0 else '')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
    fig.suptitle(rf'Parameter recovery, SNR = {snr_for_fig:.0f} dB (n={len(T_lam)})')
    plt.tight_layout()
    out_dirs = [
        Path('/Users/1amaj/Documents/MY RESEARCH/FHN-paper/IJCAI/figures'),
        Path('/Users/1amaj/Documents/MY RESEARCH/FHN-paper/ICML-workshop/figures'),
        REPO_ROOT / 'research_outputs',
    ]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / 'parameter_inference_scatter.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved parameter_inference_scatter.png')


if __name__ == '__main__':
    main()
