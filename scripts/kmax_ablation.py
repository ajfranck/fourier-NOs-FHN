"""
k_max ablation: train FiLM-FNO at several Fourier-mode truncation levels and
plot validation error, parameter count, and wall-clock against k_max.
Uses the cached dataset produced by train_param_fno.py.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fhn_fno.config import DataConfig
from fhn_fno.models.fno import FNO


def pick_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


class SingleStepDataset(Dataset):
    def __init__(self, u_traj, v_traj, params, um, us, vm, vs, pm, ps):
        self.u, self.v, self.p = u_traj, v_traj, params
        self.um, self.us, self.vm, self.vs = um, us, vm, vs
        self.pm, self.ps = pm, ps
        self.n_samples, self.n_times, _ = u_traj.shape

    def __len__(self):
        return self.n_samples * (self.n_times - 1)

    def __getitem__(self, idx):
        i, t = divmod(idx, self.n_times - 1)
        u_in = (self.u[i, t] - self.um) / self.us
        v_in = (self.v[i, t] - self.vm) / self.vs
        u_out = (self.u[i, t + 1] - self.um) / self.us
        v_out = (self.v[i, t + 1] - self.vm) / self.vs
        p_norm = (self.p[i] - self.pm) / self.ps
        x = np.stack([u_in, v_in], axis=0)
        y = np.stack([u_out, v_out], axis=0)
        return (torch.from_numpy(x).float(),
                torch.from_numpy(y).float(),
                torch.from_numpy(p_norm).float())


def compute_stats(u_traj, v_traj, params, cfg):
    um, us = float(u_traj.mean()), float(u_traj.std() + 1e-8)
    vm, vs = float(v_traj.mean()), float(v_traj.std() + 1e-8)
    ranges = np.array([cfg.Du_range, cfg.Dv_range, cfg.a_range, cfg.b_range, cfg.tau_range])
    pm = ranges.mean(axis=1).astype(np.float32)
    ps = ((ranges[:, 1] - ranges[:, 0]) / np.sqrt(12.0)).astype(np.float32)
    return um, us, vm, vs, pm, ps


def train_one(k_max: int, seed: int, train_ds, val_ds, device: str,
              epochs: int = 40, lr: float = 1e-3, width: int = 48, n_layers: int = 4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FNO(modes=k_max, width=width, n_layers=n_layers,
                in_channels=2, out_channels=2, dim=1,
                use_positional=False, use_param_conditioning=True, n_params=5).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_rel_u, best_rel_v = float('inf'), float('inf')
    for ep in range(epochs):
        model.train()
        for x, y, p in train_loader:
            x, y, p = x.to(device), y.to(device), p.to(device)
            optimizer.zero_grad()
            pred = model(x, p)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        model.eval()
        ru_n = ru_d = rv_n = rv_d = 0.0
        with torch.no_grad():
            for x, y, p in val_loader:
                x, y, p = x.to(device), y.to(device), p.to(device)
                pred = model(x, p)
                ru_n += ((pred[:, 0] - y[:, 0]) ** 2).sum().item()
                ru_d += (y[:, 0] ** 2).sum().item()
                rv_n += ((pred[:, 1] - y[:, 1]) ** 2).sum().item()
                rv_d += (y[:, 1] ** 2).sum().item()
        rel_u = (ru_n / max(ru_d, 1e-12)) ** 0.5
        rel_v = (rv_n / max(rv_d, 1e-12)) ** 0.5
        if rel_u + rel_v < best_rel_u + best_rel_v:
            best_rel_u, best_rel_v = rel_u, rel_v
        scheduler.step()

    # Forward-pass timing: median of 30 passes at batch 1.
    nx = train_ds.u.shape[-1]
    model.eval()
    x_t = torch.randn(1, 2, nx, device=device)
    p_t = torch.zeros(1, 5, device=device)
    with torch.no_grad():
        for _ in range(5):
            _ = model(x_t, p_t)
    if device == 'mps':
        torch.mps.synchronize()
    elif device == 'cuda':
        torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(30):
            t0 = time.time()
            _ = model(x_t, p_t)
            if device == 'mps':
                torch.mps.synchronize()
            elif device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - t0)
    fwd_ms = float(np.median(times) * 1000)

    return {'k_max': k_max, 'seed': seed, 'n_params': n_params,
            'rel_u': best_rel_u, 'rel_v': best_rel_v, 'fwd_ms': fwd_ms}


def main():
    device = pick_device()
    print(f'Using device: {device}')
    cache_path = REPO_ROOT / 'data' / 'param_fno_cache.npz'
    if not cache_path.exists():
        print(f'Cached dataset not found at {cache_path}. Run train_param_fno.py first.')
        sys.exit(1)
    npz = np.load(cache_path, allow_pickle=True)
    u_traj = npz['u_traj']
    v_traj = npz['v_traj']
    params = npz['params']
    config_obj = npz['config'].item()
    n_train = config_obj['n_train']

    cfg = DataConfig()
    um, us, vm, vs, pm, ps = compute_stats(
        u_traj[:n_train], v_traj[:n_train], params[:n_train], cfg)
    train_ds = SingleStepDataset(u_traj[:n_train], v_traj[:n_train], params[:n_train],
                                  um, us, vm, vs, pm, ps)
    val_ds = SingleStepDataset(u_traj[n_train:], v_traj[n_train:], params[n_train:],
                                um, us, vm, vs, pm, ps)
    print(f'Dataset: {n_train} train / {len(u_traj) - n_train} val trajectories')

    k_values = [4, 8, 16, 32, 64]
    seeds = [42, 1337]

    all_runs = []
    for k_max in k_values:
        for seed in seeds:
            t0 = time.time()
            print(f'== Training k_max={k_max:3d}, seed={seed} ==', flush=True)
            r = train_one(k_max, seed, train_ds, val_ds, device, epochs=40)
            r['wall_train_s'] = time.time() - t0
            print(f'  rel_u={r["rel_u"]:.4f}, rel_v={r["rel_v"]:.4f}, '
                  f'params={r["n_params"]:,}, fwd={r["fwd_ms"]:.2f} ms, '
                  f'train_wall={r["wall_train_s"]:.1f}s', flush=True)
            all_runs.append(r)

    out_json = REPO_ROOT / 'research_outputs' / 'kmax_ablation.json'
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(all_runs, f, indent=2)
    print(f'\nSaved {out_json}')

    summary = {}
    for k in k_values:
        runs = [r for r in all_runs if r['k_max'] == k]
        summary[k] = {
            'rel_u_mean': float(np.mean([r['rel_u'] for r in runs])),
            'rel_u_std': float(np.std([r['rel_u'] for r in runs])),
            'rel_v_mean': float(np.mean([r['rel_v'] for r in runs])),
            'rel_v_std': float(np.std([r['rel_v'] for r in runs])),
            'n_params': runs[0]['n_params'],
            'fwd_ms_mean': float(np.mean([r['fwd_ms'] for r in runs])),
        }

    print('\n' + '=' * 70)
    print(f'{"k_max":>6s} | {"rel_u (mean+/-std)":>22s} | {"rel_v":>22s} | {"params":>10s} | {"fwd(ms)":>8s}')
    print('=' * 70)
    for k in k_values:
        s = summary[k]
        print(f'{k:>6d} | {s["rel_u_mean"]:.4f} +/- {s["rel_u_std"]:.4f}    | '
              f'{s["rel_v_mean"]:.4f} +/- {s["rel_v_std"]:.4f}    | '
              f'{s["n_params"]:>10,d} | {s["fwd_ms_mean"]:>8.2f}')

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    ks = np.array(k_values, dtype=float)
    ru_mean = np.array([summary[k]['rel_u_mean'] for k in k_values])
    ru_std = np.array([summary[k]['rel_u_std'] for k in k_values])
    rv_mean = np.array([summary[k]['rel_v_mean'] for k in k_values])
    rv_std = np.array([summary[k]['rel_v_std'] for k in k_values])
    n_par = np.array([summary[k]['n_params'] for k in k_values]) / 1000.0

    ax = axes[0]
    ax.errorbar(ks, ru_mean, yerr=ru_std, marker='o', label='rel. $L^2$, $u$', color='#1f77b4', linewidth=2)
    ax.errorbar(ks, rv_mean, yerr=rv_std, marker='s', label='rel. $L^2$, $v$', color='#ff7f0e', linewidth=2)
    ax.axvline(16, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='chosen $k_{\\max}\\!=\\!16$')
    ax.set_xscale('log', base=2)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_xlabel(r'$k_{\max}$')
    ax.set_ylabel(r'Validation rel. $L^2$ error')
    ax.set_title('Accuracy')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    ax = axes[1]
    ax.plot(ks, n_par, marker='o', color='#2ca02c', linewidth=2, label='Parameters (k)')
    ax.axvline(16, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xscale('log', base=2)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])
    ax.set_xlabel(r'$k_{\max}$')
    ax.set_ylabel(r'Parameter count (\,$\times 10^3$)')
    ax.set_title('Model size')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dirs = [
        Path('/Users/1amaj/Documents/MY RESEARCH/FHN-paper/IJCAI/figures'),
        Path('/Users/1amaj/Documents/MY RESEARCH/FHN-paper/ICML-workshop/figures'),
        REPO_ROOT / 'research_outputs',
    ]
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / 'kmax_ablation.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('Saved kmax_ablation.png')


if __name__ == '__main__':
    main()
