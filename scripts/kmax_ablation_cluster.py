"""k_max ablation on the full 8000-sample dataset, run on the cluster."""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fhn_fno.config import DataConfig
from fhn_fno.models.fno import FNO


class H5SingleStepDataset(Dataset):
    """Single-step (u_t, v_t) -> (u_{t+1}, v_{t+1}) pairs with FiLM parameters."""

    def __init__(self, h5_path: str, indices: np.ndarray,
                 um: float, us: float, vm: float, vs: float,
                 pm: np.ndarray, ps: np.ndarray):
        with h5py.File(h5_path, 'r') as f:
            self.u = np.array(f['u_traj'][indices])  # (N, T, nx)
            self.v = np.array(f['v_traj'][indices])
            self.p = np.array(f['params'][indices])  # (N, 5)
        self.um, self.us, self.vm, self.vs = um, us, vm, vs
        self.pm, self.ps = pm, ps
        self.n_samples, self.n_times, self.nx = self.u.shape

    def __len__(self):
        return self.n_samples * (self.n_times - 1)

    def __getitem__(self, idx: int):
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


def compute_norm_stats(h5_path: str, train_indices: np.ndarray, cfg: DataConfig):
    """u, v stats from the training split; lambda stats from the configured ranges."""
    with h5py.File(h5_path, 'r') as f:
        u = np.array(f['u_traj'][train_indices])
        v = np.array(f['v_traj'][train_indices])
    um, us = float(u.mean()), float(u.std() + 1e-8)
    vm, vs = float(v.mean()), float(v.std() + 1e-8)
    ranges = np.array([cfg.Du_range, cfg.Dv_range, cfg.a_range,
                       cfg.b_range, cfg.tau_range])
    pm = ranges.mean(axis=1).astype(np.float32)
    ps = ((ranges[:, 1] - ranges[:, 0]) / np.sqrt(12.0)).astype(np.float32)
    return um, us, vm, vs, pm, ps


def split_indices(n_samples: int, train_frac: float = 0.8, seed: int = 42):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_samples)
    n_train = int(n_samples * train_frac)
    return perm[:n_train], perm[n_train:]


def train_one(k_max: int, seed: int, train_ds, val_ds, args, log_path: Path):
    """One training run at (k_max, seed), returning a metrics dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = args.device
    model = FNO(modes=k_max, width=args.width, n_layers=args.n_layers,
                in_channels=2, out_channels=2, dim=1,
                use_positional=False, use_param_conditioning=True, n_params=5).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device != 'cpu'))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device != 'cpu'))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_rel_u, best_rel_v = float('inf'), float('inf')
    best_epoch = -1
    with open(log_path, 'w') as flog:
        flog.write(f'# k_max={k_max} seed={seed} width={args.width} n_layers={args.n_layers}\n')
        for ep in range(args.epochs):
            model.train()
            tr_loss = 0.0
            for x, y, p in train_loader:
                x, y, p = x.to(device, non_blocking=True), y.to(device, non_blocking=True), p.to(device, non_blocking=True)
                optimizer.zero_grad()
                pred = model(x, p)
                loss = criterion(pred, y)
                loss.backward()
                # grad clipping only on CUDA: it breaks on complex spectral weights under MPS
                if args.grad_clip > 0 and device == 'cuda':
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                tr_loss += loss.item() * x.shape[0]
            tr_loss /= len(train_loader.dataset)

            # per-sample rel-L2, then averaged over the validation set
            model.eval()
            rel_u_list = []
            rel_v_list = []
            with torch.no_grad():
                for x, y, p in val_loader:
                    x, y, p = x.to(device), y.to(device), p.to(device)
                    pred = model(x, p)
                    num_u = ((pred[:, 0] - y[:, 0]) ** 2).sum(dim=-1)  # (B,)
                    den_u = (y[:, 0] ** 2).sum(dim=-1)
                    num_v = ((pred[:, 1] - y[:, 1]) ** 2).sum(dim=-1)
                    den_v = (y[:, 1] ** 2).sum(dim=-1)
                    rel_u_list.append(torch.sqrt(num_u / (den_u + 1e-12)).cpu().numpy())
                    rel_v_list.append(torch.sqrt(num_v / (den_v + 1e-12)).cpu().numpy())
            rel_u_all = np.concatenate(rel_u_list)
            rel_v_all = np.concatenate(rel_v_list)
            rel_u = float(rel_u_all.mean())
            rel_v = float(rel_v_all.mean())
            scheduler.step()

            flog.write(f'epoch {ep} train_mse {tr_loss:.6f} rel_u {rel_u:.6f} rel_v {rel_v:.6f}\n')
            flog.flush()
            if rel_u + rel_v < best_rel_u + best_rel_v:
                best_rel_u, best_rel_v, best_epoch = rel_u, rel_v, ep

    # forward-pass timing: median of 50 passes at batch 1
    nx = train_ds.nx
    model.eval()
    x_t = torch.randn(1, 2, nx, device=device)
    p_t = torch.zeros(1, 5, device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(x_t, p_t)
    if device == 'cuda':
        torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(50):
            t0 = time.time()
            _ = model(x_t, p_t)
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - t0)
    fwd_ms = float(np.median(times) * 1000)

    return {'k_max': k_max, 'seed': seed, 'n_params': n_params,
            'rel_u': best_rel_u, 'rel_v': best_rel_v,
            'best_epoch': best_epoch, 'fwd_ms': fwd_ms}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='HDF5 dataset (e.g. data/fhn_1d_8000.h5)')
    ap.add_argument('--output_dir', default='research_outputs/kmax_ablation')
    ap.add_argument('--k_values', type=int, nargs='+', default=[4, 8, 16, 32, 64])
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 1337, 2025])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--width', type=int, default=64,
                    help='Hidden dim. Use the same width as the main paper (64).')
    ap.add_argument('--n_layers', type=int, default=6,
                    help='Number of Fourier layers. Match the main paper (6).')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--train_frac', type=float, default=0.8)
    ap.add_argument('--split_seed', type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output dir: {out_dir}', flush=True)
    print(f'Device:     {args.device}', flush=True)

    with h5py.File(args.data, 'r') as f:
        n_samples = f['u_traj'].shape[0]
        nx = f['u_traj'].shape[-1]
    print(f'Dataset {args.data}: {n_samples} trajectories, nx={nx}', flush=True)

    cfg = DataConfig()
    train_idx, val_idx = split_indices(n_samples, args.train_frac, args.split_seed)
    train_idx.sort()
    val_idx.sort()

    print('Computing normalisation stats (training split)...', flush=True)
    um, us, vm, vs, pm, ps = compute_norm_stats(args.data, train_idx, cfg)
    print(f'u: mean={um:.4f} std={us:.4f}; v: mean={vm:.4f} std={vs:.4f}', flush=True)

    print('Loading datasets...', flush=True)
    train_ds = H5SingleStepDataset(args.data, train_idx, um, us, vm, vs, pm, ps)
    val_ds = H5SingleStepDataset(args.data, val_idx, um, us, vm, vs, pm, ps)
    print(f'  train: {len(train_ds.u)} trajectories, val: {len(val_ds.u)}', flush=True)

    all_runs = []
    for k_max in args.k_values:
        for seed in args.seeds:
            t0 = time.time()
            print(f'\n== k_max={k_max}, seed={seed} ==', flush=True)
            log_path = out_dir / f'run_k{k_max}_s{seed}.log'
            r = train_one(k_max, seed, train_ds, val_ds, args, log_path)
            r['wall_train_s'] = time.time() - t0
            print(f'  rel_u={r["rel_u"]:.4f} rel_v={r["rel_v"]:.4f} '
                  f'params={r["n_params"]:,} fwd={r["fwd_ms"]:.3f}ms '
                  f'best_epoch={r["best_epoch"]} train_wall={r["wall_train_s"]:.1f}s',
                  flush=True)
            all_runs.append(r)
            # write incrementally so a crash doesn't lose finished runs
            with open(out_dir / 'kmax_ablation.json', 'w') as f:
                json.dump(all_runs, f, indent=2)

    summary = {}
    for k in args.k_values:
        runs = [r for r in all_runs if r['k_max'] == k]
        if not runs:
            continue
        summary[k] = {
            'rel_u_mean': float(np.mean([r['rel_u'] for r in runs])),
            'rel_u_std': float(np.std([r['rel_u'] for r in runs])),
            'rel_v_mean': float(np.mean([r['rel_v'] for r in runs])),
            'rel_v_std': float(np.std([r['rel_v'] for r in runs])),
            'n_params': runs[0]['n_params'],
            'fwd_ms_mean': float(np.mean([r['fwd_ms'] for r in runs])),
        }

    print('\n' + '=' * 78)
    print(f'{"k_max":>6s} | {"rel_u (mean+/-std)":>22s} | {"rel_v":>22s} | {"params":>10s} | {"fwd(ms)":>8s}')
    print('=' * 78)
    for k in args.k_values:
        if k not in summary:
            continue
        s = summary[k]
        print(f'{k:>6d} | {s["rel_u_mean"]:.4f} +/- {s["rel_u_std"]:.4f}    | '
              f'{s["rel_v_mean"]:.4f} +/- {s["rel_v_std"]:.4f}    | '
              f'{s["n_params"]:>10,d} | {s["fwd_ms_mean"]:>8.3f}')

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    plt.rcParams['font.family'] = 'serif'
    ks = np.array(sorted(summary.keys()), dtype=float)
    ru_mean = np.array([summary[int(k)]['rel_u_mean'] for k in ks])
    ru_std = np.array([summary[int(k)]['rel_u_std'] for k in ks])
    rv_mean = np.array([summary[int(k)]['rel_v_mean'] for k in ks])
    rv_std = np.array([summary[int(k)]['rel_v_std'] for k in ks])
    n_par = np.array([summary[int(k)]['n_params'] for k in ks]) / 1000.0

    ax = axes[0]
    ax.errorbar(ks, ru_mean, yerr=ru_std, marker='o', label=r'rel. $L^2$, $u$',
                color='#1f77b4', linewidth=2, capsize=3)
    ax.errorbar(ks, rv_mean, yerr=rv_std, marker='s', label=r'rel. $L^2$, $v$',
                color='#ff7f0e', linewidth=2, capsize=3)
    ax.axvline(16, color='gray', linestyle='--', linewidth=1, alpha=0.7,
               label=r'chosen $k_{\max}\!=\!16$')
    ax.set_xscale('log', base=2)
    ax.set_xticks([int(k) for k in ks])
    ax.set_xticklabels([str(int(k)) for k in ks])
    ax.set_xlabel(r'$k_{\max}$')
    ax.set_ylabel(r'Validation rel. $L^2$ error')
    ax.set_title('Accuracy')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    ax = axes[1]
    ax.plot(ks, n_par, marker='o', color='#2ca02c', linewidth=2, label='Parameters')
    ax.axvline(16, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax.set_xscale('log', base=2)
    ax.set_xticks([int(k) for k in ks])
    ax.set_xticklabels([str(int(k)) for k in ks])
    ax.set_xlabel(r'$k_{\max}$')
    ax.set_ylabel(r'Parameter count ($\times 10^3$)')
    ax.set_title('Model size')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / 'kmax_ablation.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_dir / "kmax_ablation.png"}')

    with open(out_dir / 'kmax_ablation_summary.json', 'w') as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2)
    print(f'Saved {out_dir / "kmax_ablation_summary.json"}')


if __name__ == '__main__':
    main()
