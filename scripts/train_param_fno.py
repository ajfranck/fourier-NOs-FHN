"""
Train a parameter-conditioned FNO surrogate for FHN with FiLM modulation.
Generates a small dataset on the fly and trains on single-step pairs, saving the
checkpoint with parameter normalization stats for downstream scripts.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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


def generate_dataset_inmemory(
    n_train: int = 192,
    n_val: int = 64,
    nx: int = 128,
    T: float = 2.0,
    dt: float = 0.01,
    n_timesteps: int = 40,
    cfg: DataConfig = None,
    seed: int = 0,
):
    cfg = cfg or DataConfig()
    rng = np.random.RandomState(seed)
    solver = FDBackend(nx=nx, ny=None, dx=1.0 / nx, dy=1.0)

    n_total = n_train + n_val
    u_traj = np.zeros((n_total, n_timesteps + 1, nx), dtype=np.float32)
    v_traj = np.zeros((n_total, n_timesteps + 1, nx), dtype=np.float32)
    params = np.zeros((n_total, 5), dtype=np.float32)

    for i in range(n_total):
        u0, v0 = sample_initial_conditions(nx, None, ic_type='grf',
                                            alpha=cfg.grf_alpha, seed=int(rng.randint(0, 2**31 - 1)))
        p = FHNParams(
            Du=float(rng.uniform(*cfg.Du_range)),
            Dv=float(rng.uniform(*cfg.Dv_range)),
            a=float(rng.uniform(*cfg.a_range)),
            b=float(rng.uniform(*cfg.b_range)),
            tau=float(rng.uniform(*cfg.tau_range)),
        )
        sol = solver.solve(u0, v0, p, T=T, dt=dt, n_save=n_timesteps, I_ext=None)
        u_arr = sol['u'][:n_timesteps + 1]
        v_arr = sol['v'][:n_timesteps + 1]
        if u_arr.shape[0] < n_timesteps + 1:
            # pad short runs by repeating the last frame
            pad = n_timesteps + 1 - u_arr.shape[0]
            u_arr = np.concatenate([u_arr, np.repeat(u_arr[-1:], pad, axis=0)], axis=0)
            v_arr = np.concatenate([v_arr, np.repeat(v_arr[-1:], pad, axis=0)], axis=0)
        u_traj[i] = u_arr
        v_traj[i] = v_arr
        params[i] = [p.Du, p.Dv, p.a, p.b, p.tau]
        if (i + 1) % 16 == 0:
            print(f'  generated {i+1}/{n_total}')

    return {
        'u_traj': u_traj,
        'v_traj': v_traj,
        'params': params,
        'config': dict(nx=nx, T=T, dt=dt, n_timesteps=n_timesteps, n_train=n_train, n_val=n_val),
    }


class SingleStepDataset(Dataset):
    def __init__(self, u_traj: np.ndarray, v_traj: np.ndarray, params: np.ndarray,
                 u_mean: float, u_std: float, v_mean: float, v_std: float,
                 p_mean: np.ndarray, p_std: np.ndarray):
        self.u = u_traj
        self.v = v_traj
        self.p = params
        self.um, self.us, self.vm, self.vs = u_mean, u_std, v_mean, v_std
        self.pm, self.ps = p_mean, p_std
        self.n_samples, self.n_times, _ = u_traj.shape

    def __len__(self):
        return self.n_samples * (self.n_times - 1)

    def __getitem__(self, idx):
        i = idx // (self.n_times - 1)
        t = idx % (self.n_times - 1)
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


def compute_stats(u_traj: np.ndarray, v_traj: np.ndarray, params: np.ndarray, cfg: DataConfig):
    u_mean = float(u_traj.mean())
    u_std = float(u_traj.std() + 1e-8)
    v_mean = float(v_traj.mean())
    v_std = float(v_traj.std() + 1e-8)
    # Normalize params by their configured ranges so inversion keeps physical units.
    ranges = np.array([cfg.Du_range, cfg.Dv_range, cfg.a_range, cfg.b_range, cfg.tau_range])
    p_mean = ranges.mean(axis=1)
    p_std = (ranges[:, 1] - ranges[:, 0]) / np.sqrt(12.0)  # std of uniform on [lo, hi]
    return u_mean, u_std, v_mean, v_std, p_mean.astype(np.float32), p_std.astype(np.float32)


def main():
    out_dir = REPO_ROOT / 'checkpoints'
    out_dir.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f'Using device: {device}')

    cfg = DataConfig()

    n_train, n_val = 192, 64
    nx = 128
    T = 2.0
    dt = 0.01
    n_timesteps = 40

    cache_path = REPO_ROOT / 'data' / 'param_fno_cache.npz'
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        print(f'Loading cached dataset from {cache_path}')
        npz = np.load(cache_path, allow_pickle=True)
        data = {'u_traj': npz['u_traj'], 'v_traj': npz['v_traj'], 'params': npz['params'],
                'config': npz['config'].item()}
    else:
        t0 = time.time()
        print('Generating dataset...')
        data = generate_dataset_inmemory(
            n_train=n_train, n_val=n_val, nx=nx, T=T, dt=dt,
            n_timesteps=n_timesteps, cfg=cfg, seed=42)
        print(f'Data generation took {time.time() - t0:.1f}s')
        np.savez(cache_path, u_traj=data['u_traj'], v_traj=data['v_traj'],
                 params=data['params'], config=data['config'])

    u_mean, u_std, v_mean, v_std, p_mean, p_std = compute_stats(
        data['u_traj'][:n_train], data['v_traj'][:n_train], data['params'][:n_train], cfg)
    print(f'u stats: mean={u_mean:.3f} std={u_std:.3f}; v stats: mean={v_mean:.3f} std={v_std:.3f}')
    print(f'param mean: {p_mean}')
    print(f'param std:  {p_std}')

    train_ds = SingleStepDataset(
        data['u_traj'][:n_train], data['v_traj'][:n_train], data['params'][:n_train],
        u_mean, u_std, v_mean, v_std, p_mean, p_std)
    val_ds = SingleStepDataset(
        data['u_traj'][n_train:], data['v_traj'][n_train:], data['params'][n_train:],
        u_mean, u_std, v_mean, v_std, p_mean, p_std)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    model = FNO(modes=16, width=48, n_layers=4, in_channels=2, out_channels=2,
                dim=1, use_positional=False, use_param_conditioning=True, n_params=5).to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 60
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    best_val = float('inf')
    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y, p in train_loader:
            x, y, p = x.to(device), y.to(device), p.to(device)
            optimizer.zero_grad()
            pred = model(x, p)
            loss = criterion(pred, y)
            loss.backward()
            # grad-norm clipping breaks on complex spectral weights with this
            # torch+MPS build, so rely on small LR plus cosine schedule instead
            optimizer.step()
            train_loss += loss.item() * x.shape[0]
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        rel_u_num = rel_u_den = rel_v_num = rel_v_den = 0.0
        with torch.no_grad():
            for x, y, p in val_loader:
                x, y, p = x.to(device), y.to(device), p.to(device)
                pred = model(x, p)
                val_loss += criterion(pred, y).item() * x.shape[0]
                rel_u_num += ((pred[:, 0] - y[:, 0]) ** 2).sum().item()
                rel_u_den += (y[:, 0] ** 2).sum().item()
                rel_v_num += ((pred[:, 1] - y[:, 1]) ** 2).sum().item()
                rel_v_den += (y[:, 1] ** 2).sum().item()
        val_loss /= len(val_loader.dataset)
        rel_u = (rel_u_num / max(rel_u_den, 1e-12)) ** 0.5
        rel_v = (rel_v_num / max(rel_v_den, 1e-12)) ** 0.5

        scheduler.step()
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'u_mean': u_mean, 'u_std': u_std, 'v_mean': v_mean, 'v_std': v_std,
                'p_mean': p_mean, 'p_std': p_std,
                'model_kwargs': dict(modes=16, width=48, n_layers=4, in_channels=2, out_channels=2,
                                     dim=1, use_positional=False, use_param_conditioning=True, n_params=5),
                'data_config': dict(nx=nx, T=T, dt=dt, n_timesteps=n_timesteps,
                                    Du_range=cfg.Du_range, Dv_range=cfg.Dv_range, a_range=cfg.a_range,
                                    b_range=cfg.b_range, tau_range=cfg.tau_range),
            }, out_dir / 'param_fno_best.pt')
        if ep % 5 == 0 or ep == epochs - 1:
            print(f'epoch {ep:3d} | train {train_loss:.5f} | val {val_loss:.5f} | rel_L2 u {rel_u:.4f} v {rel_v:.4f}')

    print(f'Done. Best val MSE: {best_val:.5f}. Checkpoint at {out_dir / "param_fno_best.pt"}')


if __name__ == '__main__':
    main()
