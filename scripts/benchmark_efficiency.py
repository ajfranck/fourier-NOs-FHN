"""Efficiency benchmark comparing the FD solvers against the FNO on equal hardware."""

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch

import sys
sys.path.append(".")

from fhn_fno.config import Config, FHNParams
from fhn_fno.data.generate_fhn import FDBackend, FDBackendGPU, sample_initial_conditions
from fhn_fno.models.fno import FNO


def _time_fn(fn, n_warmup: int, n_runs: int, cuda_sync: bool):
    """Time a callable, returning mean and std in ms."""
    for _ in range(n_warmup):
        fn()
    if cuda_sync:
        torch.cuda.synchronize()

    samples = []
    for _ in range(n_runs):
        if cuda_sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda_sync:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples = np.asarray(samples)
    return samples.mean(), samples.std()


def validate_gpu_matches_cpu(nx: int, T: float, dt: float, n_save: int,
                              device: str) -> None:
    """Check FD(GPU) matches FD(CPU) on one trajectory before trusting timings."""
    print("\n[validation] FD(GPU) vs FD(CPU) ...")
    u0, v0 = sample_initial_conditions(nx, None, ic_type="grf", alpha=2.0, seed=0)
    params = FHNParams(Du=0.05, Dv=0.01, a=0.0, b=0.2, tau=5.0)

    fd_cpu = FDBackend(nx=nx, dx=1.0 / nx)
    fd_gpu = FDBackendGPU(nx=nx, dx=1.0 / nx, device=device, dtype=torch.float64)

    out_cpu = fd_cpu.solve(u0, v0, params, T=T, dt=dt, n_save=n_save)
    out_gpu = fd_gpu.solve(u0, v0, params, T=T, dt=dt, n_save=n_save)

    err_u = float(np.max(np.abs(out_cpu["u"] - out_gpu["u"])))
    err_v = float(np.max(np.abs(out_cpu["v"] - out_gpu["v"])))
    print(f"  max |u_cpu - u_gpu| = {err_u:.3e}")
    print(f"  max |v_cpu - v_gpu| = {err_v:.3e}")
    tol = 1e-6
    if err_u > tol or err_v > tol:
        print(f"  WARNING: errors exceed tol={tol:.0e}. Investigate before trusting timings.")
    else:
        print(f"  OK: within tol={tol:.0e}")


def load_fno(checkpoint_path: str, data_file: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", None)
    if cfg_dict is None:
        cfg = Config()
    else:
        cfg = Config()
        for k, v in cfg_dict.get("model", {}).items():
            if hasattr(cfg.model, k):
                setattr(cfg.model, k, v)

    model = FNO(
        modes=cfg.model.modes,
        width=cfg.model.width,
        n_layers=cfg.model.n_layers,
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        dim=1,
        use_positional=cfg.model.use_positional,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def benchmark(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: CUDA not available; 'GPU' benchmarks will run on CPU.")

    print(f"Device: {device}")
    print(f"Problem: nx={args.nx}, n_steps={args.n_steps}, dt={args.dt}")
    print(f"Runs: warmup={args.n_warmup}, timed={args.n_runs}")

    if args.validate:
        validate_gpu_matches_cpu(args.nx, args.n_steps * args.dt, args.dt,
                                 args.n_steps, device)

    u0, v0 = sample_initial_conditions(args.nx, None, ic_type="grf", alpha=2.0, seed=0)
    params = FHNParams(Du=0.05, Dv=0.01, a=0.0, b=0.2, tau=5.0)
    T = args.n_steps * args.dt

    results = {}

    print("\n[bench] FD (CPU, NumPy) ...")
    fd_cpu = FDBackend(nx=args.nx, dx=1.0 / args.nx)

    def run_fd_cpu():
        fd_cpu.solve(u0, v0, params, T=T, dt=args.dt, n_save=args.n_steps)

    mean, std = _time_fn(run_fd_cpu, n_warmup=3, n_runs=min(args.n_runs, 10),
                         cuda_sync=False)
    results["fd_cpu"] = {"total_ms": mean, "std_ms": std, "per_step_ms": mean / args.n_steps}
    print(f"  total: {mean:.2f} +/- {std:.2f} ms   per-step: {mean/args.n_steps:.3f} ms")

    print("\n[bench] FD (GPU, batch=1) ...")
    fd_gpu = FDBackendGPU(nx=args.nx, dx=1.0 / args.nx, device=device,
                          dtype=torch.float32)
    u0_t = torch.from_numpy(u0).to(device, torch.float32).unsqueeze(0)
    v0_t = torch.from_numpy(v0).to(device, torch.float32).unsqueeze(0)

    def run_fd_gpu_1():
        fd_gpu.solve_batched(u0_t, v0_t, params, T=T, dt=args.dt, n_save=args.n_steps)

    mean, std = _time_fn(run_fd_gpu_1, n_warmup=args.n_warmup, n_runs=args.n_runs,
                         cuda_sync=(device == "cuda"))
    results["fd_gpu_b1"] = {"total_ms": mean, "std_ms": std, "per_step_ms": mean / args.n_steps}
    print(f"  total: {mean:.3f} +/- {std:.3f} ms   per-step: {mean/args.n_steps:.4f} ms")

    print("\n[bench] FD (GPU, batch=32) ...")
    B = 32
    u0_b = u0_t.expand(B, -1).contiguous()
    v0_b = v0_t.expand(B, -1).contiguous()

    def run_fd_gpu_b():
        fd_gpu.solve_batched(u0_b, v0_b, params, T=T, dt=args.dt, n_save=args.n_steps)

    mean, std = _time_fn(run_fd_gpu_b, n_warmup=args.n_warmup, n_runs=args.n_runs,
                         cuda_sync=(device == "cuda"))
    results["fd_gpu_b32"] = {
        "total_ms": mean, "std_ms": std,
        "per_step_ms": mean / args.n_steps,
        "per_step_per_traj_ms": mean / args.n_steps / B,
        "per_traj_total_ms": mean / B,
    }
    print(f"  total: {mean:.3f} +/- {std:.3f} ms   per-traj: {mean/B:.4f} ms   "
          f"per-step-per-traj: {mean/args.n_steps/B:.5f} ms")

    if args.checkpoint:
        print(f"\n[bench] FNO loading from {args.checkpoint} ...")
        model = load_fno(args.checkpoint, args.data, device)
        print(f"  loaded: {sum(p.numel() for p in model.parameters()):,} params")

        # conditioning vector, ignored if the model wasn't trained with params
        param_vec = torch.tensor(
            [[params.Du, params.Dv, params.a, params.b, params.tau]],
            device=device, dtype=torch.float32,
        )

        def _fno_step(x, p):
            with torch.no_grad():
                try:
                    return model(x, p)
                except TypeError:
                    return model(x)

        print("\n[bench] FNO (GPU, batch=1) single-step ...")
        x1 = torch.randn(1, 2, args.nx, device=device)

        def run_fno_1():
            _fno_step(x1, param_vec)

        mean_step, std_step = _time_fn(run_fno_1, n_warmup=args.n_warmup,
                                       n_runs=args.n_runs,
                                       cuda_sync=(device == "cuda"))
        def run_fno_1_rollout():
            x = x1
            for _ in range(args.n_steps):
                x = _fno_step(x, param_vec)

        mean_roll, std_roll = _time_fn(run_fno_1_rollout, n_warmup=3,
                                        n_runs=min(args.n_runs, 50),
                                        cuda_sync=(device == "cuda"))
        results["fno_gpu_b1"] = {
            "per_step_ms": mean_step, "per_step_std": std_step,
            "rollout_ms": mean_roll, "rollout_std": std_roll,
        }
        print(f"  single-step: {mean_step:.4f} +/- {std_step:.4f} ms")
        print(f"  50-step rollout: {mean_roll:.3f} +/- {std_roll:.3f} ms")

        print("\n[bench] FNO (GPU, batch=32) ...")
        x32 = torch.randn(B, 2, args.nx, device=device)
        param_vec_b = param_vec.expand(B, -1).contiguous()

        def run_fno_32():
            _fno_step(x32, param_vec_b)

        mean_step, std_step = _time_fn(run_fno_32, n_warmup=args.n_warmup,
                                       n_runs=args.n_runs,
                                       cuda_sync=(device == "cuda"))

        def run_fno_32_rollout():
            x = x32
            for _ in range(args.n_steps):
                x = _fno_step(x, param_vec_b)

        mean_roll, std_roll = _time_fn(run_fno_32_rollout, n_warmup=3,
                                        n_runs=min(args.n_runs, 50),
                                        cuda_sync=(device == "cuda"))
        results["fno_gpu_b32"] = {
            "per_step_ms": mean_step, "per_step_std": std_step,
            "per_step_per_traj_ms": mean_step / B,
            "rollout_ms": mean_roll, "rollout_std": std_roll,
            "rollout_per_traj_ms": mean_roll / B,
        }
        print(f"  single-step (total): {mean_step:.4f} ms, per-traj: {mean_step/B:.5f} ms")
        print(f"  50-step rollout (total): {mean_roll:.3f} ms, per-traj: {mean_roll/B:.4f} ms")
    else:
        print("\n[bench] --checkpoint not provided, skipping FNO benchmarks.")

    print("\n" + "=" * 78)
    print(" SUMMARY  (all times in ms, 50-step rollout unless noted)")
    print("=" * 78)
    print(f"{'Method':<28} {'per-step':>12} {'total':>12} {'per-traj':>12}")
    print("-" * 78)
    r = results["fd_cpu"]
    print(f"{'FD (CPU, NumPy)':<28} {r['per_step_ms']:>12.3f} {r['total_ms']:>12.2f} {r['total_ms']:>12.2f}")
    r = results["fd_gpu_b1"]
    print(f"{'FD (GPU, batch=1)':<28} {r['per_step_ms']:>12.4f} {r['total_ms']:>12.3f} {r['total_ms']:>12.3f}")
    r = results["fd_gpu_b32"]
    print(f"{'FD (GPU, batch=32)':<28} {r['per_step_per_traj_ms']:>12.5f} {r['total_ms']:>12.3f} {r['per_traj_total_ms']:>12.4f}")

    if "fno_gpu_b1" in results:
        r = results["fno_gpu_b1"]
        print(f"{'FNO (GPU, batch=1)':<28} {r['per_step_ms']:>12.4f} {r['rollout_ms']:>12.3f} {r['rollout_ms']:>12.3f}")
    if "fno_gpu_b32" in results:
        r = results["fno_gpu_b32"]
        print(f"{'FNO (GPU, batch=32)':<28} {r['per_step_per_traj_ms']:>12.5f} {r['rollout_ms']:>12.3f} {r['rollout_per_traj_ms']:>12.4f}")

    print("=" * 78)

    if "fno_gpu_b1" in results:
        s_cpu = results["fd_cpu"]["total_ms"] / results["fno_gpu_b1"]["rollout_ms"]
        s_gpu = results["fd_gpu_b1"]["total_ms"] / results["fno_gpu_b1"]["rollout_ms"]
        print(f"\n Speedup, FNO(GPU,b=1) over FD(CPU): {s_cpu:.1f}x  (mixes HW + algorithm)")
        print(f" Speedup, FNO(GPU,b=1) over FD(GPU,b=1): {s_gpu:.2f}x  (algorithmic only, fair HW)")
    if "fno_gpu_b32" in results:
        s_gpu_b = results["fd_gpu_b32"]["per_traj_total_ms"] / results["fno_gpu_b32"]["rollout_per_traj_ms"]
        print(f" Speedup, FNO(GPU,b=32) over FD(GPU,b=32) per-traj: {s_gpu_b:.2f}x")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "config": {
                        "nx": args.nx,
                        "n_steps": args.n_steps,
                        "dt": args.dt,
                        "device": device,
                        "n_warmup": args.n_warmup,
                        "n_runs": args.n_runs,
                    },
                    "results": results,
                },
                f, indent=2,
            )
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to trained FNO checkpoint (.pt). If omitted, only FD is benchmarked.")
    p.add_argument("--data", type=str, default="data/fhn_1d_128.h5",
                   help="Path to dataset (used to read config). Not strictly required.")
    p.add_argument("--nx", type=int, default=256)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--n-warmup", type=int, default=10)
    p.add_argument("--n-runs", type=int, default=100)
    p.add_argument("--validate", action="store_true", default=True,
                   help="Run FD(GPU) vs FD(CPU) correctness check before timings.")
    p.add_argument("--no-validate", dest="validate", action="store_false")
    p.add_argument("--output", type=str, default="outputs/efficiency_benchmark.json")
    args = p.parse_args()

    benchmark(args)
