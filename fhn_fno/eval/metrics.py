import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
import time
import sys

sys.path.append(".")

from fhn_fno.models.fno import FNO
from fhn_fno.data.dataset import FHNOperatorDataset


def relative_l2_error(
    pred: torch.Tensor, target: torch.Tensor, dim: Optional[Tuple[int, ...]] = None
) -> torch.Tensor:
    if dim is None:
        dim = tuple(range(1, len(pred.shape)))

    diff_norm = torch.norm(pred - target, p=2, dim=dim)
    target_norm = torch.norm(target, p=2, dim=dim)

    rel_error = diff_norm / (target_norm + 1e-8)

    return rel_error.mean()


def mse_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def compute_memory_usage() -> dict:
    mem_stats = {}

    if torch.cuda.is_available():
        mem_stats["allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
        mem_stats["reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
    else:
        import psutil

        process = psutil.Process()
        mem_stats["rss_mb"] = process.memory_info().rss / 1024**2

    return mem_stats


def benchmark_inference(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cpu",
    n_runs: int = 100,
) -> dict:
    model.eval()
    model = model.to(device)

    x = torch.randn(*input_shape, device=device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        x = torch.randn(*input_shape, device=device)

        if device == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)

        if device == "cuda":
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)

    return {
        "mean_ms": np.mean(times) * 1000,
        "std_ms": np.std(times) * 1000,
        "min_ms": np.min(times) * 1000,
        "max_ms": np.max(times) * 1000,
        "median_ms": np.median(times) * 1000,
    }


def compute_pde_residual(
    u: torch.Tensor, v: torch.Tensor, params: dict, dx: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:

    if len(u.shape) == 2:

        u_xx = (torch.roll(u, 1, dims=-1) - 2 * u + torch.roll(u, -1, dims=-1)) / (
            dx**2
        )
        v_xx = (torch.roll(v, 1, dims=-1) - 2 * v + torch.roll(v, -1, dims=-1)) / (
            dx**2
        )
    else:
        u_xx = (torch.roll(u, 1, dims=-2) - 2 * u + torch.roll(u, -1, dims=-2)) / (
            dx**2
        )
        u_yy = (torch.roll(u, 1, dims=-1) - 2 * u + torch.roll(u, -1, dims=-1)) / (
            dx**2
        )
        u_lap = u_xx + u_yy

        v_xx = (torch.roll(v, 1, dims=-2) - 2 * v + torch.roll(v, -1, dims=-2)) / (
            dx**2
        )
        v_yy = (torch.roll(v, 1, dims=-1) - 2 * v + torch.roll(v, -1, dims=-1)) / (
            dx**2
        )
        v_lap = v_xx + v_yy

    u_reaction = u - u**3 / 3 - v
    v_reaction = (u + params.get("a", 0.0) - params.get("b", 0.2) * v) / params.get(
        "tau", 5.0
    )

    if len(u.shape) == 2:
        u_residual = params.get("Du", 0.05) * u_xx + u_reaction
        v_residual = params.get("Dv", 0.01) * v_xx + v_reaction
    else:
        u_residual = params.get("Du", 0.05) * u_lap + u_reaction
        v_residual = params.get("Dv", 0.01) * v_lap + v_reaction

    return u_residual, v_residual


def evaluate_model_performance():

    DATA_FILE = "data/fhn_1d_tiny.h5"
    CHECKPOINT_FILE = "checkpoints/best_model.pt"
    DEVICE = "cpu"
    N_SAMPLES = 10
    INPUT_SHAPE = (4, 2, 256)
    N_BENCHMARK_RUNS = 50

    print("Loading model and data...")

    dataset = FHNOperatorDataset(DATA_FILE, mode="single_step", train=False)

    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    config = checkpoint.get("config", {})

    model = FNO(
        modes=config.get("model", {}).get("modes", 16),
        width=config.get("model", {}).get("width", 64),
        n_layers=config.get("model", {}).get("n_layers", 4),
        dim=dataset.dim,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    total_rel_error_u = 0.0
    total_rel_error_v = 0.0
    total_mse = 0.0

    print(f"Evaluating on {min(N_SAMPLES, len(dataset))} samples...")

    with torch.no_grad():
        for i in range(min(N_SAMPLES, len(dataset))):
            sample = dataset[i]
            x = sample["input"].unsqueeze(0).to(DEVICE)
            y = sample["target"].unsqueeze(0).to(DEVICE)

            pred = model(x)

            rel_err_u = relative_l2_error(pred[:, 0:1], y[:, 0:1])
            rel_err_v = relative_l2_error(pred[:, 1:2], y[:, 1:2])
            mse = mse_error(pred, y)

            total_rel_error_u += rel_err_u.item()
            total_rel_error_v += rel_err_v.item()
            total_mse += mse.item()

    avg_rel_error_u = total_rel_error_u / min(N_SAMPLES, len(dataset))
    avg_rel_error_v = total_rel_error_v / min(N_SAMPLES, len(dataset))
    avg_mse = total_mse / min(N_SAMPLES, len(dataset))

    print(f"\nPerformance Metrics:")
    print(f"Average Relative L2 Error (u): {avg_rel_error_u:.6f}")
    print(f"Average Relative L2 Error (v): {avg_rel_error_v:.6f}")
    print(f"Average MSE: {avg_mse:.6f}")

    print(f"\nBenchmarking inference time ({N_BENCHMARK_RUNS} runs)...")
    timing_stats = benchmark_inference(model, INPUT_SHAPE, DEVICE, N_BENCHMARK_RUNS)

    print(f"Inference Time Statistics:")
    print(f"  Mean: {timing_stats['mean_ms']:.2f} ms")
    print(f"  Std:  {timing_stats['std_ms']:.2f} ms")
    print(f"  Min:  {timing_stats['min_ms']:.2f} ms")
    print(f"  Max:  {timing_stats['max_ms']:.2f} ms")
    print(f"  Median: {timing_stats['median_ms']:.2f} ms")

    mem_stats = compute_memory_usage()
    print(f"\nMemory Usage:")
    for key, value in mem_stats.items():
        print(f"  {key}: {value:.2f} MB")

    print("\nEvaluation complete!")


def main():
    evaluate_model_performance()


if __name__ == "__main__":
    main()
