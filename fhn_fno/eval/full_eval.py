import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import h5py
from scipy import stats

sys.path.append(".")

from fhn_fno.models.fno import FNO
from fhn_fno.data.dataset import FHNOperatorDataset


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["figure.titlesize"] = 12


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    rel_l2 = np.linalg.norm(pred - target) / (np.linalg.norm(target) + 1e-8)
    mse = np.mean((pred - target) ** 2)
    mae = np.mean(np.abs(pred - target))
    max_ae = np.max(np.abs(pred - target))

    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    corr, pval = stats.pearsonr(pred.flatten(), target.flatten())

    return {
        "rel_l2": rel_l2,
        "mse": mse,
        "mae": mae,
        "max_ae": max_ae,
        "r2": r2,
        "pearson_r": corr,
        "pearson_pval": pval,
    }


def evaluate_single_step(
    model, dataset, device="cpu", n_samples=500, output_dir="outputs"
):
    print("\n" + "=" * 70)
    print(" SINGLE-STEP EVALUATION (t → t+1)")
    print("=" * 70)
    print(f"Evaluating {n_samples} single-step predictions...\n")

    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_u_pred, all_v_pred = [], []
    all_u_true, all_v_true = [], []

    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            sample = dataset[i]
            x = sample["input"].unsqueeze(0).to(device)
            y = sample["target"].unsqueeze(0).to(device)

            pred = model(x)

            if dataset.normalize:
                pred_u = pred[0, 0].cpu().numpy() * dataset.u_std + dataset.u_mean
                pred_v = pred[0, 1].cpu().numpy() * dataset.v_std + dataset.v_mean
                true_u = y[0, 0].cpu().numpy() * dataset.u_std + dataset.u_mean
                true_v = y[0, 1].cpu().numpy() * dataset.v_std + dataset.v_mean
            else:
                pred_u = pred[0, 0].cpu().numpy()
                pred_v = pred[0, 1].cpu().numpy()
                true_u = y[0, 0].cpu().numpy()
                true_v = y[0, 1].cpu().numpy()

            all_u_pred.append(pred_u)
            all_v_pred.append(pred_v)
            all_u_true.append(true_u)
            all_v_true.append(true_v)

    all_u_pred = np.array(all_u_pred)
    all_v_pred = np.array(all_v_pred)
    all_u_true = np.array(all_u_true)
    all_v_true = np.array(all_v_true)

    u_metrics = compute_metrics(all_u_pred, all_u_true)
    v_metrics = compute_metrics(all_v_pred, all_v_true)

    print("Metrics for u (activator):")
    print(f"  Relative L2 Error:      {u_metrics['rel_l2']:.8f}")
    print(f"  Mean Squared Error:     {u_metrics['mse']:.8e}")
    print(f"  Mean Absolute Error:    {u_metrics['mae']:.8e}")
    print(f"  Max Absolute Error:     {u_metrics['max_ae']:.8e}")
    print(f"  R² Score:               {u_metrics['r2']:.8f}")
    print(f"  Pearson Correlation:    {u_metrics['pearson_r']:.8f}")

    print("\nMetrics for v (inhibitor):")
    print(f"  Relative L2 Error:      {v_metrics['rel_l2']:.8f}")
    print(f"  Mean Squared Error:     {v_metrics['mse']:.8e}")
    print(f"  Mean Absolute Error:    {v_metrics['mae']:.8e}")
    print(f"  Max Absolute Error:     {v_metrics['max_ae']:.8e}")
    print(f"  R² Score:               {v_metrics['r2']:.8f}")
    print(f"  Pearson Correlation:    {v_metrics['pearson_r']:.8f}")

    print("\nGenerating visualizations...")
    nx = all_u_true.shape[-1]
    x_coords = np.linspace(0, 1, nx)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    example_indices = [0, n_samples // 4, n_samples // 2, 3 * n_samples // 4]

    for i, idx in enumerate(example_indices):

        axes[0, i].plot(
            x_coords, all_u_true[idx], "b-", linewidth=2, label="Ground Truth"
        )
        axes[0, i].plot(
            x_coords, all_u_pred[idx], "r--", linewidth=2, label="Prediction"
        )
        axes[0, i].fill_between(x_coords, all_u_true[idx], all_u_pred[idx], alpha=0.3)
        axes[0, i].set_xlabel("x")
        axes[0, i].set_ylabel("u")
        axes[0, i].set_title(
            f"Sample {idx} - u (MAE: {np.abs(all_u_true[idx] - all_u_pred[idx]).mean():.4e})"
        )
        axes[0, i].legend(fontsize=8)
        axes[0, i].grid(True, alpha=0.3)

        axes[1, i].plot(
            x_coords, all_v_true[idx], "b-", linewidth=2, label="Ground Truth"
        )
        axes[1, i].plot(
            x_coords, all_v_pred[idx], "r--", linewidth=2, label="Prediction"
        )
        axes[1, i].fill_between(x_coords, all_v_true[idx], all_v_pred[idx], alpha=0.3)
        axes[1, i].set_xlabel("x")
        axes[1, i].set_ylabel("v")
        axes[1, i].set_title(
            f"Sample {idx} - v (MAE: {np.abs(all_v_true[idx] - all_v_pred[idx]).mean():.4e})"
        )
        axes[1, i].legend(fontsize=8)
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "single_step_examples.png", dpi=300, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sample_size = min(5000, all_u_true.size)
    sample_indices = np.random.choice(all_u_true.size, size=sample_size, replace=False)

    axes[0].scatter(
        all_u_true.flatten()[sample_indices],
        all_u_pred.flatten()[sample_indices],
        alpha=0.3,
        s=2,
        c="blue",
    )
    u_range = [all_u_true.min(), all_u_true.max()]
    axes[0].plot(u_range, u_range, "k--", linewidth=2, label="Perfect")
    axes[0].set_xlabel("True u", fontweight="bold")
    axes[0].set_ylabel("Predicted u", fontweight="bold")
    axes[0].set_title(f'u Predictions (R²={u_metrics["r2"]:.6f})', fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect("equal")

    axes[1].scatter(
        all_v_true.flatten()[sample_indices],
        all_v_pred.flatten()[sample_indices],
        alpha=0.3,
        s=2,
        c="orange",
    )
    v_range = [all_v_true.min(), all_v_true.max()]
    axes[1].plot(v_range, v_range, "k--", linewidth=2, label="Perfect")
    axes[1].set_xlabel("True v", fontweight="bold")
    axes[1].set_ylabel("Predicted v", fontweight="bold")
    axes[1].set_title(f'v Predictions (R²={v_metrics["r2"]:.6f})', fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_dir / "single_step_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()

    with open(output_dir / "single_step_metrics.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write(" SINGLE-STEP EVALUATION METRICS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Number of samples: {n_samples}\n")
        f.write(f"Spatial resolution: {nx}\n\n")

        f.write("--- u (Activator) ---\n")
        for key, val in u_metrics.items():
            f.write(f"  {key:.<30} {val:.8e}\n")

        f.write("\n--- v (Inhibitor) ---\n")
        for key, val in v_metrics.items():
            f.write(f"  {key:.<30} {val:.8e}\n")

    print(f"\nSingle-step results saved to {output_dir}/")
    return {"u": u_metrics, "v": v_metrics}


def evaluate_rollout(
    model, dataset, device="cpu", sample_idx=0, n_steps=50, output_dir="outputs"
):
    print("\n" + "=" * 70)
    print(" ROLLOUT EVALUATION (Multi-step prediction)")
    print("=" * 70)
    print(f"Sample: {sample_idx}, Steps: {n_steps}\n")

    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset.data_file, "r") as f:
        u_traj_true = f["u_traj"][sample_idx][: n_steps + 1]
        v_traj_true = f["v_traj"][sample_idx][: n_steps + 1]

    u0 = u_traj_true[0]
    v0 = v_traj_true[0]

    if dataset.normalize:
        u0_norm = (u0 - dataset.u_mean) / dataset.u_std
        v0_norm = (v0 - dataset.v_mean) / dataset.v_std
    else:
        u0_norm = u0
        v0_norm = v0

    x0 = (
        torch.tensor(np.stack([u0_norm, v0_norm], axis=0), dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        trajectory = model.rollout(x0, n_steps)

    u_traj_pred = trajectory[0, :, 0].cpu().numpy()
    v_traj_pred = trajectory[0, :, 1].cpu().numpy()

    if dataset.normalize:
        u_traj_pred = u_traj_pred * dataset.u_std + dataset.u_mean
        v_traj_pred = v_traj_pred * dataset.v_std + dataset.v_mean

    u_metrics = compute_metrics(u_traj_pred, u_traj_true)
    v_metrics = compute_metrics(v_traj_pred, v_traj_true)

    print("Rollout metrics for u:")
    print(f"  Relative L2 Error:      {u_metrics['rel_l2']:.6f}")
    print(f"  Mean Squared Error:     {u_metrics['mse']:.6e}")
    print(f"  Mean Absolute Error:    {u_metrics['mae']:.6e}")
    print(f"  Max Absolute Error:     {u_metrics['max_ae']:.6e}")
    print(f"  R² Score:               {u_metrics['r2']:.6f}")

    print("\nRollout metrics for v:")
    print(f"  Relative L2 Error:      {v_metrics['rel_l2']:.6f}")
    print(f"  Mean Squared Error:     {v_metrics['mse']:.6e}")
    print(f"  Mean Absolute Error:    {v_metrics['mae']:.6e}")
    print(f"  Max Absolute Error:     {v_metrics['max_ae']:.6e}")
    print(f"  R² Score:               {v_metrics['r2']:.6f}")

    print("\nGenerating visualizations...")
    nx = u_traj_true.shape[-1]
    x = np.linspace(0, 1, nx)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    vmax_u = max(np.abs(u_traj_true).max(), np.abs(u_traj_pred).max())
    vmax_v = max(np.abs(v_traj_true).max(), np.abs(v_traj_pred).max())

    im = axes[0, 0].imshow(
        u_traj_true.T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax_u,
        vmax=vmax_u,
        origin="lower",
    )
    axes[0, 0].set_title("u: Ground Truth", fontweight="bold")
    axes[0, 0].set_xlabel("Time step")
    axes[0, 0].set_ylabel("Spatial position")
    plt.colorbar(im, ax=axes[0, 0])

    im = axes[0, 1].imshow(
        v_traj_true.T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax_v,
        vmax=vmax_v,
        origin="lower",
    )
    axes[0, 1].set_title("v: Ground Truth", fontweight="bold")
    axes[0, 1].set_xlabel("Time step")
    axes[0, 1].set_ylabel("Spatial position")
    plt.colorbar(im, ax=axes[0, 1])

    im = axes[1, 0].imshow(
        u_traj_pred.T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax_u,
        vmax=vmax_u,
        origin="lower",
    )
    axes[1, 0].set_title(
        f'u: Prediction (Rel L2: {u_metrics["rel_l2"]:.4f})', fontweight="bold"
    )
    axes[1, 0].set_xlabel("Time step")
    axes[1, 0].set_ylabel("Spatial position")
    plt.colorbar(im, ax=axes[1, 0])

    im = axes[1, 1].imshow(
        v_traj_pred.T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-vmax_v,
        vmax=vmax_v,
        origin="lower",
    )
    axes[1, 1].set_title(
        f'v: Prediction (Rel L2: {v_metrics["rel_l2"]:.4f})', fontweight="bold"
    )
    axes[1, 1].set_xlabel("Time step")
    axes[1, 1].set_ylabel("Spatial position")
    plt.colorbar(im, ax=axes[1, 1])

    u_error = np.abs(u_traj_true - u_traj_pred)
    im = axes[2, 0].imshow(u_error.T, aspect="auto", cmap="hot", origin="lower")
    axes[2, 0].set_title(
        f"u: Absolute Error (Max: {u_error.max():.3e})", fontweight="bold"
    )
    axes[2, 0].set_xlabel("Time step")
    axes[2, 0].set_ylabel("Spatial position")
    plt.colorbar(im, ax=axes[2, 0])

    v_error = np.abs(v_traj_true - v_traj_pred)
    im = axes[2, 1].imshow(v_error.T, aspect="auto", cmap="hot", origin="lower")
    axes[2, 1].set_title(
        f"v: Absolute Error (Max: {v_error.max():.3e})", fontweight="bold"
    )
    axes[2, 1].set_xlabel("Time step")
    axes[2, 1].set_ylabel("Spatial position")
    plt.colorbar(im, ax=axes[2, 1])

    plt.tight_layout()
    plt.savefig(output_dir / "rollout_spatiotemporal.png", dpi=300, bbox_inches="tight")
    plt.close()

    time_indices = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps]
    fig, axes = plt.subplots(2, len(time_indices), figsize=(4 * len(time_indices), 7))

    for i, t_idx in enumerate(time_indices):

        axes[0, i].plot(x, u_traj_true[t_idx], "b-", linewidth=2, label="True")
        axes[0, i].plot(x, u_traj_pred[t_idx], "r--", linewidth=2, label="Pred")
        axes[0, i].fill_between(x, u_traj_true[t_idx], u_traj_pred[t_idx], alpha=0.3)
        axes[0, i].set_xlabel("x")
        axes[0, i].set_ylabel("u")
        axes[0, i].set_title(f"t={t_idx}")
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)

        axes[1, i].plot(x, v_traj_true[t_idx], "b-", linewidth=2, label="True")
        axes[1, i].plot(x, v_traj_pred[t_idx], "r--", linewidth=2, label="Pred")
        axes[1, i].fill_between(x, v_traj_true[t_idx], v_traj_pred[t_idx], alpha=0.3)
        axes[1, i].set_xlabel("x")
        axes[1, i].set_ylabel("v")
        axes[1, i].set_title(f"t={t_idx}")
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "rollout_snapshots.png", dpi=300, bbox_inches="tight")
    plt.close()

    u_errors_time = [
        compute_metrics(u_traj_pred[t : t + 1], u_traj_true[t : t + 1])["rel_l2"]
        for t in range(n_steps + 1)
    ]
    v_errors_time = [
        compute_metrics(v_traj_pred[t : t + 1], v_traj_true[t : t + 1])["rel_l2"]
        for t in range(n_steps + 1)
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        range(n_steps + 1),
        u_errors_time,
        "b-",
        linewidth=2,
        marker="o",
        markersize=3,
        label="u",
    )
    ax.plot(
        range(n_steps + 1),
        v_errors_time,
        "orange",
        linewidth=2,
        marker="s",
        markersize=3,
        label="v",
    )
    ax.set_xlabel("Time step", fontweight="bold")
    ax.set_ylabel("Relative L2 Error", fontweight="bold")
    ax.set_title("Error Evolution During Rollout", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        output_dir / "rollout_error_evolution.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    with open(output_dir / "rollout_metrics.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write(" ROLLOUT EVALUATION METRICS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Sample index: {sample_idx}\n")
        f.write(f"Number of steps: {n_steps}\n")
        f.write(f"Spatial resolution: {nx}\n\n")

        f.write("--- u (Activator) ---\n")
        for key, val in u_metrics.items():
            f.write(f"  {key:.<30} {val:.6e}\n")

        f.write("\n--- v (Inhibitor) ---\n")
        for key, val in v_metrics.items():
            f.write(f"  {key:.<30} {val:.6e}\n")

    print(f"\nRollout results saved to {output_dir}/")
    return {"u": u_metrics, "v": v_metrics}


def evaluate_parameter_generalization(
    model, dataset, device="cpu", n_samples=500, output_dir="outputs"
):
    print("\n" + "=" * 70)
    print(" PARAMETER GENERALIZATION ANALYSIS (Single-Step)")
    print("=" * 70)
    print(f"Analyzing {n_samples} samples across parameter space...\n")

    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_params = []
    u_errors = []
    v_errors = []

    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            sample = dataset[i]
            x = sample["input"].unsqueeze(0).to(device)
            y = sample["target"].unsqueeze(0).to(device)
            params = sample["params"].cpu().numpy()

            pred = model(x)

            if dataset.normalize:
                pred_u = pred[0, 0].cpu().numpy() * dataset.u_std + dataset.u_mean
                pred_v = pred[0, 1].cpu().numpy() * dataset.v_std + dataset.v_mean
                true_u = y[0, 0].cpu().numpy() * dataset.u_std + dataset.u_mean
                true_v = y[0, 1].cpu().numpy() * dataset.v_std + dataset.v_mean
            else:
                pred_u = pred[0, 0].cpu().numpy()
                pred_v = pred[0, 1].cpu().numpy()
                true_u = y[0, 0].cpu().numpy()
                true_v = y[0, 1].cpu().numpy()

            u_err = compute_metrics(pred_u[np.newaxis, :], true_u[np.newaxis, :])[
                "rel_l2"
            ]
            v_err = compute_metrics(pred_v[np.newaxis, :], true_v[np.newaxis, :])[
                "rel_l2"
            ]

            all_params.append(params)
            u_errors.append(u_err)
            v_errors.append(v_err)

    all_params = np.array(all_params)
    u_errors = np.array(u_errors)
    v_errors = np.array(v_errors)

    param_names = ["Du", "Dv", "a", "b", "tau"]

    print("Parameter Ranges in Dataset:")
    for i, name in enumerate(param_names):
        print(
            f"  {name:5s}: [{all_params[:, i].min():.4f}, {all_params[:, i].max():.4f}]"
        )

    print(f"\nError Statistics:")
    print(f"  u mean error: {u_errors.mean():.6f} ± {u_errors.std():.6f}")
    print(f"  v mean error: {v_errors.mean():.6f} ± {v_errors.std():.6f}")

    print("\nParameter-Error Correlations:")
    for i, name in enumerate(param_names):
        u_corr = np.corrcoef(all_params[:, i], u_errors)[0, 1]
        v_corr = np.corrcoef(all_params[:, i], v_errors)[0, 1]
        print(f"  {name:5s}: u={u_corr:+.4f}, v={v_corr:+.4f}")

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    for i, name in enumerate(param_names):

        ax = axes[0, i]
        scatter = ax.scatter(
            all_params[:, i], u_errors, alpha=0.4, s=20, c=u_errors, cmap="viridis"
        )

        z = np.polyfit(all_params[:, i], u_errors, 1)
        p = np.poly1d(z)
        x_line = np.linspace(all_params[:, i].min(), all_params[:, i].max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label=f"Trend")

        corr = np.corrcoef(all_params[:, i], u_errors)[0, 1]
        ax.set_xlabel(f"{name}", fontweight="bold", fontsize=11)
        ax.set_ylabel("u Rel. L2 Error", fontweight="bold")
        ax.set_title(f"{name} vs u Error (ρ={corr:+.3f})", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.colorbar(scatter, ax=ax, label="Error")

        ax = axes[1, i]
        scatter = ax.scatter(
            all_params[:, i], v_errors, alpha=0.4, s=20, c=v_errors, cmap="plasma"
        )

        z = np.polyfit(all_params[:, i], v_errors, 1)
        p = np.poly1d(z)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label=f"Trend")

        corr = np.corrcoef(all_params[:, i], v_errors)[0, 1]
        ax.set_xlabel(f"{name}", fontweight="bold", fontsize=11)
        ax.set_ylabel("v Rel. L2 Error", fontweight="bold")
        ax.set_title(f"{name} vs v Error (ρ={corr:+.3f})", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.colorbar(scatter, ax=ax, label="Error")

    plt.suptitle(
        "Parameter Generalization: Single-Step Prediction Errors",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "parameter_generalization.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 5))

    correlations = np.zeros((2, 5))
    for i in range(5):
        correlations[0, i] = np.corrcoef(all_params[:, i], u_errors)[0, 1]
        correlations[1, i] = np.corrcoef(all_params[:, i], v_errors)[0, 1]

    im = ax.imshow(correlations, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(5))
    ax.set_xticklabels(param_names, fontweight="bold")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["u error", "v error"], fontweight="bold")
    ax.set_title("Parameter-Error Correlation Matrix", fontweight="bold", fontsize=13)

    for i in range(2):
        for j in range(5):
            text = ax.text(
                j,
                i,
                f"{correlations[i, j]:+.3f}",
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
            )

    plt.colorbar(im, ax=ax, label="Correlation")
    plt.tight_layout()
    plt.savefig(
        output_dir / "parameter_correlation_matrix.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    with open(output_dir / "parameter_generalization.txt", "w") as f:
        f.write("=" * 70 + "\n")
        f.write(" PARAMETER GENERALIZATION ANALYSIS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Number of samples: {n_samples}\n")
        f.write(f"Evaluation mode: Single-step prediction\n\n")

        f.write("Parameter Ranges:\n")
        for i, name in enumerate(param_names):
            f.write(
                f"  {name:5s}: [{all_params[:, i].min():.6f}, {all_params[:, i].max():.6f}]\n"
            )

        f.write("\nError Statistics:\n")
        f.write(f"  u mean: {u_errors.mean():.8e} ± {u_errors.std():.8e}\n")
        f.write(f"  u min:  {u_errors.min():.8e}\n")
        f.write(f"  u max:  {u_errors.max():.8e}\n")
        f.write(f"  v mean: {v_errors.mean():.8e} ± {v_errors.std():.8e}\n")
        f.write(f"  v min:  {v_errors.min():.8e}\n")
        f.write(f"  v max:  {v_errors.max():.8e}\n")

        f.write("\nParameter-Error Correlations:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Parameter':<12} {'u corr':>12} {'v corr':>12} {'Interpretation'}\n")
        f.write("-" * 70 + "\n")

        for i, name in enumerate(param_names):
            u_corr = np.corrcoef(all_params[:, i], u_errors)[0, 1]
            v_corr = np.corrcoef(all_params[:, i], v_errors)[0, 1]

            u_strength = (
                "strong"
                if abs(u_corr) > 0.5
                else "moderate" if abs(u_corr) > 0.3 else "weak"
            )
            v_strength = (
                "strong"
                if abs(v_corr) > 0.5
                else "moderate" if abs(v_corr) > 0.3 else "weak"
            )
            direction = "positive" if u_corr > 0 else "negative"

            f.write(
                f"{name:<12} {u_corr:>+12.4f} {v_corr:>+12.4f} "
                f"   {u_strength} {direction}\n"
            )

        f.write("\nInterpretation Guide:\n")
        f.write("  Positive correlation: Error increases as parameter increases\n")
        f.write("  Negative correlation: Error decreases as parameter increases\n")
        f.write("  |ρ| > 0.5: Strong relationship\n")
        f.write("  |ρ| > 0.3: Moderate relationship\n")
        f.write("  |ρ| < 0.3: Weak relationship\n")

        f.write("\nConclusion:\n")
        f.write(
            f"  Model shows {'consistent' if u_errors.std() / u_errors.mean() < 0.5 else 'variable'} "
            f"performance across parameter space\n"
        )
        f.write(
            f"  Coefficient of variation (u): {u_errors.std() / u_errors.mean():.3f}\n"
        )
        f.write(
            f"  Coefficient of variation (v): {v_errors.std() / v_errors.mean():.3f}\n"
        )

    print(f"\nParameter generalization analysis saved to {output_dir}/")
    print("Generated files:")
    print("  - parameter_generalization.png (scatter plots)")
    print("  - parameter_correlation_matrix.png (heatmap)")
    print("  - parameter_generalization.txt (detailed statistics)")

    return {
        "params": all_params,
        "u_errors": u_errors,
        "v_errors": v_errors,
        "param_names": param_names,
    }


def main():

    DATA_FILE = "data/fhn_1d_8000.h5"
    CHECKPOINT_FILE = "checkpoints/best_model.pt"
    DEVICE = "cpu"
    OUTPUT_DIR = "eval_results/"

    RUN_SINGLE_STEP = True
    RUN_ROLLOUT = True
    RUN_PARAM_GENERALIZATION = True

    N_SINGLE_STEP_SAMPLES = 500

    ROLLOUT_SAMPLE_IDX = 0
    N_ROLLOUT_STEPS = 50

    N_PARAM_SAMPLES = 500

    print("\n" + "=" * 70)
    print(" FHN-FNO EVALUATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Data: {DATA_FILE}")
    print(f"  Checkpoint: {CHECKPOINT_FILE}")
    print(f"  Device: {DEVICE}")
    print(f"  Output: {OUTPUT_DIR}")

    print("\nLoading dataset...")
    dataset = FHNOperatorDataset(DATA_FILE, mode="single_step", train=False)
    dataset.data_file = DATA_FILE
    print(f"  Dataset size: {len(dataset)} samples")

    print("Loading model...")
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
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if RUN_SINGLE_STEP:
        single_step_metrics = evaluate_single_step(
            model, dataset, DEVICE, N_SINGLE_STEP_SAMPLES, OUTPUT_DIR
        )

    if RUN_ROLLOUT:
        rollout_metrics = evaluate_rollout(
            model, dataset, DEVICE, ROLLOUT_SAMPLE_IDX, N_ROLLOUT_STEPS, OUTPUT_DIR
        )

    if RUN_PARAM_GENERALIZATION:
        param_results = evaluate_parameter_generalization(
            model, dataset, DEVICE, N_PARAM_SAMPLES, OUTPUT_DIR
        )

    print("\n" + "=" * 70)
    print(" EVALUATION COMPLETE!")
    print("=" * 70)
    print(f"\nAll results saved to: {OUTPUT_DIR}/")

    if RUN_SINGLE_STEP:
        print("\nSingle-step files:")
        print("  - single_step_examples.png")
        print("  - single_step_scatter.png")
        print("  - single_step_metrics.txt")

    if RUN_ROLLOUT:
        print("\nRollout files:")
        print("  - rollout_spatiotemporal.png")
        print("  - rollout_snapshots.png")
        print("  - rollout_error_evolution.png")
        print("  - rollout_metrics.txt")

    if RUN_PARAM_GENERALIZATION:
        print("\nParameter generalization files:")
        print("  - parameter_generalization.png (scatter plots)")
        print("  - parameter_correlation_matrix.png (heatmap)")
        print("  - parameter_generalization.txt (detailed statistics)")

    print("\n")


if __name__ == "__main__":
    main()
