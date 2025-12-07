import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from pathlib import Path
import sys
import h5py

sys.path.append(".")

from fhn_fno.models.fno import FNO
from fhn_fno.data.dataset import FHNOperatorDataset
from fhn_fno.eval.metrics import relative_l2_error


rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
rcParams["font.size"] = 10
rcParams["axes.labelsize"] = 11
rcParams["axes.titlesize"] = 12
rcParams["xtick.labelsize"] = 9
rcParams["ytick.labelsize"] = 9
rcParams["legend.fontsize"] = 9
rcParams["figure.titlesize"] = 13
rcParams["lines.linewidth"] = 1.5
rcParams["axes.linewidth"] = 1.0
rcParams["grid.linewidth"] = 0.5
rcParams["text.usetex"] = False


def plot_multi_sample_comparison(
    u_true_list,
    v_true_list,
    u_pred_list,
    v_pred_list,
    x,
    sample_indices,
    save_path=None,
):
    """
    Research-grade plot comparing ground truth vs predictions for multiple samples at t=50

    Args:
        u_true_list: List of true u solutions (each nx,)
        v_true_list: List of true v solutions (each nx,)
        u_pred_list: List of predicted u solutions (each nx,)
        v_pred_list: List of predicted v solutions (each nx,)
        x: Spatial coordinates
        sample_indices: List of sample indices
        save_path: Path to save figure
    """
    n_samples = len(sample_indices)
    fig = plt.figure(figsize=(14, 2.5 * n_samples))
    gs = gridspec.GridSpec(
        n_samples,
        4,
        hspace=0.35,
        wspace=0.35,
        left=0.08,
        right=0.98,
        top=0.95,
        bottom=0.06,
    )

    true_color = "#1f77b4"
    pred_color = "#ff7f0e"
    error_color = "#d62728"

    for idx, sample_idx in enumerate(sample_indices):
        u_true = u_true_list[idx]
        v_true = v_true_list[idx]
        u_pred = u_pred_list[idx]
        v_pred = v_pred_list[idx]

        u_error = np.abs(u_true - u_pred)
        v_error = np.abs(v_true - v_pred)
        u_rel_err = np.linalg.norm(u_error) / (np.linalg.norm(u_true) + 1e-8)
        v_rel_err = np.linalg.norm(v_error) / (np.linalg.norm(v_true) + 1e-8)

        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.plot(
            x, u_true, color=true_color, label="Ground Truth", linewidth=2, alpha=0.85
        )
        ax1.plot(
            x,
            u_pred,
            color=pred_color,
            label="FNO Prediction",
            linewidth=1.8,
            alpha=0.85,
            linestyle="--",
        )
        ax1.set_xlabel("Spatial Position $x$")
        ax1.set_ylabel("$u(x, t_{50})$")
        ax1.set_title(f"Sample {sample_idx}: $u$ Field")
        ax1.legend(loc="best", framealpha=0.9)
        ax1.grid(True, alpha=0.3, linestyle=":")
        ax1.set_xlim([x.min(), x.max()])

        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.plot(
            x, v_true, color=true_color, label="Ground Truth", linewidth=2, alpha=0.85
        )
        ax2.plot(
            x,
            v_pred,
            color=pred_color,
            label="FNO Prediction",
            linewidth=1.8,
            alpha=0.85,
            linestyle="--",
        )
        ax2.set_xlabel("Spatial Position $x$")
        ax2.set_ylabel("$v(x, t_{50})$")
        ax2.set_title(f"Sample {sample_idx}: $v$ Field")
        ax2.legend(loc="best", framealpha=0.9)
        ax2.grid(True, alpha=0.3, linestyle=":")
        ax2.set_xlim([x.min(), x.max()])

        ax3 = fig.add_subplot(gs[idx, 2])
        ax3.plot(x, u_error, color=error_color, linewidth=2)
        ax3.fill_between(x, 0, u_error, color=error_color, alpha=0.3)
        ax3.set_xlabel("Spatial Position $x$")
        ax3.set_ylabel("$|u_{true} - u_{pred}|$")
        ax3.set_title(f"$u$ Pointwise Error (Rel. $L^2$: {u_rel_err:.4f})")
        ax3.grid(True, alpha=0.3, linestyle=":")
        ax3.set_xlim([x.min(), x.max()])
        ax3.set_ylim(bottom=0)

        ax4 = fig.add_subplot(gs[idx, 3])
        ax4.plot(x, v_error, color=error_color, linewidth=2)
        ax4.fill_between(x, 0, v_error, color=error_color, alpha=0.3)
        ax4.set_xlabel("Spatial Position $x$")
        ax4.set_ylabel("$|v_{true} - v_{pred}|$")
        ax4.set_title(f"$v$ Pointwise Error (Rel. $L^2$: {v_rel_err:.4f})")
        ax4.grid(True, alpha=0.3, linestyle=":")
        ax4.set_xlim([x.min(), x.max()])
        ax4.set_ylim(bottom=0)

    fig.suptitle(
        "FNO Performance: Multi-Sample Comparison at $t = 50$",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.close()


def plot_multi_sample_phase_portraits(
    u_true_list,
    v_true_list,
    u_pred_list,
    v_pred_list,
    sample_indices,
    spatial_points=None,
    save_path=None,
):
    """
    Research-grade phase portraits for multiple samples

    Args:
        u_true_list: List of true u trajectories (each n_times, nx)
        v_true_list: List of true v trajectories (each n_times, nx)
        u_pred_list: List of predicted u trajectories (each n_times, nx)
        v_pred_list: List of predicted v trajectories (each n_times, nx)
        sample_indices: List of sample indices
        spatial_points: List of spatial indices to plot (default: 4 evenly spaced)
        save_path: Path to save figure
    """
    n_samples = len(sample_indices)
    n_times, nx = u_true_list[0].shape

    if spatial_points is None:

        spatial_points = [nx // 5, 2 * nx // 5, 3 * nx // 5, 4 * nx // 5]

    n_points = len(spatial_points)

    fig = plt.figure(figsize=(3.5 * n_points, 3 * n_samples))
    gs = gridspec.GridSpec(
        n_samples,
        n_points,
        hspace=0.3,
        wspace=0.3,
        left=0.08,
        right=0.98,
        top=0.96,
        bottom=0.06,
    )

    true_color = "#1f77b4"
    pred_color = "#ff7f0e"
    start_color = "#2ca02c"
    end_true_color = "#d62728"
    end_pred_color = "#ff9896"

    for row, sample_idx in enumerate(sample_indices):
        u_true = u_true_list[row]
        v_true = v_true_list[row]
        u_pred = u_pred_list[row]
        v_pred = v_pred_list[row]

        for col, pt in enumerate(spatial_points):
            ax = fig.add_subplot(gs[row, col])

            n_plot = min(n_times, 51)
            alphas = np.linspace(0.3, 1.0, n_plot)

            for i in range(n_plot - 1):
                ax.plot(
                    u_true[i : i + 2, pt],
                    v_true[i : i + 2, pt],
                    color=true_color,
                    alpha=alphas[i],
                    linewidth=1.5,
                )
                ax.plot(
                    u_pred[i : i + 2, pt],
                    v_pred[i : i + 2, pt],
                    color=pred_color,
                    alpha=alphas[i],
                    linewidth=1.5,
                    linestyle="--",
                )

            ax.scatter(
                u_true[0, pt],
                v_true[0, pt],
                c=start_color,
                s=100,
                marker="o",
                edgecolors="black",
                linewidths=1.5,
                label="$t_0$",
                zorder=5,
            )
            ax.scatter(
                u_true[-1, pt],
                v_true[-1, pt],
                c=end_true_color,
                s=100,
                marker="s",
                edgecolors="black",
                linewidths=1.5,
                label="$t_{50}$ (True)",
                zorder=5,
            )
            ax.scatter(
                u_pred[-1, pt],
                v_pred[-1, pt],
                c=end_pred_color,
                s=100,
                marker="^",
                edgecolors="black",
                linewidths=1.5,
                label="$t_{50}$ (FNO)",
                zorder=5,
            )

            true_traj = np.stack([u_true[:, pt], v_true[:, pt]], axis=1)
            pred_traj = np.stack([u_pred[:, pt], v_pred[:, pt]], axis=1)
            traj_error = np.linalg.norm(true_traj - pred_traj) / (
                np.linalg.norm(true_traj) + 1e-8
            )

            ax.set_xlabel("$u$")
            ax.set_ylabel("$v$")

            x_frac = pt / nx
            if row == 0:
                ax.set_title(f"$x = {x_frac:.2f}$", fontweight="bold")

            ax.grid(True, alpha=0.3, linestyle=":")

            ax.text(
                0.05,
                0.95,
                f"Rel. $L^2$: {traj_error:.4f}",
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

            if col == 0:
                ax.set_ylabel(f"Sample {sample_idx}\n$v$", fontsize=10)

            if col == n_points - 1:
                ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                    framealpha=0.9,
                    fontsize=8,
                )

    fig.suptitle(
        "Phase Space Trajectories: $(u, v)$ Dynamics at Multiple Spatial Locations",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    plt.close()


def visualize_multiple_rollouts(
    model: nn.Module,
    dataset: FHNOperatorDataset,
    sample_indices: list = None,
    n_steps: int = 50,
    device: str = "cpu",
    output_dir: str = "outputs",
):
    """
    Visualize model rollouts for multiple samples with research-grade plots

    Args:
        model: Trained FNO model
        dataset: Dataset with ground truth
        sample_indices: List of sample indices to visualize (default: [0,1,2,3,4])
        n_steps: Number of rollout steps
        device: Device to use
        output_dir: Directory to save figures
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sample_indices is None:
        sample_indices = [0, 1, 2, 3, 4]

    u_true_t50_list = []
    v_true_t50_list = []
    u_pred_t50_list = []
    v_pred_t50_list = []

    u_true_traj_list = []
    v_true_traj_list = []
    u_pred_traj_list = []
    v_pred_traj_list = []

    print(f"Processing {len(sample_indices)} samples for visualization...")

    with h5py.File(dataset.data_file, "r") as f:
        for sample_idx in sample_indices:

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

            u_true_t50_list.append(u_traj_true[-1])
            v_true_t50_list.append(v_traj_true[-1])
            u_pred_t50_list.append(u_traj_pred[-1])
            v_pred_t50_list.append(v_traj_pred[-1])

            u_true_traj_list.append(u_traj_true)
            v_true_traj_list.append(v_traj_true)
            u_pred_traj_list.append(u_traj_pred)
            v_pred_traj_list.append(v_traj_pred)

            u_rel_error = relative_l2_error(
                torch.tensor(u_traj_pred), torch.tensor(u_traj_true)
            )
            v_rel_error = relative_l2_error(
                torch.tensor(v_traj_pred), torch.tensor(v_traj_true)
            )

            print(
                f"Sample {sample_idx} - Rel. L2 errors: u={u_rel_error:.6f}, v={v_rel_error:.6f}"
            )

    nx = u_true_t50_list[0].shape[-1]
    x = np.linspace(0, 1, nx)

    print("\nGenerating research-grade visualizations...")

    plot_multi_sample_comparison(
        u_true_t50_list,
        v_true_t50_list,
        u_pred_t50_list,
        v_pred_t50_list,
        x,
        sample_indices,
        save_path=output_dir / "multi_sample_t50_comparison.png",
    )

    plot_multi_sample_phase_portraits(
        u_true_traj_list,
        v_true_traj_list,
        u_pred_traj_list,
        v_pred_traj_list,
        sample_indices,
        save_path=output_dir / "multi_sample_phase_portraits.png",
    )

    all_u_errors = []
    all_v_errors = []

    for i in range(len(sample_indices)):
        u_err = relative_l2_error(
            torch.tensor(u_pred_traj_list[i]), torch.tensor(u_true_traj_list[i])
        ).item()
        v_err = relative_l2_error(
            torch.tensor(v_pred_traj_list[i]), torch.tensor(v_true_traj_list[i])
        ).item()
        all_u_errors.append(u_err)
        all_v_errors.append(v_err)

    print(f"\n{'='*60}")
    print(f"Overall Statistics across {len(sample_indices)} samples:")
    print(f"{'='*60}")
    print(
        f"u field - Mean Rel. L2: {np.mean(all_u_errors):.6f} ± {np.std(all_u_errors):.6f}"
    )
    print(
        f"v field - Mean Rel. L2: {np.mean(all_v_errors):.6f} ± {np.std(all_v_errors):.6f}"
    )
    print(f"{'='*60}")
    print(f"\nAll visualizations saved to: {output_dir.absolute()}")


def main():
    """Main function with configurable parameters"""

    DATA_FILE = "data/fhn_1d_8000.h5"
    CHECKPOINT_FILE = "checkpoints/best_model_16000.pt"
    SAMPLE_INDICES = [0, 1, 2, 3, 4]
    N_STEPS = 50
    DEVICE = "cpu"
    OUTPUT_DIR = "outputs/"

    print("=" * 60)
    print("Research-Grade FNO Visualization")
    print("=" * 60)
    print(f"Data file: {DATA_FILE}")
    print(f"Checkpoint: {CHECKPOINT_FILE}")
    print(f"Samples: {SAMPLE_INDICES}")
    print(f"Rollout steps: {N_STEPS}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

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

    dataset.data_file = DATA_FILE

    visualize_multiple_rollouts(
        model, dataset, SAMPLE_INDICES, N_STEPS, DEVICE, OUTPUT_DIR
    )


if __name__ == "__main__":
    main()
