import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from pathlib import Path
import sys
import h5py
from scipy import stats
from scipy.fft import fft, fftfreq
import seaborn as sns
import time
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm
from dataclasses import dataclass

sys.path.append(".")

from fhn_fno.models.fno import FNO
from fhn_fno.data.dataset import FHNOperatorDataset
from fhn_fno.config import Config, FHNParams, DataConfig
from fhn_fno.data.generate_fhn import FDBackend

# publication-style figures
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 11
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9
plt.rcParams["figure.titlesize"] = 13


@dataclass
class EvaluationConfig:
    data_file: str = "data/fhn_1d_8000.h5"
    checkpoint_file: str = "checkpoints/best_model.pt"
    output_dir: str = "research_outputs/"
    device: str = "cpu"

    n_test_samples: int = 100
    rollout_steps: int = 50

    test_Du_range: Tuple[float, float] = (0.005, 0.15)
    test_Dv_range: Tuple[float, float] = (0.001, 0.08)
    test_a_range: Tuple[float, float] = (-0.2, 0.2)
    test_b_range: Tuple[float, float] = (0.05, 0.7)
    test_tau_range: Tuple[float, float] = (0.5, 30.0)

    test_resolutions: List[int] = None
    test_batch_sizes: List[int] = None
    efficiency_n_runs: int = 50

    extrapolation_T_multiples: List[float] = None

    def __post_init__(self):
        if self.test_resolutions is None:
            self.test_resolutions = [64, 128, 256, 512, 1024]
        if self.test_batch_sizes is None:
            self.test_batch_sizes = [1, 2, 4, 8, 16, 32]
        if self.extrapolation_T_multiples is None:
            self.extrapolation_T_multiples = [1.5, 2.0, 3.0]


class ComprehensiveMetrics:
    @staticmethod
    def relative_l2_error(pred: np.ndarray, target: np.ndarray) -> float:
        diff_norm = np.linalg.norm(pred - target)
        target_norm = np.linalg.norm(target)
        return diff_norm / (target_norm + 1e-8)

    @staticmethod
    def mse(pred: np.ndarray, target: np.ndarray) -> float:
        return np.mean((pred - target) ** 2)

    @staticmethod
    def mae(pred: np.ndarray, target: np.ndarray) -> float:
        return np.mean(np.abs(pred - target))

    @staticmethod
    def max_absolute_error(pred: np.ndarray, target: np.ndarray) -> float:
        return np.max(np.abs(pred - target))

    @staticmethod
    def r2_score(pred: np.ndarray, target: np.ndarray) -> float:
        ss_res = np.sum((target - pred) ** 2)
        ss_tot = np.sum((target - np.mean(target)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-8))

    @staticmethod
    def pearson_correlation(pred: np.ndarray, target: np.ndarray) -> tuple:
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        corr, pval = stats.pearsonr(pred_flat, target_flat)
        return corr, pval

    @staticmethod
    def temporal_error_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
        n_steps = pred.shape[0]
        errors = {
            "rel_l2": np.zeros(n_steps),
            "mse": np.zeros(n_steps),
            "mae": np.zeros(n_steps),
            "max_ae": np.zeros(n_steps),
        }

        for t in range(n_steps):
            errors["rel_l2"][t] = ComprehensiveMetrics.relative_l2_error(
                pred[t], target[t]
            )
            errors["mse"][t] = ComprehensiveMetrics.mse(pred[t], target[t])
            errors["mae"][t] = ComprehensiveMetrics.mae(pred[t], target[t])
            errors["max_ae"][t] = ComprehensiveMetrics.max_absolute_error(
                pred[t], target[t]
            )

        return errors

    @staticmethod
    def compute_all_metrics(
        u_pred: np.ndarray, v_pred: np.ndarray, u_true: np.ndarray, v_true: np.ndarray
    ) -> dict:
        metrics = {
            "u": {
                "rel_l2": ComprehensiveMetrics.relative_l2_error(u_pred, u_true),
                "mse": ComprehensiveMetrics.mse(u_pred, u_true),
                "mae": ComprehensiveMetrics.mae(u_pred, u_true),
                "max_ae": ComprehensiveMetrics.max_absolute_error(u_pred, u_true),
                "r2": ComprehensiveMetrics.r2_score(u_pred, u_true),
            },
            "v": {
                "rel_l2": ComprehensiveMetrics.relative_l2_error(v_pred, v_true),
                "mse": ComprehensiveMetrics.mse(v_pred, v_true),
                "mae": ComprehensiveMetrics.mae(v_pred, v_true),
                "max_ae": ComprehensiveMetrics.max_absolute_error(v_pred, v_true),
                "r2": ComprehensiveMetrics.r2_score(v_pred, v_true),
            },
        }

        u_corr, u_pval = ComprehensiveMetrics.pearson_correlation(u_pred, u_true)
        v_corr, v_pval = ComprehensiveMetrics.pearson_correlation(v_pred, v_true)

        metrics["u"]["pearson_r"] = u_corr
        metrics["u"]["pearson_pval"] = u_pval
        metrics["v"]["pearson_r"] = v_corr
        metrics["v"]["pearson_pval"] = v_pval

        return metrics


class ResearchVisualizer:
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.colors = {
            "true": "#1f77b4",
            "pred": "#ff7f0e",
            "error": "#d62728",
            "start": "#2ca02c",
        }

    def plot_spatiotemporal_comparison(
        self,
        u_true: np.ndarray,
        v_true: np.ndarray,
        u_pred: np.ndarray,
        v_pred: np.ndarray,
        metrics: dict,
        save_name: str = "spatiotemporal.png",
    ):
        fig = plt.figure(figsize=(14, 12))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1], hspace=0.3, wspace=0.3)

        u_vmax = max(np.abs(u_true).max(), np.abs(u_pred).max())
        v_vmax = max(np.abs(v_true).max(), np.abs(v_pred).max())

        ax1 = plt.subplot(gs[0, 0])
        im1 = ax1.imshow(
            u_true.T,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-u_vmax,
            vmax=u_vmax,
            origin="lower",
        )
        ax1.set_title("u: Ground Truth", fontweight="bold")
        ax1.set_xlabel("Time step")
        ax1.set_ylabel("Spatial position")
        plt.colorbar(im1, ax=ax1, label="u")

        ax2 = plt.subplot(gs[0, 1])
        im2 = ax2.imshow(
            v_true.T,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-v_vmax,
            vmax=v_vmax,
            origin="lower",
        )
        ax2.set_title("v: Ground Truth", fontweight="bold")
        ax2.set_xlabel("Time step")
        ax2.set_ylabel("Spatial position")
        plt.colorbar(im2, ax=ax2, label="v")

        ax3 = plt.subplot(gs[1, 0])
        im3 = ax3.imshow(
            u_pred.T,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-u_vmax,
            vmax=u_vmax,
            origin="lower",
        )
        ax3.set_title(
            f'u: Prediction (Rel. L2: {metrics["u"]["rel_l2"]:.4f})', fontweight="bold"
        )
        ax3.set_xlabel("Time step")
        ax3.set_ylabel("Spatial position")
        plt.colorbar(im3, ax=ax3, label="u")

        ax4 = plt.subplot(gs[1, 1])
        im4 = ax4.imshow(
            v_pred.T,
            aspect="auto",
            cmap="RdBu_r",
            vmin=-v_vmax,
            vmax=v_vmax,
            origin="lower",
        )
        ax4.set_title(
            f'v: Prediction (Rel. L2: {metrics["v"]["rel_l2"]:.4f})', fontweight="bold"
        )
        ax4.set_xlabel("Time step")
        ax4.set_ylabel("Spatial position")
        plt.colorbar(im4, ax=ax4, label="v")

        u_error = np.abs(u_true - u_pred)
        ax5 = plt.subplot(gs[2, 0])
        im5 = ax5.imshow(u_error.T, aspect="auto", cmap="hot", origin="lower")
        ax5.set_title(
            f'u: Absolute Error (Max: {metrics["u"]["max_ae"]:.3e})', fontweight="bold"
        )
        ax5.set_xlabel("Time step")
        ax5.set_ylabel("Spatial position")
        plt.colorbar(im5, ax=ax5, label="|Error|")

        v_error = np.abs(v_true - v_pred)
        ax6 = plt.subplot(gs[2, 1])
        im6 = ax6.imshow(v_error.T, aspect="auto", cmap="hot", origin="lower")
        ax6.set_title(
            f'v: Absolute Error (Max: {metrics["v"]["max_ae"]:.3e})', fontweight="bold"
        )
        ax6.set_xlabel("Time step")
        ax6.set_ylabel("Spatial position")
        plt.colorbar(im6, ax=ax6, label="|Error|")

        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_single_step_comparison_regimes(
        self,
        dataset: FHNOperatorDataset,
        model: nn.Module,
        device: str,
        save_name: str = "single_step_regimes.png",
    ):
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, hspace=0.3, wspace=0.3)

        with h5py.File(dataset.data_file, "r") as f:
            params_all = np.array(f["params"])  # columns: Du, Dv, a, b, tau
            u_traj = np.array(f["u_traj"])
            v_traj = np.array(f["v_traj"])

        Du_values = params_all[:, 0]

        low_idx = np.argmin(Du_values)
        high_idx = np.argmax(Du_values)
        mid_idx = np.argsort(Du_values)[len(Du_values) // 2]

        regimes = [
            (low_idx, "Low Diffusion", 0),
            (mid_idx, "Medium Diffusion", 1),
            (high_idx, "High Diffusion", 2),
        ]

        model.eval()
        nx = u_traj.shape[-1]
        x = np.linspace(0, 1, nx)

        for sample_idx, regime_name, row in regimes:
            t_idx = 25  # mid trajectory
            u_in = u_traj[sample_idx, t_idx]
            v_in = v_traj[sample_idx, t_idx]
            u_out_true = u_traj[sample_idx, t_idx + 1]
            v_out_true = v_traj[sample_idx, t_idx + 1]

            if dataset.normalize:
                u_in_norm = (u_in - dataset.u_mean) / dataset.u_std
                v_in_norm = (v_in - dataset.v_mean) / dataset.v_std
            else:
                u_in_norm = u_in
                v_in_norm = v_in

            x_input = (
                torch.tensor(
                    np.stack([u_in_norm, v_in_norm], axis=0), dtype=torch.float32
                )
                .unsqueeze(0)
                .to(device)
            )

            with torch.no_grad():
                pred = model(x_input)

            u_out_pred = pred[0, 0].cpu().numpy()
            v_out_pred = pred[0, 1].cpu().numpy()

            if dataset.normalize:
                u_out_pred = u_out_pred * dataset.u_std + dataset.u_mean
                v_out_pred = v_out_pred * dataset.v_std + dataset.v_mean

            ax_u = plt.subplot(gs[row, 0])
            ax_u.plot(
                x, u_out_true, color=self.colors["true"], linewidth=2, label="True"
            )
            ax_u.plot(
                x,
                u_out_pred,
                color=self.colors["pred"],
                linewidth=2,
                linestyle="--",
                label="Predicted",
            )
            error_u = np.abs(u_out_true - u_out_pred)
            ax_u.fill_between(
                x, u_out_true, u_out_pred, alpha=0.3, color=self.colors["error"]
            )
            ax_u.set_ylabel("u")
            ax_u.set_title(
                f"{regime_name}: u (MAE: {error_u.mean():.3e})", fontweight="bold"
            )
            ax_u.grid(True, alpha=0.3)
            if row == 0:
                ax_u.legend()

            ax_v = plt.subplot(gs[row, 1])
            ax_v.plot(
                x, v_out_true, color=self.colors["true"], linewidth=2, label="True"
            )
            ax_v.plot(
                x,
                v_out_pred,
                color=self.colors["pred"],
                linewidth=2,
                linestyle="--",
                label="Predicted",
            )
            error_v = np.abs(v_out_true - v_out_pred)
            ax_v.fill_between(
                x, v_out_true, v_out_pred, alpha=0.3, color=self.colors["error"]
            )
            ax_v.set_ylabel("v")
            ax_v.set_title(
                f"{regime_name}: v (MAE: {error_v.mean():.3e})", fontweight="bold"
            )
            ax_v.grid(True, alpha=0.3)

            ax_err = plt.subplot(gs[row, 2])
            ax_err.plot(
                x, error_u, color=self.colors["true"], linewidth=2, label="u error"
            )
            ax_err.plot(
                x, error_v, color=self.colors["pred"], linewidth=2, label="v error"
            )
            ax_err.set_ylabel("Absolute Error")
            ax_err.set_title(f"{regime_name}: Errors", fontweight="bold")
            ax_err.grid(True, alpha=0.3)
            ax_err.legend()

            Du, Dv, a, b, tau = params_all[sample_idx]
            param_text = f"Du={Du:.3f}, Dv={Dv:.3f}\na={a:.2f}, b={b:.2f}, τ={tau:.1f}"
            ax_u.text(
                0.02,
                0.98,
                param_text,
                transform=ax_u.transAxes,
                verticalalignment="top",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_rollout_error_evolution(
        self,
        rollout_errors: Dict[int, Dict[str, float]],
        save_name: str = "rollout_error_evolution.png",
    ):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        steps = sorted(rollout_errors.keys())
        metrics = ["rel_l2", "mse", "mae", "max_ae"]
        titles = ["Relative L2 Error", "MSE", "MAE", "Max Absolute Error"]

        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx // 2, idx % 2]

            u_values = [rollout_errors[s]["u"][metric] for s in steps]
            v_values = [rollout_errors[s]["v"][metric] for s in steps]

            ax.plot(
                steps,
                u_values,
                color=self.colors["true"],
                linewidth=2,
                marker="o",
                markersize=4,
                label="u (Activator)",
            )
            ax.plot(
                steps,
                v_values,
                color=self.colors["pred"],
                linewidth=2,
                marker="s",
                markersize=4,
                label="v (Inhibitor)",
            )

            ax.set_xlabel("Rollout Step", fontweight="bold")
            ax.set_ylabel(title, fontweight="bold")
            ax.set_title(f"{title} vs Rollout Length", fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)

            if metric == "mse":
                ax.set_yscale("log")

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_parameter_generalization(
        self,
        param_analysis: Dict[str, Dict],
        save_name: str = "parameter_generalization.png",
    ):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        param_names = ["Du", "Dv", "a", "b", "tau"]
        param_labels = [
            "Du (u diffusion)",
            "Dv (v diffusion)",
            "a",
            "b",
            "τ (timescale)",
        ]

        for idx, (param_name, param_label) in enumerate(zip(param_names, param_labels)):
            ax = axes[idx]
            data = param_analysis[param_name]

            ax.scatter(
                data["values"],
                data["u_errors"],
                alpha=0.5,
                s=20,
                color=self.colors["true"],
                label="u",
            )
            ax.scatter(
                data["values"],
                data["v_errors"],
                alpha=0.5,
                s=20,
                color=self.colors["pred"],
                label="v",
            )

            bin_edges = np.linspace(data["values"].min(), data["values"].max(), 10)
            u_bin_means = []
            v_bin_means = []
            bin_centers = []

            for i in range(len(bin_edges) - 1):
                mask = (data["values"] >= bin_edges[i]) & (
                    data["values"] < bin_edges[i + 1]
                )
                if mask.sum() > 0:
                    u_bin_means.append(data["u_errors"][mask].mean())
                    v_bin_means.append(data["v_errors"][mask].mean())
                    bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)

            ax.plot(
                bin_centers,
                u_bin_means,
                color=self.colors["true"],
                linewidth=3,
                marker="D",
                markersize=8,
                label="u (binned mean)",
            )
            ax.plot(
                bin_centers,
                v_bin_means,
                color=self.colors["pred"],
                linewidth=3,
                marker="D",
                markersize=8,
                label="v (binned mean)",
            )

            ax.set_xlabel(param_label, fontweight="bold")
            ax.set_ylabel("Relative L2 Error", fontweight="bold")
            ax.set_title(f"Error vs {param_label}", fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # only 5 params, drop the 6th axis
        axes[5].remove()

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_efficiency_scaling(
        self, efficiency_data: Dict, save_name: str = "efficiency_scaling.png"
    ):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax1 = axes[0]
        resolutions = efficiency_data["resolutions"]["resolutions"]
        fno_times = efficiency_data["resolutions"]["fno_times"]
        fd_times = efficiency_data["resolutions"]["fd_times"]

        ax1.plot(
            resolutions,
            fno_times,
            color=self.colors["pred"],
            linewidth=2.5,
            marker="o",
            markersize=8,
            label="FNO (Neural Operator)",
        )
        ax1.plot(
            resolutions,
            fd_times,
            color=self.colors["true"],
            linewidth=2.5,
            marker="s",
            markersize=8,
            label="Finite Difference",
        )

        ax1.set_xlabel("Spatial Resolution (nx)", fontweight="bold")
        ax1.set_ylabel("Time per Step (ms)", fontweight="bold")
        ax1.set_title(
            "Computational Scaling: FNO vs Traditional Solver", fontweight="bold"
        )
        ax1.set_xscale("log", base=2)
        ax1.set_yscale("log")
        ax1.legend(loc="upper left", fontsize=10)
        ax1.grid(True, alpha=0.3, which="both")

        speedup_at_1024 = fd_times[-1] / fno_times[-1] if len(fno_times) > 0 else 0
        ax1.text(
            0.05,
            0.95,
            f"Speedup at {resolutions[-1]}:\n{speedup_at_1024:.1f}×",
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
            fontsize=10,
            fontweight="bold",
        )

        ax2 = axes[1]
        batch_sizes = efficiency_data["batch_sizes"]["batch_sizes"]
        times_per_sample = efficiency_data["batch_sizes"]["times_per_sample"]

        ax2.plot(
            batch_sizes,
            times_per_sample,
            color=self.colors["pred"],
            linewidth=2.5,
            marker="o",
            markersize=8,
            label="FNO",
        )
        ax2.set_xlabel("Batch Size", fontweight="bold")
        ax2.set_ylabel("Time per Sample (ms)", fontweight="bold")
        ax2.set_title("FNO Efficiency with Batching", fontweight="bold")
        ax2.set_xscale("log", base=2)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        speedup = times_per_sample[0] / times_per_sample[-1]
        ax2.text(
            0.05,
            0.95,
            f"Batching Speedup:\n{speedup:.1f}×",
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
            fontsize=10,
            fontweight="bold",
        )

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_temporal_snapshots(
        self,
        u_true: np.ndarray,
        v_true: np.ndarray,
        u_pred: np.ndarray,
        v_pred: np.ndarray,
        x: np.ndarray,
        time_indices: list,
        save_name: str = "temporal_snapshots.png",
    ):
        n_times = len(time_indices)
        fig, axes = plt.subplots(2, n_times, figsize=(4 * n_times, 7))

        if n_times == 1:
            axes = axes.reshape(2, 1)

        for i, t_idx in enumerate(time_indices):
            ax_u = axes[0, i]
            ax_u.plot(
                x,
                u_true[t_idx],
                color=self.colors["true"],
                linewidth=2,
                label="Ground Truth",
            )
            ax_u.plot(
                x,
                u_pred[t_idx],
                color=self.colors["pred"],
                linewidth=2,
                linestyle="--",
                label="Prediction",
            )
            ax_u.fill_between(
                x, u_true[t_idx], u_pred[t_idx], alpha=0.3, color=self.colors["error"]
            )

            error_u = np.abs(u_true[t_idx] - u_pred[t_idx])
            ax_u.set_xlabel("Spatial coordinate")
            ax_u.set_ylabel("u")
            ax_u.set_title(f"t = {t_idx} (MAE: {error_u.mean():.3e})")
            ax_u.grid(True, alpha=0.3)
            if i == 0:
                ax_u.legend(loc="best")

            ax_v = axes[1, i]
            ax_v.plot(
                x,
                v_true[t_idx],
                color=self.colors["true"],
                linewidth=2,
                label="Ground Truth",
            )
            ax_v.plot(
                x,
                v_pred[t_idx],
                color=self.colors["pred"],
                linewidth=2,
                linestyle="--",
                label="Prediction",
            )
            ax_v.fill_between(
                x, v_true[t_idx], v_pred[t_idx], alpha=0.3, color=self.colors["error"]
            )

            error_v = np.abs(v_true[t_idx] - v_pred[t_idx])
            ax_v.set_xlabel("Spatial coordinate")
            ax_v.set_ylabel("v")
            ax_v.set_title(f"t = {t_idx} (MAE: {error_v.mean():.3e})")
            ax_v.grid(True, alpha=0.3)
            if i == 0:
                ax_v.legend(loc="best")

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_phase_portraits(
        self,
        u_true: np.ndarray,
        v_true: np.ndarray,
        u_pred: np.ndarray,
        v_pred: np.ndarray,
        spatial_indices: list,
        save_name: str = "phase_portraits.png",
    ):
        n_points = len(spatial_indices)
        fig, axes = plt.subplots(1, n_points, figsize=(5 * n_points, 5))

        if n_points == 1:
            axes = [axes]

        for i, x_idx in enumerate(spatial_indices):
            ax = axes[i]

            ax.plot(
                u_true[:, x_idx],
                v_true[:, x_idx],
                color=self.colors["true"],
                linewidth=2.5,
                label="Ground Truth",
                alpha=0.8,
            )

            ax.plot(
                u_pred[:, x_idx],
                v_pred[:, x_idx],
                color=self.colors["pred"],
                linewidth=2,
                linestyle="--",
                label="Prediction",
                alpha=0.8,
            )

            ax.scatter(
                u_true[0, x_idx],
                v_true[0, x_idx],
                c=self.colors["start"],
                s=150,
                marker="o",
                label="Start",
                zorder=10,
                edgecolors="black",
                linewidths=1.5,
            )
            ax.scatter(
                u_true[-1, x_idx],
                v_true[-1, x_idx],
                c=self.colors["true"],
                s=150,
                marker="s",
                label="True End",
                zorder=10,
                edgecolors="black",
                linewidths=1.5,
            )
            ax.scatter(
                u_pred[-1, x_idx],
                v_pred[-1, x_idx],
                c=self.colors["pred"],
                s=150,
                marker="^",
                label="Pred End",
                zorder=10,
                edgecolors="black",
                linewidths=1.5,
            )

            traj_error = np.sqrt(
                (u_true[:, x_idx] - u_pred[:, x_idx]) ** 2
                + (v_true[:, x_idx] - v_pred[:, x_idx]) ** 2
            )
            mean_traj_error = traj_error.mean()

            ax.set_xlabel("u", fontweight="bold")
            ax.set_ylabel("v", fontweight="bold")
            ax.set_title(
                f"x = {x_idx} (Traj. Error: {mean_traj_error:.3e})", fontweight="bold"
            )
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", framealpha=0.9)

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_error_analysis(
        self,
        u_true: np.ndarray,
        v_true: np.ndarray,
        u_pred: np.ndarray,
        v_pred: np.ndarray,
        save_name: str = "error_analysis.png",
    ):
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.35)

        u_error = u_true - u_pred
        v_error = v_true - v_pred
        u_abs_error = np.abs(u_error)
        v_abs_error = np.abs(v_error)

        u_temporal = ComprehensiveMetrics.temporal_error_metrics(u_pred, u_true)
        v_temporal = ComprehensiveMetrics.temporal_error_metrics(v_pred, v_true)
        time_steps = np.arange(len(u_temporal["rel_l2"]))

        ax1 = plt.subplot(gs[0, 0])
        ax1.hist(
            u_error.flatten(),
            bins=50,
            alpha=0.7,
            color=self.colors["true"],
            label="u",
            density=True,
        )
        ax1.axvline(0, color="black", linestyle="--", linewidth=1)
        ax1.set_xlabel("Error")
        ax1.set_ylabel("Density")
        ax1.set_title("u Error Distribution", fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        u_mean_err = u_error.mean()
        u_std_err = u_error.std()
        ax1.text(
            0.05,
            0.95,
            f"μ = {u_mean_err:.3e}\nσ = {u_std_err:.3e}",
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax2 = plt.subplot(gs[0, 1])
        ax2.hist(
            v_error.flatten(),
            bins=50,
            alpha=0.7,
            color=self.colors["pred"],
            label="v",
            density=True,
        )
        ax2.axvline(0, color="black", linestyle="--", linewidth=1)
        ax2.set_xlabel("Error")
        ax2.set_ylabel("Density")
        ax2.set_title("v Error Distribution", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        v_mean_err = v_error.mean()
        v_std_err = v_error.std()
        ax2.text(
            0.05,
            0.95,
            f"μ = {v_mean_err:.3e}\nσ = {v_std_err:.3e}",
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        ax3 = plt.subplot(gs[0, 2])
        stats.probplot(u_error.flatten(), dist="norm", plot=ax3)
        ax3.set_title("u Error Q-Q Plot", fontweight="bold")
        ax3.grid(True, alpha=0.3)

        ax4 = plt.subplot(gs[1, 0])
        ax4.plot(
            time_steps,
            u_temporal["rel_l2"],
            color=self.colors["true"],
            linewidth=2,
            marker="o",
            markersize=3,
            label="u",
        )
        ax4.plot(
            time_steps,
            v_temporal["rel_l2"],
            color=self.colors["pred"],
            linewidth=2,
            marker="s",
            markersize=3,
            label="v",
        )
        ax4.set_xlabel("Time step")
        ax4.set_ylabel("Relative L2 Error")
        ax4.set_title("Error Evolution Over Time", fontweight="bold")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        ax5 = plt.subplot(gs[1, 1])
        ax5.semilogy(
            time_steps,
            u_temporal["mse"],
            color=self.colors["true"],
            linewidth=2,
            marker="o",
            markersize=3,
            label="u",
        )
        ax5.semilogy(
            time_steps,
            v_temporal["mse"],
            color=self.colors["pred"],
            linewidth=2,
            marker="s",
            markersize=3,
            label="v",
        )
        ax5.set_xlabel("Time step")
        ax5.set_ylabel("MSE (log scale)")
        ax5.set_title("MSE Evolution (Log Scale)", fontweight="bold")
        ax5.legend()
        ax5.grid(True, alpha=0.3, which="both")

        ax6 = plt.subplot(gs[1, 2])
        ax6.plot(
            time_steps,
            u_temporal["max_ae"],
            color=self.colors["true"],
            linewidth=2,
            marker="o",
            markersize=3,
            label="u",
        )
        ax6.plot(
            time_steps,
            v_temporal["max_ae"],
            color=self.colors["pred"],
            linewidth=2,
            marker="s",
            markersize=3,
            label="v",
        )
        ax6.set_xlabel("Time step")
        ax6.set_ylabel("Max Absolute Error")
        ax6.set_title("Maximum Error Evolution", fontweight="bold")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        ax7 = plt.subplot(gs[2, 0])
        spatial_u_error = u_abs_error.mean(axis=0)
        spatial_v_error = v_abs_error.mean(axis=0)
        x_coords = np.arange(len(spatial_u_error))

        ax7.plot(
            x_coords, spatial_u_error, color=self.colors["true"], linewidth=2, label="u"
        )
        ax7.plot(
            x_coords, spatial_v_error, color=self.colors["pred"], linewidth=2, label="v"
        )
        ax7.set_xlabel("Spatial position")
        ax7.set_ylabel("Mean Absolute Error")
        ax7.set_title("Spatial Error Distribution", fontweight="bold")
        ax7.legend()
        ax7.grid(True, alpha=0.3)

        ax8 = plt.subplot(gs[2, 1])
        sample_indices = np.random.choice(
            u_true.size, size=min(5000, u_true.size), replace=False
        )
        ax8.scatter(
            u_true.flatten()[sample_indices],
            u_pred.flatten()[sample_indices],
            alpha=0.3,
            s=1,
            c=self.colors["true"],
            label="u",
        )

        u_range = [u_true.min(), u_true.max()]
        ax8.plot(u_range, u_range, "k--", linewidth=2, label="Perfect")

        ax8.set_xlabel("True u")
        ax8.set_ylabel("Predicted u")
        ax8.set_title("Prediction Scatter Plot (u)", fontweight="bold")
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        ax8.set_aspect("equal", adjustable="box")

        ax9 = plt.subplot(gs[2, 2])
        ax9.scatter(
            v_true.flatten()[sample_indices],
            v_pred.flatten()[sample_indices],
            alpha=0.3,
            s=1,
            c=self.colors["pred"],
            label="v",
        )

        v_range = [v_true.min(), v_true.max()]
        ax9.plot(v_range, v_range, "k--", linewidth=2, label="Perfect")

        ax9.set_xlabel("True v")
        ax9.set_ylabel("Predicted v")
        ax9.set_title("Prediction Scatter Plot (v)", fontweight="bold")
        ax9.legend()
        ax9.grid(True, alpha=0.3)
        ax9.set_aspect("equal", adjustable="box")

        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches="tight")
        plt.close()

    def plot_spectral_analysis(
        self,
        u_true: np.ndarray,
        v_true: np.ndarray,
        u_pred: np.ndarray,
        v_pred: np.ndarray,
        save_name: str = "spectral_analysis.png",
    ):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        t_mid = u_true.shape[0] // 2

        u_true_fft = np.abs(fft(u_true[t_mid]))
        u_pred_fft = np.abs(fft(u_pred[t_mid]))
        n = len(u_true_fft)
        freq = fftfreq(n, d=1.0 / n)[: n // 2]

        ax1 = axes[0, 0]
        ax1.semilogy(
            freq,
            u_true_fft[: n // 2],
            color=self.colors["true"],
            linewidth=2,
            label="Ground Truth",
        )
        ax1.semilogy(
            freq,
            u_pred_fft[: n // 2],
            color=self.colors["pred"],
            linewidth=2,
            linestyle="--",
            label="Prediction",
        )
        ax1.set_xlabel("Frequency")
        ax1.set_ylabel("Power (log scale)")
        ax1.set_title(f"u Power Spectrum (t={t_mid})", fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3, which="both")

        v_true_fft = np.abs(fft(v_true[t_mid]))
        v_pred_fft = np.abs(fft(v_pred[t_mid]))

        ax2 = axes[0, 1]
        ax2.semilogy(
            freq,
            v_true_fft[: n // 2],
            color=self.colors["true"],
            linewidth=2,
            label="Ground Truth",
        )
        ax2.semilogy(
            freq,
            v_pred_fft[: n // 2],
            color=self.colors["pred"],
            linewidth=2,
            linestyle="--",
            label="Prediction",
        )
        ax2.set_xlabel("Frequency")
        ax2.set_ylabel("Power (log scale)")
        ax2.set_title(f"v Power Spectrum (t={t_mid})", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3, which="both")

        u_spectral_error = np.abs(u_true_fft[: n // 2] - u_pred_fft[: n // 2])
        v_spectral_error = np.abs(v_true_fft[: n // 2] - v_pred_fft[: n // 2])

        ax3 = axes[1, 0]
        ax3.semilogy(freq, u_spectral_error, color=self.colors["error"], linewidth=2)
        ax3.set_xlabel("Frequency")
        ax3.set_ylabel("Spectral Error (log scale)")
        ax3.set_title("u Spectral Error", fontweight="bold")
        ax3.grid(True, alpha=0.3, which="both")

        ax4 = axes[1, 1]
        ax4.semilogy(freq, v_spectral_error, color=self.colors["error"], linewidth=2)
        ax4.set_xlabel("Frequency")
        ax4.set_ylabel("Spectral Error (log scale)")
        ax4.set_title("v Spectral Error", fontweight="bold")
        ax4.grid(True, alpha=0.3, which="both")

        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches="tight")
        plt.close()

    def create_metrics_summary_figure(
        self, metrics: dict, save_name: str = "metrics_summary.png"
    ):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("tight")
        ax.axis("off")

        metric_names = [
            "Relative L2 Error",
            "Mean Squared Error",
            "Mean Absolute Error",
            "Max Absolute Error",
            "R² Score",
            "Pearson Correlation",
        ]

        u_values = [
            f"{metrics['u']['rel_l2']:.6f}",
            f"{metrics['u']['mse']:.6e}",
            f"{metrics['u']['mae']:.6e}",
            f"{metrics['u']['max_ae']:.6e}",
            f"{metrics['u']['r2']:.6f}",
            f"{metrics['u']['pearson_r']:.6f}",
        ]

        v_values = [
            f"{metrics['v']['rel_l2']:.6f}",
            f"{metrics['v']['mse']:.6e}",
            f"{metrics['v']['mae']:.6e}",
            f"{metrics['v']['max_ae']:.6e}",
            f"{metrics['v']['r2']:.6f}",
            f"{metrics['v']['pearson_r']:.6f}",
        ]

        table_data = []
        for i, name in enumerate(metric_names):
            table_data.append([name, u_values[i], v_values[i]])

        table = ax.table(
            cellText=table_data,
            colLabels=["Metric", "u (Activator)", "v (Inhibitor)"],
            cellLoc="center",
            loc="center",
            colWidths=[0.4, 0.3, 0.3],
        )

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)

        for i in range(3):
            table[(0, i)].set_facecolor("#4472C4")
            table[(0, i)].set_text_props(weight="bold", color="white")

        for i in range(1, len(table_data) + 1):
            for j in range(3):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor("#E7E6E6")
                else:
                    table[(i, j)].set_facecolor("white")

        plt.title(
            "Quantitative Performance Metrics", fontsize=14, fontweight="bold", pad=20
        )

        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches="tight")
        plt.close()


class ComprehensiveEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.visualizer = ResearchVisualizer(config.output_dir)
        self.model = None
        self.dataset = None
        self.device = config.device

    def load_model_and_data(self):
        print("Loading model and dataset...")

        self.dataset = FHNOperatorDataset(
            self.config.data_file, mode="single_step", train=False
        )
        self.dataset.data_file = self.config.data_file

        checkpoint = torch.load(self.config.checkpoint_file, map_location=self.device)
        model_config = checkpoint.get("config", {})

        self.model = FNO(
            modes=model_config.get("model", {}).get("modes", 16),
            width=model_config.get("model", {}).get("width", 64),
            n_layers=model_config.get("model", {}).get("n_layers", 4),
            dim=self.dataset.dim,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(
            f"Model loaded: {sum(p.numel() for p in self.model.parameters()):,} parameters"
        )

    def evaluate_single_sample(self, sample_idx: int = 0):
        print(f"\n{'='*70}")
        print(f" SINGLE SAMPLE EVALUATION (Sample {sample_idx})")
        print(f"{'='*70}")

        with h5py.File(self.dataset.data_file, "r") as f:
            u_traj_true = f["u_traj"][sample_idx][: self.config.rollout_steps + 1]
            v_traj_true = f["v_traj"][sample_idx][: self.config.rollout_steps + 1]

        u0 = u_traj_true[0]
        v0 = v_traj_true[0]

        if self.dataset.normalize:
            u0_norm = (u0 - self.dataset.u_mean) / self.dataset.u_std
            v0_norm = (v0 - self.dataset.v_mean) / self.dataset.v_std
        else:
            u0_norm = u0
            v0_norm = v0

        x0 = (
            torch.tensor(np.stack([u0_norm, v0_norm], axis=0), dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        print("Generating predictions...")
        with torch.no_grad():
            trajectory = self.model.rollout(x0, self.config.rollout_steps)

        u_traj_pred = trajectory[0, :, 0].cpu().numpy()
        v_traj_pred = trajectory[0, :, 1].cpu().numpy()

        if self.dataset.normalize:
            u_traj_pred = u_traj_pred * self.dataset.u_std + self.dataset.u_mean
            v_traj_pred = v_traj_pred * self.dataset.v_std + self.dataset.v_mean

        metrics = ComprehensiveMetrics.compute_all_metrics(
            u_traj_pred, v_traj_pred, u_traj_true, v_traj_true
        )

        self._print_metrics(metrics)

        print("\nGenerating visualizations...")
        nx = u_traj_true.shape[-1]
        x = np.linspace(0, 1, nx)

        self.visualizer.plot_spatiotemporal_comparison(
            u_traj_true,
            v_traj_true,
            u_traj_pred,
            v_traj_pred,
            metrics,
            save_name="01_spatiotemporal_comparison.png",
        )

        time_indices = [
            0,
            self.config.rollout_steps // 4,
            self.config.rollout_steps // 2,
            3 * self.config.rollout_steps // 4,
            self.config.rollout_steps,
        ]
        self.visualizer.plot_temporal_snapshots(
            u_traj_true,
            v_traj_true,
            u_traj_pred,
            v_traj_pred,
            x,
            time_indices,
            save_name="02_temporal_snapshots.png",
        )

        spatial_indices = [nx // 4, nx // 2, 3 * nx // 4]
        self.visualizer.plot_phase_portraits(
            u_traj_true,
            v_traj_true,
            u_traj_pred,
            v_traj_pred,
            spatial_indices,
            save_name="03_phase_portraits.png",
        )

        self.visualizer.plot_error_analysis(
            u_traj_true,
            v_traj_true,
            u_traj_pred,
            v_traj_pred,
            save_name="04_error_analysis.png",
        )

        self.visualizer.plot_spectral_analysis(
            u_traj_true,
            v_traj_true,
            u_traj_pred,
            v_traj_pred,
            save_name="05_spectral_analysis.png",
        )

        self.visualizer.create_metrics_summary_figure(
            metrics, save_name="06_metrics_summary.png"
        )

        return metrics

    def evaluate_rollout_error_accumulation(self):
        print(f"\n{'='*70}")
        print(" ROLLOUT ERROR ACCUMULATION ANALYSIS")
        print(f"{'='*70}")

        test_steps = [1, 5, 10, 20, 30, 40, 50, 75, 100]
        test_steps = [s for s in test_steps if s <= self.config.rollout_steps]

        rollout_errors = {}

        n_samples = min(10, len(self.dataset))

        with h5py.File(self.dataset.data_file, "r") as f:
            u_traj_all = np.array(f["u_traj"])
            v_traj_all = np.array(f["v_traj"])

        for n_steps in tqdm(test_steps, desc="Testing rollout lengths"):
            u_errors = []
            v_errors = []

            for sample_idx in range(n_samples):
                u_traj_true = u_traj_all[sample_idx][: n_steps + 1]
                v_traj_true = v_traj_all[sample_idx][: n_steps + 1]

                u0 = u_traj_true[0]
                v0 = v_traj_true[0]

                if self.dataset.normalize:
                    u0_norm = (u0 - self.dataset.u_mean) / self.dataset.u_std
                    v0_norm = (v0 - self.dataset.v_mean) / self.dataset.v_std
                else:
                    u0_norm = u0
                    v0_norm = v0

                x0 = (
                    torch.tensor(
                        np.stack([u0_norm, v0_norm], axis=0), dtype=torch.float32
                    )
                    .unsqueeze(0)
                    .to(self.device)
                )

                with torch.no_grad():
                    trajectory = self.model.rollout(x0, n_steps)

                u_traj_pred = trajectory[0, :, 0].cpu().numpy()
                v_traj_pred = trajectory[0, :, 1].cpu().numpy()

                if self.dataset.normalize:
                    u_traj_pred = u_traj_pred * self.dataset.u_std + self.dataset.u_mean
                    v_traj_pred = v_traj_pred * self.dataset.v_std + self.dataset.v_mean

                u_metrics = {
                    "rel_l2": ComprehensiveMetrics.relative_l2_error(
                        u_traj_pred, u_traj_true
                    ),
                    "mse": ComprehensiveMetrics.mse(u_traj_pred, u_traj_true),
                    "mae": ComprehensiveMetrics.mae(u_traj_pred, u_traj_true),
                    "max_ae": ComprehensiveMetrics.max_absolute_error(
                        u_traj_pred, u_traj_true
                    ),
                }
                v_metrics = {
                    "rel_l2": ComprehensiveMetrics.relative_l2_error(
                        v_traj_pred, v_traj_true
                    ),
                    "mse": ComprehensiveMetrics.mse(v_traj_pred, v_traj_true),
                    "mae": ComprehensiveMetrics.mae(v_traj_pred, v_traj_true),
                    "max_ae": ComprehensiveMetrics.max_absolute_error(
                        v_traj_pred, v_traj_true
                    ),
                }

                u_errors.append(u_metrics)
                v_errors.append(v_metrics)

            rollout_errors[n_steps] = {
                "u": {k: np.mean([e[k] for e in u_errors]) for k in u_errors[0].keys()},
                "v": {k: np.mean([e[k] for e in v_errors]) for k in v_errors[0].keys()},
            }

        self.visualizer.plot_rollout_error_evolution(
            rollout_errors, save_name="07_rollout_error_evolution.png"
        )

        with open(self.config.output_dir + "/rollout_errors.json", "w") as f:
            json.dump({str(k): v for k, v in rollout_errors.items()}, f, indent=2)

        print("Rollout error analysis complete")
        return rollout_errors

    def evaluate_parameter_generalization(self):
        print(f"\n{'='*70}")
        print(" PARAMETER GENERALIZATION ANALYSIS")
        print(f"{'='*70}")

        with h5py.File(self.dataset.data_file, "r") as f:
            params_all = np.array(f["params"])  # columns: Du, Dv, a, b, tau
            u_traj_all = np.array(f["u_traj"])
            v_traj_all = np.array(f["v_traj"])

        n_samples = min(self.config.n_test_samples, len(params_all))

        u_errors = []
        v_errors = []

        for sample_idx in tqdm(range(n_samples), desc="Computing errors"):
            u_traj_true = u_traj_all[sample_idx][: self.config.rollout_steps + 1]
            v_traj_true = v_traj_all[sample_idx][: self.config.rollout_steps + 1]

            u0 = u_traj_true[0]
            v0 = v_traj_true[0]

            if self.dataset.normalize:
                u0_norm = (u0 - self.dataset.u_mean) / self.dataset.u_std
                v0_norm = (v0 - self.dataset.v_mean) / self.dataset.v_std
            else:
                u0_norm = u0
                v0_norm = v0

            x0 = (
                torch.tensor(np.stack([u0_norm, v0_norm], axis=0), dtype=torch.float32)
                .unsqueeze(0)
                .to(self.device)
            )

            with torch.no_grad():
                trajectory = self.model.rollout(x0, self.config.rollout_steps)

            u_traj_pred = trajectory[0, :, 0].cpu().numpy()
            v_traj_pred = trajectory[0, :, 1].cpu().numpy()

            if self.dataset.normalize:
                u_traj_pred = u_traj_pred * self.dataset.u_std + self.dataset.u_mean
                v_traj_pred = v_traj_pred * self.dataset.v_std + self.dataset.v_mean

            u_err = ComprehensiveMetrics.relative_l2_error(u_traj_pred, u_traj_true)
            v_err = ComprehensiveMetrics.relative_l2_error(v_traj_pred, v_traj_true)

            u_errors.append(u_err)
            v_errors.append(v_err)

        u_errors = np.array(u_errors)
        v_errors = np.array(v_errors)
        params_subset = params_all[:n_samples]

        param_analysis = {}
        param_names = ["Du", "Dv", "a", "b", "tau"]

        for i, param_name in enumerate(param_names):
            param_analysis[param_name] = {
                "values": params_subset[:, i],
                "u_errors": u_errors,
                "v_errors": v_errors,
            }

        self.visualizer.plot_parameter_generalization(
            param_analysis, save_name="08_parameter_generalization.png"
        )

        self._create_parameter_stats_table(param_analysis)

        print("Parameter generalization analysis complete")
        return param_analysis

    def evaluate_efficiency_scaling(self):
        print(f"\n{'='*70}")
        print(" COMPUTATIONAL EFFICIENCY ANALYSIS")
        print(f"{'='*70}")

        efficiency_data = {}

        print("\nTesting resolution scaling...")
        fno_resolution_times = []
        fd_resolution_times = []

        with h5py.File(self.dataset.data_file, "r") as f:
            params_sample = f["params"][0]

        fhn_params = FHNParams(
            Du=float(params_sample[0]),
            Dv=float(params_sample[1]),
            a=float(params_sample[2]),
            b=float(params_sample[3]),
            tau=float(params_sample[4]),
        )

        for nx in tqdm(self.config.test_resolutions, desc="Resolution tests"):
            x = torch.randn(1, 2, nx, device=self.device)

            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = self.model(x)

            if self.device == "cuda":
                torch.cuda.synchronize()

            times = []
            for _ in range(self.config.efficiency_n_runs):
                if self.device == "cuda":
                    torch.cuda.synchronize()

                start = time.perf_counter()
                with torch.no_grad():
                    _ = self.model(x)

                if self.device == "cuda":
                    torch.cuda.synchronize()

                end = time.perf_counter()
                times.append((end - start) * 1000)

            fno_resolution_times.append(np.mean(times))

            fd_solver = FDBackend(nx=nx, dx=1.0 / nx)

            u0 = np.random.randn(nx) * 0.1
            v0 = np.random.randn(nx) * 0.1

            # Warmup
            for _ in range(3):
                _ = fd_solver.solve(u0, v0, fhn_params, T=0.01, dt=0.001, n_save=1)

            fd_times = []
            for _ in range(
                min(self.config.efficiency_n_runs, 10)
            ):  # FD is slow, use fewer runs
                start = time.perf_counter()
                _ = fd_solver.solve(u0, v0, fhn_params, T=0.01, dt=0.001, n_save=1)
                end = time.perf_counter()
                fd_times.append((end - start) * 1000)

            fd_resolution_times.append(np.mean(fd_times))

        efficiency_data["resolutions"] = {
            "resolutions": self.config.test_resolutions,
            "fno_times": fno_resolution_times,
            "fd_times": fd_resolution_times,
        }

        print("\nTesting batch size scaling...")
        batch_times = []
        nx = 256  # Fixed resolution

        for batch_size in tqdm(self.config.test_batch_sizes, desc="Batch size tests"):
            x = torch.randn(batch_size, 2, nx, device=self.device)

            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = self.model(x)

            if self.device == "cuda":
                torch.cuda.synchronize()

            times = []
            for _ in range(self.config.efficiency_n_runs):
                if self.device == "cuda":
                    torch.cuda.synchronize()

                start = time.perf_counter()
                with torch.no_grad():
                    _ = self.model(x)

                if self.device == "cuda":
                    torch.cuda.synchronize()

                end = time.perf_counter()
                times.append((end - start) * 1000 / batch_size)

            batch_times.append(np.mean(times))

        efficiency_data["batch_sizes"] = {
            "batch_sizes": self.config.test_batch_sizes,
            "times_per_sample": batch_times,
        }

        self.visualizer.plot_efficiency_scaling(
            efficiency_data, save_name="09_efficiency_scaling.png"
        )

        self._create_efficiency_table(efficiency_data)

        print("Efficiency analysis complete")
        return efficiency_data

    def evaluate_single_step_regimes(self):
        print(f"\n{'='*70}")
        print(" SINGLE-STEP PREDICTION ACROSS REGIMES")
        print(f"{'='*70}")

        self.visualizer.plot_single_step_comparison_regimes(
            self.dataset,
            self.model,
            self.device,
            save_name="10_single_step_regimes.png",
        )

        print("Single-step regime analysis complete")

    def _print_metrics(self, metrics: dict):
        print(f"\n{'='*70}")
        print(" QUANTITATIVE METRICS")
        print(f"{'='*70}")
        print("\n--- u (Activator) ---")
        print(f"  Relative L2 Error:      {metrics['u']['rel_l2']:.6f}")
        print(f"  Mean Squared Error:     {metrics['u']['mse']:.6e}")
        print(f"  Mean Absolute Error:    {metrics['u']['mae']:.6e}")
        print(f"  Max Absolute Error:     {metrics['u']['max_ae']:.6e}")
        print(f"  R² Score:               {metrics['u']['r2']:.6f}")
        print(f"  Pearson Correlation:    {metrics['u']['pearson_r']:.6f}")

        print("\n--- v (Inhibitor) ---")
        print(f"  Relative L2 Error:      {metrics['v']['rel_l2']:.6f}")
        print(f"  Mean Squared Error:     {metrics['v']['mse']:.6e}")
        print(f"  Mean Absolute Error:    {metrics['v']['mae']:.6e}")
        print(f"  Max Absolute Error:     {metrics['v']['max_ae']:.6e}")
        print(f"  R² Score:               {metrics['v']['r2']:.6f}")
        print(f"  Pearson Correlation:    {metrics['v']['pearson_r']:.6f}")

    def _create_parameter_stats_table(self, param_analysis: Dict):
        output_file = Path(self.config.output_dir) / "parameter_statistics.txt"

        with open(output_file, "w") as f:
            f.write("=" * 70 + "\n")
            f.write(" PARAMETER GENERALIZATION STATISTICS\n")
            f.write("=" * 70 + "\n\n")

            for param_name, data in param_analysis.items():
                f.write(f"\n{param_name}:\n")
                f.write(
                    f"  Range: [{data['values'].min():.4f}, {data['values'].max():.4f}]\n"
                )
                f.write(
                    f"  Mean u error: {data['u_errors'].mean():.6f} ± {data['u_errors'].std():.6f}\n"
                )
                f.write(
                    f"  Mean v error: {data['v_errors'].mean():.6f} ± {data['v_errors'].std():.6f}\n"
                )

                u_corr = np.corrcoef(data["values"], data["u_errors"])[0, 1]
                v_corr = np.corrcoef(data["values"], data["v_errors"])[0, 1]
                f.write(f"  Correlation with u error: {u_corr:.4f}\n")
                f.write(f"  Correlation with v error: {v_corr:.4f}\n")

        print(f"Parameter statistics saved to {output_file}")

    def _create_efficiency_table(self, efficiency_data: Dict):
        output_file = Path(self.config.output_dir) / "efficiency_comparison.txt"

        with open(output_file, "w") as f:
            f.write("=" * 70 + "\n")
            f.write(" COMPUTATIONAL EFFICIENCY COMPARISON\n")
            f.write("=" * 70 + "\n\n")

            f.write("Resolution Scaling (Single Time Step):\n")
            f.write("-" * 70 + "\n")
            f.write(
                f"{'Resolution':<15} {'FNO (ms)':<15} {'FD Solver (ms)':<20} {'Speedup':<15}\n"
            )
            f.write("-" * 70 + "\n")
            for res, fno_time, fd_time in zip(
                efficiency_data["resolutions"]["resolutions"],
                efficiency_data["resolutions"]["fno_times"],
                efficiency_data["resolutions"]["fd_times"],
            ):
                speedup = fd_time / fno_time
                f.write(
                    f"{res:<15} {fno_time:<15.2f} {fd_time:<20.2f} {speedup:<15.1f}×\n"
                )

            avg_speedup = np.mean(
                [
                    fd / fno
                    for fd, fno in zip(
                        efficiency_data["resolutions"]["fd_times"],
                        efficiency_data["resolutions"]["fno_times"],
                    )
                ]
            )
            f.write("-" * 70 + "\n")
            f.write(f"Average Speedup: {avg_speedup:.1f}×\n")

            f.write("\n\nBatch Size Scaling (FNO only, nx=256):\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Batch Size':<15} {'Time/Sample (ms)':<20}\n")
            f.write("-" * 40 + "\n")
            for bs, time_ms in zip(
                efficiency_data["batch_sizes"]["batch_sizes"],
                efficiency_data["batch_sizes"]["times_per_sample"],
            ):
                f.write(f"{bs:<15} {time_ms:<20.2f}\n")

            batch_speedup = (
                efficiency_data["batch_sizes"]["times_per_sample"][0]
                / efficiency_data["batch_sizes"]["times_per_sample"][-1]
            )
            f.write("-" * 40 + "\n")
            f.write(
                f"Batching Speedup (1→{efficiency_data['batch_sizes']['batch_sizes'][-1]}): {batch_speedup:.1f}×\n"
            )

        print(f"Efficiency comparison saved to {output_file}")

    def run_full_evaluation(self):
        print("\n" + "=" * 70)
        print(" COMPREHENSIVE FHN-FNO EVALUATION SUITE")
        print("=" * 70 + "\n")

        self.load_model_and_data()

        metrics = self.evaluate_single_sample(sample_idx=0)

        self.evaluate_single_step_regimes()

        rollout_errors = self.evaluate_rollout_error_accumulation()

        param_analysis = self.evaluate_parameter_generalization()

        efficiency_data = self.evaluate_efficiency_scaling()

        self._save_evaluation_summary(
            metrics, rollout_errors, param_analysis, efficiency_data
        )

        print("\n" + "=" * 70)
        print(" EVALUATION COMPLETE")
        print("=" * 70)
        print(f"\nAll results saved to: {self.config.output_dir}")
        print("\nGenerated files:")
        print("  - 01_spatiotemporal_comparison.png")
        print("  - 02_temporal_snapshots.png")
        print("  - 03_phase_portraits.png")
        print("  - 04_error_analysis.png")
        print("  - 05_spectral_analysis.png")
        print("  - 06_metrics_summary.png")
        print("  - 07_rollout_error_evolution.png")
        print("  - 08_parameter_generalization.png")
        print("  - 09_efficiency_scaling.png")
        print("  - 10_single_step_regimes.png")
        print("  - parameter_statistics.txt")
        print("  - efficiency_comparison.txt")
        print("  - evaluation_summary.txt")
        print()

    def _save_evaluation_summary(
        self, metrics, rollout_errors, param_analysis, efficiency_data
    ):
        output_file = Path(self.config.output_dir) / "evaluation_summary.txt"

        with open(output_file, "w") as f:
            f.write("=" * 70 + "\n")
            f.write(" COMPREHENSIVE EVALUATION SUMMARY\n")
            f.write("=" * 70 + "\n\n")

            f.write("Model Configuration:\n")
            f.write(
                f"  Parameters: {sum(p.numel() for p in self.model.parameters()):,}\n"
            )
            f.write(f"  Device: {self.device}\n")
            f.write(f"  Test samples: {self.config.n_test_samples}\n")
            f.write(f"  Rollout steps: {self.config.rollout_steps}\n\n")

            f.write("Single Sample Metrics:\n")
            f.write(f"  u Rel. L2: {metrics['u']['rel_l2']:.6f}\n")
            f.write(f"  v Rel. L2: {metrics['v']['rel_l2']:.6f}\n")
            f.write(f"  u R²: {metrics['u']['r2']:.6f}\n")
            f.write(f"  v R²: {metrics['v']['r2']:.6f}\n\n")

            f.write("Rollout Error Growth:\n")
            steps_list = sorted(rollout_errors.keys())
            f.write(f"  Steps 1→{steps_list[-1]}: ")
            f.write(f"u error {rollout_errors[steps_list[0]]['u']['rel_l2']:.4f} → ")
            f.write(f"{rollout_errors[steps_list[-1]]['u']['rel_l2']:.4f}\n")

            f.write("\nParameter Generalization:\n")
            for param_name, data in param_analysis.items():
                u_corr = np.corrcoef(data["values"], data["u_errors"])[0, 1]
                f.write(f"  {param_name}: correlation = {u_corr:.4f}\n")

            f.write("\nComputational Efficiency:\n")
            f.write(
                f"  FNO at nx=256: {efficiency_data['resolutions']['fno_times'][2]:.2f} ms\n"
            )
            f.write(
                f"  FD at nx=256: {efficiency_data['resolutions']['fd_times'][2]:.2f} ms\n"
            )
            speedup_256 = (
                efficiency_data["resolutions"]["fd_times"][2]
                / efficiency_data["resolutions"]["fno_times"][2]
            )
            f.write(f"  FNO Speedup: {speedup_256:.1f}×\n")
            f.write(f"  Batch size 32 speedup: ")
            speedup = (
                efficiency_data["batch_sizes"]["times_per_sample"][0]
                / efficiency_data["batch_sizes"]["times_per_sample"][-1]
            )
            f.write(f"{speedup:.1f}×\n")

        print(f"Evaluation summary saved to {output_file}")


def main():
    config = EvaluationConfig(
        data_file="data/fhn_1d_8000.h5",
        checkpoint_file="checkpoints/best_model.pt",
        output_dir="research_outputs/",
        device="cpu",
        n_test_samples=100,
        rollout_steps=50,
    )

    evaluator = ComprehensiveEvaluator(config)

    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()
