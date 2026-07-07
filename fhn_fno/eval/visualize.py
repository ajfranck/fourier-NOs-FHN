import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys
import h5py

sys.path.append('.')

from fhn_fno.models.fno import FNO
from fhn_fno.data.dataset import FHNOperatorDataset
from fhn_fno.eval.metrics import relative_l2_error


def plot_1d_comparison(u_true: np.ndarray, v_true: np.ndarray,
                       u_pred: np.ndarray, v_pred: np.ndarray,
                       x: np.ndarray, save_path: str = None):
    """True vs predicted u and v at a single time."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes[0, 0].plot(x, u_true, 'b-', label='True', linewidth=2)
    axes[0, 0].plot(x, u_pred, 'r--', label='Predicted', linewidth=2)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('u')
    axes[0, 0].set_title('u: True vs Predicted')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(x, v_true, 'b-', label='True', linewidth=2)
    axes[0, 1].plot(x, v_pred, 'r--', label='Predicted', linewidth=2)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('v')
    axes[0, 1].set_title('v: True vs Predicted')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    u_error = np.abs(u_true - u_pred)
    axes[1, 0].plot(x, u_error, 'g-', linewidth=2)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('|Error|')
    axes[1, 0].set_title(f'u Error (max: {u_error.max():.3e})')
    axes[1, 0].grid(True, alpha=0.3)
    
    v_error = np.abs(v_true - v_pred)
    axes[1, 1].plot(x, v_error, 'g-', linewidth=2)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('|Error|')
    axes[1, 1].set_title(f'v Error (max: {v_error.max():.3e})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_error_heatmap(u_true: np.ndarray, v_true: np.ndarray,
                       u_pred: np.ndarray, v_pred: np.ndarray,
                       save_path: str = None):
    """Space-time heatmaps of truth, prediction, and error."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 1])
    
    ax1 = plt.subplot(gs[0, 0])
    im1 = ax1.imshow(u_true.T, aspect='auto', cmap='RdBu_r', origin='lower')
    ax1.set_title('u: Ground Truth')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Spatial position')
    plt.colorbar(im1, ax=ax1)
    
    ax2 = plt.subplot(gs[0, 1])
    im2 = ax2.imshow(v_true.T, aspect='auto', cmap='RdBu_r', origin='lower')
    ax2.set_title('v: Ground Truth')
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Spatial position')
    plt.colorbar(im2, ax=ax2)
    
    ax3 = plt.subplot(gs[1, 0])
    im3 = ax3.imshow(u_pred.T, aspect='auto', cmap='RdBu_r', origin='lower')
    ax3.set_title('u: Prediction')
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Spatial position')
    plt.colorbar(im3, ax=ax3)
    
    ax4 = plt.subplot(gs[1, 1])
    im4 = ax4.imshow(v_pred.T, aspect='auto', cmap='RdBu_r', origin='lower')
    ax4.set_title('v: Prediction')
    ax4.set_xlabel('Time step')
    ax4.set_ylabel('Spatial position')
    plt.colorbar(im4, ax=ax4)
    
    u_error = np.abs(u_true - u_pred)
    v_error = np.abs(v_true - v_pred)
    
    ax5 = plt.subplot(gs[2, 0])
    im5 = ax5.imshow(u_error.T, aspect='auto', cmap='hot', origin='lower')
    ax5.set_title(f'u: Absolute Error (max: {u_error.max():.3e})')
    ax5.set_xlabel('Time step')
    ax5.set_ylabel('Spatial position')
    plt.colorbar(im5, ax=ax5)
    
    ax6 = plt.subplot(gs[2, 1])
    im6 = ax6.imshow(v_error.T, aspect='auto', cmap='hot', origin='lower')
    ax6.set_title(f'v: Absolute Error (max: {v_error.max():.3e})')
    ax6.set_xlabel('Time step')
    ax6.set_ylabel('Spatial position')
    plt.colorbar(im6, ax=ax6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_phase_portrait(u_true: np.ndarray, v_true: np.ndarray,
                        u_pred: np.ndarray, v_pred: np.ndarray,
                        spatial_points: list = None, save_path: str = None):
    """u-v phase portraits at selected spatial points."""
    n_times, nx = u_true.shape

    if spatial_points is None:
        spatial_points = [nx//4, nx//2, 3*nx//4, nx-1]
    
    n_points = len(spatial_points)
    fig, axes = plt.subplots(1, n_points, figsize=(4*n_points, 4))
    
    if n_points == 1:
        axes = [axes]
    
    for i, pt in enumerate(spatial_points):
        ax = axes[i]
        
        ax.plot(u_true[:, pt], v_true[:, pt], 'b-', label='True', linewidth=2, alpha=0.7)
        ax.plot(u_pred[:, pt], v_pred[:, pt], 'r--', label='Predicted', linewidth=2, alpha=0.7)
        
        ax.scatter(u_true[0, pt], v_true[0, pt], c='green', s=100, marker='o', 
                  label='Start', zorder=5)
        ax.scatter(u_true[-1, pt], v_true[-1, pt], c='blue', s=100, marker='s', zorder=5)
        ax.scatter(u_pred[-1, pt], v_pred[-1, pt], c='red', s=100, marker='^', zorder=5)
        
        ax.set_xlabel('u')
        ax.set_ylabel('v')
        ax.set_title(f'Phase Portrait at x={pt}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_rollout(model: nn.Module, dataset: FHNOperatorDataset, 
                     sample_idx: int = 0, n_steps: int = 50,
                     device: str = 'cpu', output_dir: str = 'outputs'):
    """Roll out the model on one sample and save comparison plots."""
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(dataset.data_file, 'r') as f:
        u_traj_true = f['u_traj'][sample_idx][:n_steps+1]
        v_traj_true = f['v_traj'][sample_idx][:n_steps+1]
    
    u0 = u_traj_true[0]
    v0 = v_traj_true[0]
    
    if dataset.normalize:
        u0_norm = (u0 - dataset.u_mean) / dataset.u_std
        v0_norm = (v0 - dataset.v_mean) / dataset.v_std
    else:
        u0_norm = u0
        v0_norm = v0
    
    x0 = torch.tensor(np.stack([u0_norm, v0_norm], axis=0), 
                      dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        trajectory = model.rollout(x0, n_steps)  # (1, n_steps+1, 2, nx)
    
    u_traj_pred = trajectory[0, :, 0].cpu().numpy()
    v_traj_pred = trajectory[0, :, 1].cpu().numpy()
    
    if dataset.normalize:
        u_traj_pred = u_traj_pred * dataset.u_std + dataset.u_mean
        v_traj_pred = v_traj_pred * dataset.v_std + dataset.v_mean
    
    nx = u_traj_true.shape[-1]
    x = np.linspace(0, 1, nx)
    
    print(f"Generating visualizations for sample {sample_idx}...")
    
    plot_1d_comparison(
        u_traj_true[-1], v_traj_true[-1],
        u_traj_pred[-1], v_traj_pred[-1],
        x, save_path=output_dir / f'comparison_t{n_steps}.png'
    )
    
    plot_error_heatmap(
        u_traj_true, v_traj_true,
        u_traj_pred, v_traj_pred,
        save_path=output_dir / 'error_heatmap.png'
    )
    
    plot_phase_portrait(
        u_traj_true, v_traj_true,
        u_traj_pred, v_traj_pred,
        save_path=output_dir / 'phase_portraits.png'
    )
    
    u_rel_error = relative_l2_error(
        torch.tensor(u_traj_pred), torch.tensor(u_traj_true)
    )
    v_rel_error = relative_l2_error(
        torch.tensor(v_traj_pred), torch.tensor(v_traj_true)
    )
    
    print(f"Relative L2 errors - u: {u_rel_error:.4f}, v: {v_rel_error:.4f}")
    print(f"Visualizations saved to {output_dir}")


def main():
    # edit these directly, no CLI args
    DATA_FILE = "data/fhn_1d_tiny.h5"
    CHECKPOINT_FILE = "checkpoints/best_model.pt"
    SAMPLE_IDX = 0
    N_STEPS = 50
    DEVICE = "cpu"
    OUTPUT_DIR = "outputs/"
    
    dataset = FHNOperatorDataset(DATA_FILE, mode="single_step", train=False)
    
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    config = checkpoint.get('config', {})
    
    model = FNO(
        modes=config.get('model', {}).get('modes', 16),
        width=config.get('model', {}).get('width', 64),
        n_layers=config.get('model', {}).get('n_layers', 4),
        dim=dataset.dim
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    
    # visualize_rollout reads this off the dataset
    dataset.data_file = DATA_FILE

    visualize_rollout(model, dataset, SAMPLE_IDX, N_STEPS,
                     DEVICE, OUTPUT_DIR)


if __name__ == "__main__":
    main()