from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import torch


@dataclass
class FHNParams:
    Du: float = 0.05
    Dv: float = 0.01
    a: float = 0.0
    b: float = 0.2
    tau: float = 5.0        # recovery time scale


@dataclass
class DataConfig:
    dim: int = 1
    nx: int = 256
    ny: Optional[int] = None  # 2D only
    T: float = 5.0
    dt: float = 0.01
    n_timesteps: int = 100

    ic_type: str = "grf"  # "grf", "gaussian", "sinusoidal"
    grf_alpha: float = 2.0  # GRF spectral decay

    Du_range: Tuple[float, float] = (0.01, 0.1)
    Dv_range: Tuple[float, float] = (0.005, 0.05)
    a_range: Tuple[float, float] = (-0.1, 0.1)
    b_range: Tuple[float, float] = (0.1, 0.5)
    tau_range: Tuple[float, float] = (1.0, 20.0)

    use_stimulus: bool = False
    stimulus_amplitude: float = 0.1

    noise_std: float = 0.0


@dataclass
class FNOConfig:
    modes: int = 16         # k_max
    width: int = 64
    n_layers: int = 6
    activation: str = "gelu"

    in_channels: int = 2    # u, v
    out_channels: int = 2

    use_positional: bool = False
    spectral_norm: bool = False


@dataclass
class TrainingConfig:
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 1000
    gradient_clip: float = 1.0

    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # "cosine", "step"

    loss_type: str = "l2"  # "l2", "mse", "combined"
    u_weight: float = 1.0
    v_weight: float = 1.0

    # physics-informed loss, unused for now
    use_pde_loss: bool = False
    pde_weight: float = 0.1

    use_amp: bool = True
    precision: str = "32"  # "16", "32", "bf16"
    num_workers: int = 4
    pin_memory: bool = True

    checkpoint_dir: str = "checkpoints"
    save_every: int = 50
    early_stop_patience: int = 20

    log_every: int = 10
    val_every: int = 20

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    fhn: FHNParams = field(default_factory=FHNParams)
    data: DataConfig = field(default_factory=DataConfig)
    model: FNOConfig = field(default_factory=FNOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    seed: int = 42
    experiment_name: str = "fhn_fno_baseline"

    def to_dict(self) -> dict:
        return {
            "fhn": self.fhn.__dict__,
            "data": self.data.__dict__,
            "model": self.model.__dict__,
            "training": self.training.__dict__,
            "seed": self.seed,
            "experiment_name": self.experiment_name
        }