import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from typing import Optional, Tuple, Dict


class FHNOperatorDataset(Dataset):

    def __init__(
        self,
        data_file: str,
        mode: str = "single_step",
        normalize: bool = True,
        train: bool = True,
        train_split: float = 0.8,
        device: str = "cpu",
    ):
        self.mode = mode
        self.normalize = normalize
        self.device = device

        with h5py.File(data_file, "r") as f:
            self.u_traj = np.array(
                f["u_traj"]
            )  # (n_samples, n_times, nx) or (n_samples, n_times, nx, ny)
            self.v_traj = np.array(f["v_traj"])
            self.params = np.array(f["params"])  # (n_samples, 5)
            self.I_ext = np.array(f["I_ext"])  # (n_samples, nx) or (n_samples, nx, ny)
            self.times = np.array(f["times"])

        n_samples = len(self.u_traj)
        n_train = int(n_samples * train_split)

        if train:
            indices = np.arange(n_train)
        else:
            indices = np.arange(n_train, n_samples)

        self.u_traj = self.u_traj[indices]
        self.v_traj = self.v_traj[indices]
        self.params = self.params[indices]
        self.I_ext = self.I_ext[indices]

        if normalize and train:
            self.u_mean = np.mean(self.u_traj)
            self.u_std = np.std(self.u_traj) + 1e-8
            self.v_mean = np.mean(self.v_traj)
            self.v_std = np.std(self.v_traj) + 1e-8
        elif normalize:
            self.u_mean = 0.0
            self.u_std = 1.0
            self.v_mean = 0.0
            self.v_std = 1.0
        else:
            self.u_mean = 0.0
            self.u_std = 1.0
            self.v_mean = 0.0
            self.v_std = 1.0

        # dimensions
        self.n_samples = len(self.u_traj)
        self.n_times = self.u_traj.shape[1]
        self.spatial_shape = self.u_traj.shape[2:]
        self.dim = len(self.spatial_shape)

    def __len__(self) -> int:
        if self.mode == "single_step":
            return self.n_samples * (self.n_times - 1)
        else:
            return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.mode == "single_step":

            sample_idx = idx // (self.n_times - 1)
            time_idx = idx % (self.n_times - 1)

            u_in = self.u_traj[sample_idx, time_idx]
            v_in = self.v_traj[sample_idx, time_idx]
            u_out = self.u_traj[sample_idx, time_idx + 1]
            v_out = self.v_traj[sample_idx, time_idx + 1]

        else:
            sample_idx = idx

            # First and last time steps
            u_in = self.u_traj[sample_idx, 0]
            v_in = self.v_traj[sample_idx, 0]
            u_out = self.u_traj[sample_idx, -1]
            v_out = self.v_traj[sample_idx, -1]

        if self.normalize:
            u_in = (u_in - self.u_mean) / self.u_std
            v_in = (v_in - self.v_mean) / self.v_std
            u_out = (u_out - self.u_mean) / self.u_std
            v_out = (v_out - self.v_mean) / self.v_std

        # stack (2, *spatial_shape)
        x = np.stack([u_in, v_in], axis=0)
        y = np.stack([u_out, v_out], axis=0)

        #! optional include external stimulus and parameters
        params = self.params[sample_idx]
        I_ext = self.I_ext[sample_idx]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        params = torch.tensor(params, dtype=torch.float32)
        I_ext = torch.tensor(I_ext, dtype=torch.float32)

        return {"input": x, "target": y, "params": params, "I_ext": I_ext}

    def denormalize(
        self, u: torch.Tensor, v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.normalize:
            u = u * self.u_std + self.u_mean
            v = v * self.v_std + self.v_mean
        return u, v
