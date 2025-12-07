import numpy as np
import h5py
from typing import Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
from tqdm import tqdm
import sys

sys.path.append(".")

from fhn_fno.config import FHNParams, DataConfig


class PDESolver(ABC):

    @abstractmethod
    def solve(
        self,
        u0: np.ndarray,
        v0: np.ndarray,
        params: FHNParams,
        T: float,
        dt: float,
        n_save: int,
        I_ext: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        pass


class FDBackend(PDESolver):

    def __init__(
        self, nx: int, ny: Optional[int] = None, dx: float = 1.0, dy: float = 1.0
    ):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dim = 1 if ny is None else 2

    def solve(
        self,
        u0: np.ndarray,
        v0: np.ndarray,
        params: FHNParams,
        T: float,
        dt: float,
        n_save: int,
        I_ext: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        u, v = u0.copy(), v0.copy()
        n_steps = int(T / dt)
        save_interval = max(1, n_steps // n_save)

        u_hist = [u0.copy()]
        v_hist = [v0.copy()]
        times = [0.0]

        if self.dim == 1:
            lap = self._laplacian_1d(self.nx, self.dx)
        else:
            lap = self._laplacian_2d(self.nx, self.ny, self.dx, self.dy)

        for step in range(n_steps):
            if self.dim == 1:
                u_new = self._step_1d(u, v, params.Du, lap, dt, I_ext)
                v_new = self._step_1d_v(
                    u, v, params.Dv, params.a, params.b, params.tau, lap, dt
                )
            else:
                u_flat = u.flatten()
                v_flat = v.flatten()
                I_flat = I_ext.flatten() if I_ext is not None else None

                u_new_flat = self._step_2d(u_flat, v_flat, params.Du, lap, dt, I_flat)
                v_new_flat = self._step_2d_v(
                    u_flat, v_flat, params.Dv, params.a, params.b, params.tau, lap, dt
                )

                u_new = u_new_flat.reshape(u.shape)
                v_new = v_new_flat.reshape(v.shape)

            u, v = u_new, v_new

            if (step + 1) % save_interval == 0:
                u_hist.append(u.copy())
                v_hist.append(v.copy())
                times.append((step + 1) * dt)

        return {"u": np.array(u_hist), "v": np.array(v_hist), "t": np.array(times)}

    def _laplacian_1d(self, n: int, dx: float) -> np.ndarray:
        lap = np.zeros((n, n))
        c = 1.0 / (dx * dx)

        for i in range(n):
            lap[i, i] = -2 * c
            lap[i, (i + 1) % n] = c
            lap[i, (i - 1) % n] = c

        return lap

    def _laplacian_2d(self, nx: int, ny: int, dx: float, dy: float) -> np.ndarray:
        n = nx * ny
        lap = np.zeros((n, n))
        cx = 1.0 / (dx * dx)
        cy = 1.0 / (dy * dy)

        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                lap[idx, idx] = -2 * (cx + cy)

                idx_right = ((i + 1) % nx) * ny + j
                idx_left = ((i - 1) % nx) * ny + j
                lap[idx, idx_right] = cx
                lap[idx, idx_left] = cx

                idx_up = i * ny + ((j + 1) % ny)
                idx_down = i * ny + ((j - 1) % ny)
                lap[idx, idx_up] = cy
                lap[idx, idx_down] = cy

        return lap

    def _step_1d(
        self,
        u: np.ndarray,
        v: np.ndarray,
        Du: float,
        lap: np.ndarray,
        dt: float,
        I_ext: Optional[np.ndarray],
    ) -> np.ndarray:
        reaction = u - u**3 / 3 - v
        if I_ext is not None:
            reaction += I_ext

        A = np.eye(len(u)) - dt * Du * lap
        b = u + dt * reaction
        u_new = np.linalg.solve(A, b)

        return u_new

    def _step_1d_v(
        self,
        u: np.ndarray,
        v: np.ndarray,
        Dv: float,
        a: float,
        b: float,
        tau: float,
        lap: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        reaction = (u + a - b * v) / tau

        A = np.eye(len(v)) - dt * Dv * lap
        b_vec = v + dt * reaction
        v_new = np.linalg.solve(A, b_vec)

        return v_new

    def _step_2d(
        self,
        u: np.ndarray,
        v: np.ndarray,
        Du: float,
        lap: np.ndarray,
        dt: float,
        I_ext: Optional[np.ndarray],
    ) -> np.ndarray:
        reaction = u - u**3 / 3 - v
        if I_ext is not None:
            reaction += I_ext

        A = np.eye(len(u)) - dt * Du * lap
        b = u + dt * reaction
        u_new = np.linalg.solve(A, b)

        return u_new

    def _step_2d_v(
        self,
        u: np.ndarray,
        v: np.ndarray,
        Dv: float,
        a: float,
        b: float,
        tau: float,
        lap: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        reaction = (u + a - b * v) / tau

        A = np.eye(len(v)) - dt * Dv * lap
        b_vec = v + dt * reaction
        v_new = np.linalg.solve(A, b_vec)

        return v_new


class DedalusBackend(PDESolver):
    # ended up not needing this / overdoing it
    # dont mess with it

    def __init__(self, nx: int, ny: Optional[int] = None):
        self.nx = nx
        self.ny = ny
        raise NotImplementedError(
            "Dedalus backend requires dedalus package. Use --backend=fallback"
        )

    def solve(
        self,
        u0: np.ndarray,
        v0: np.ndarray,
        params: FHNParams,
        T: float,
        dt: float,
        n_save: int,
        I_ext: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        pass


def sample_initial_conditions(
    nx: int,
    ny: Optional[int],
    ic_type: str = "grf",
    alpha: float = 2.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)

    if ic_type == "grf":
        if ny is None:
            k = np.fft.fftfreq(nx, d=1.0 / nx)
            k[0] = 1e-10

            power = (1 + np.abs(k) ** 2) ** (-alpha / 2)
            u_hat = np.random.randn(nx) + 1j * np.random.randn(nx)
            u_hat *= np.sqrt(power)

            v_hat = np.random.randn(nx) + 1j * np.random.randn(nx)
            v_hat *= np.sqrt(power) * 0.5

            u0 = np.real(np.fft.ifft(u_hat))
            v0 = np.real(np.fft.ifft(v_hat))

        else:  # 2d
            kx = np.fft.fftfreq(nx, d=1.0 / nx)
            ky = np.fft.fftfreq(ny, d=1.0 / ny)
            kx, ky = np.meshgrid(kx, ky, indexing="ij")
            k2 = kx**2 + ky**2
            k2[0, 0] = 1e-10

            power = (1 + k2) ** (-alpha / 2)
            u_hat = np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)
            u_hat *= np.sqrt(power)

            v_hat = np.random.randn(nx, ny) + 1j * np.random.randn(nx, ny)
            v_hat *= np.sqrt(power) * 0.5

            u0 = np.real(np.fft.ifft2(u_hat))
            v0 = np.real(np.fft.ifft2(v_hat))

    elif ic_type == "gaussian":
        # gauss bump
        if ny is None:
            x = np.linspace(0, 1, nx, endpoint=False)
            x0 = np.random.uniform(0.3, 0.7)
            sigma = np.random.uniform(0.05, 0.15)
            u0 = np.exp(-((x - x0) ** 2) / (2 * sigma**2))
            v0 = 0.5 * u0 + 0.1 * np.random.randn(nx)
        else:
            x = np.linspace(0, 1, nx, endpoint=False)
            y = np.linspace(0, 1, ny, endpoint=False)
            xx, yy = np.meshgrid(x, y, indexing="ij")
            x0, y0 = np.random.uniform(0.3, 0.7, 2)
            sigma = np.random.uniform(0.05, 0.15)
            u0 = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma**2))
            v0 = 0.5 * u0 + 0.1 * np.random.randn(nx, ny)

    elif ic_type == "sinusoidal":
        # sin perturb
        if ny is None:
            x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
            k1, k2 = np.random.randint(1, 5, 2)
            u0 = 0.5 * (np.sin(k1 * x) + np.sin(k2 * x))
            v0 = 0.3 * np.cos(k1 * x)
        else:
            x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
            y = np.linspace(0, 2 * np.pi, ny, endpoint=False)
            xx, yy = np.meshgrid(x, y, indexing="ij")
            kx, ky = np.random.randint(1, 4, 2)
            u0 = 0.5 * np.sin(kx * xx) * np.cos(ky * yy)
            v0 = 0.3 * np.cos(kx * xx) * np.sin(ky * yy)

    else:
        raise ValueError(f"Unknown IC type: {ic_type}")

    u0 = (u0 - np.mean(u0)) / (np.std(u0) + 1e-8)
    v0 = (v0 - np.mean(v0)) / (np.std(v0) + 1e-8)

    return u0, v0


def sample_parameters(config: DataConfig, seed: Optional[int] = None) -> FHNParams:
    if seed is not None:
        np.random.seed(seed)

    return FHNParams(
        Du=np.random.uniform(*config.Du_range),
        Dv=np.random.uniform(*config.Dv_range),
        a=np.random.uniform(*config.a_range),
        b=np.random.uniform(*config.b_range),
        tau=np.random.uniform(*config.tau_range),
    )


def generate_dataset(
    config: DataConfig,
    n_samples: int,
    backend: str = "fallback",
    output_file: str = "fhn_data.h5",
    verbose: bool = True,
):
    if backend == "fallback":
        solver = FDBackend(
            config.nx,
            config.ny,
            dx=1.0 / config.nx,
            dy=1.0 / config.ny if config.ny else 1.0,
        )
    elif backend == "dedalus":
        solver = DedalusBackend(config.nx, config.ny)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    all_u0, all_v0 = [], []
    all_u_traj, all_v_traj = [], []
    all_params = []
    all_I_ext = []

    iterator = (
        tqdm(range(n_samples), desc="Generating samples")
        if verbose
        else range(n_samples)
    )

    for i in iterator:
        u0, v0 = sample_initial_conditions(
            config.nx, config.ny, config.ic_type, config.grf_alpha, seed=i
        )

        params = sample_parameters(config, seed=i * 100)

        I_ext = None
        if config.use_stimulus:
            if config.ny is None:
                I_ext = config.stimulus_amplitude * np.random.randn(config.nx)
            else:
                I_ext = config.stimulus_amplitude * np.random.randn(
                    config.nx, config.ny
                )

        if config.noise_std > 0:
            u0 += config.noise_std * np.random.randn(*u0.shape)
            v0 += config.noise_std * np.random.randn(*v0.shape)

        sol = solver.solve(
            u0, v0, params, config.T, config.dt, config.n_timesteps, I_ext
        )

        all_u0.append(u0)
        all_v0.append(v0)
        all_u_traj.append(sol["u"])
        all_v_traj.append(sol["v"])
        all_params.append([params.Du, params.Dv, params.a, params.b, params.tau])
        all_I_ext.append(I_ext if I_ext is not None else np.zeros_like(u0))

    with h5py.File(output_file, "w") as f:
        f.create_dataset("u0", data=np.array(all_u0))
        f.create_dataset("v0", data=np.array(all_v0))
        f.create_dataset("u_traj", data=np.array(all_u_traj))
        f.create_dataset("v_traj", data=np.array(all_v_traj))
        f.create_dataset("params", data=np.array(all_params))
        f.create_dataset("I_ext", data=np.array(all_I_ext))
        f.create_dataset("times", data=sol["t"])


def main():
    BACKEND = "fallback"
    DIM = 1
    N_SAMPLES = 8000
    NX = 256
    NY = None
    T = 1.0
    DT = 0.01
    N_TIMESTEPS = 50
    IC_TYPE = "grf"
    USE_STIMULUS = False
    OUTPUT_FILE = "data/fhn_1d_stim.h5"
    SEED = 42

    config = DataConfig(
        dim=DIM,
        nx=NX,
        ny=NY if DIM == 2 else None,
        T=T,
        dt=DT,
        n_timesteps=N_TIMESTEPS,
        ic_type=IC_TYPE,
        use_stimulus=USE_STIMULUS,
    )

    np.random.seed(SEED)

    print(f"Generating {N_SAMPLES} samples...")
    print(f"Configuration: {DIM}D, nx={NX}, T={T}, backend={BACKEND}")

    import os

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    generate_dataset(
        config=config,
        n_samples=N_SAMPLES,
        backend=BACKEND,
        output_file=OUTPUT_FILE,
        verbose=True,
    )

    print(f"Dataset saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
