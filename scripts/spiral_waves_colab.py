"""2D FitzHugh-Nagumo spiral waves and a FiLM-conditioned 2D FNO surrogate, runnable on Colab.

Split on the "# %%" markers to run as notebook cells."""

# %% imports and config
import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class SpiralConfig:
    nx: int = 256
    L: float = 120.0              # physical side length, no-flux boundaries
    dt: float = 0.05
    # reaction params tuned for a single rest state and a rigidly rotating spiral
    # (too-strong recovery retracts it, too-weak front breaks it up)
    a: float = 0.70
    b: float = 0.50
    tau: float = 12.5
    Du: float = 1.0
    Dv: float = 0.30              # inhibitor diffusion, smooths the front to prevent breakup
    u_exc: float = 1.0            # excited activator value in the left half
    dv_refr: float = 0.70         # recovery offset making the bottom half refractory
    total_steps: int = 4000       # ~1-2 spiral rotations
    save_every: int = 40


def rest_state(a: float, b: float):
    """Rest state: most-negative real root of b u^3 - 3(b-1)u + 3a = 0, with v* = (u*+a)/b."""
    roots = np.roots([b, 0.0, -3.0 * (b - 1.0), 3.0 * a])
    real = sorted(float(r.real) for r in roots if abs(r.imag) < 1e-8)
    u_star = real[0]
    return u_star, (u_star + a) / b


# %% 2D FHN solver
import scipy.sparse as sp
from scipy.sparse.linalg import splu


def _neumann_lap_1d(n, dx):
    """1D second-difference matrix with no-flux (Neumann) boundaries."""
    main = -2.0 * np.ones(n)
    off = np.ones(n - 1)
    L = sp.diags([off, main, off], [-1, 0, 1], format="lil")
    L[0, 1] = 2.0          # ghost point u_{-1} = u_1
    L[-1, -2] = 2.0        # ghost point u_{n} = u_{n-2}
    return (L.tocsc()) / dx**2


class FHN2DSolver:
    """IMEX stepping: explicit reaction, implicit diffusion via an ADI split into
    two tridiagonal Neumann solves. Unconditionally stable and O(N) per step, and
    unlike an FFT solver it keeps a single seeded spiral from tiling periodically."""

    def __init__(self, cfg: SpiralConfig):
        self.cfg = cfg
        n = cfg.nx
        dx = cfg.L / n
        Lap = _neumann_lap_1d(n, dx)
        I = sp.identity(n, format="csc")
        # factorize the 1D half-step operators once, reused for every row and column
        self.lu_u = splu((I - cfg.dt * cfg.Du * Lap).tocsc())
        self.lu_v = splu((I - cfg.dt * cfg.Dv * Lap).tocsc()) if cfg.Dv > 0 else None
        self.u_star, self.v_star = rest_state(cfg.a, cfg.b)

    @staticmethod
    def _adi(lu, f):
        """Approximate (I - dt D L2D)^{-1} f by sweeping x then y."""
        if lu is None:
            return f
        h = lu.solve(f)            # solve along axis 0
        return lu.solve(h.T).T     # solve along axis 1

    def step(self, u, v, I_ext=0.0):
        c = self.cfg
        u1 = u + c.dt * (u - u**3 / 3.0 - v + I_ext)
        v1 = v + c.dt * (u + c.a - c.b * v) / c.tau
        u_new = self._adi(self.lu_u, u1)
        v_new = self._adi(self.lu_v, v1)
        return u_new, v_new

    def make_spiral(self, verbose=True):
        c = self.cfg
        n = c.nx
        cx, cy = n // 2, n // 2
        # Two perpendicular phase steps: left half excited, bottom half refractory.
        # The four quadrants then hold four phases of the AP cycle around the centre,
        # so the broken front curls into one rotating spiral.
        u = np.full((n, n), self.u_star)
        u[:cx, :] = c.u_exc
        v = np.full((n, n), self.v_star)
        v[:, :cy] = self.v_star + c.dv_refr

        frames_u, frames_v = [u.copy()], [v.copy()]
        for i in range(c.total_steps):
            u, v = self.step(u, v, 0.0)
            if (i + 1) % c.save_every == 0:
                frames_u.append(u.copy())
                frames_v.append(v.copy())
        if verbose:
            print(f"Generated spiral: {len(frames_u)} frames, "
                  f"u in [{u.min():.2f}, {u.max():.2f}]")
        return np.stack(frames_u), np.stack(frames_v)


# %% plotting
def plot_spiral_snapshots(frames_u, out_path, n_show=6):
    idx = np.linspace(0, len(frames_u) - 1, n_show).astype(int)
    fig, axes = plt.subplots(1, n_show, figsize=(2.4 * n_show, 2.6))
    for ax, k in zip(axes, idx):
        ax.imshow(frames_u[k], cmap="RdBu_r", vmin=-2, vmax=2, origin="lower")
        ax.set_title(f"step {k}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("FitzHugh-Nagumo spiral wave: activator $u(x,y,t)$", y=1.02)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def animate_spiral(frames_u, out_path, fps=20):
    try:
        import imageio.v2 as imageio
    except Exception:
        print("imageio not available; skipping GIF (run `pip install imageio`).")
        return
    vmin, vmax = -2.0, 2.0
    imgs = []
    cmap = plt.get_cmap("RdBu_r")
    for f in frames_u:
        norm = np.clip((f - vmin) / (vmax - vmin), 0, 1)
        imgs.append((cmap(norm)[..., :3] * 255).astype(np.uint8))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, imgs, fps=fps)
    print(f"Saved {out_path}")


# %% dataset generation
@dataclass
class DataGenConfig:
    n_trajectories: int = 200
    nx: int = 128                 # coarser than the demo for memory and throughput
    L: float = 120.0
    dt: float = 0.05
    save_every: int = 48
    total_steps: int = 2400       # ~50 saved frames
    # ranges that keep a single rest state with rigid rotation
    Du_range: tuple = (0.8, 1.3)
    Dv_range: tuple = (0.20, 0.40)
    a_range: tuple = (0.60, 0.80)
    b_range: tuple = (0.45, 0.60)
    tau_range: tuple = (10.0, 16.0)
    seed: int = 42


def generate_dataset(dg: DataGenConfig, out_dir="research_outputs/spiral_data"):
    rng = np.random.default_rng(dg.seed)
    trajs, lambdas = [], []
    t0 = time.time()
    for i in range(dg.n_trajectories):
        cfg = SpiralConfig(
            nx=dg.nx, L=dg.L, dt=dg.dt, save_every=dg.save_every,
            total_steps=dg.total_steps,
            Du=float(rng.uniform(*dg.Du_range)),
            Dv=float(rng.uniform(*dg.Dv_range)),
            a=float(rng.uniform(*dg.a_range)),
            b=float(rng.uniform(*dg.b_range)),
            tau=float(rng.uniform(*dg.tau_range)),
        )
        solver = FHN2DSolver(cfg)
        fu, fv = solver.make_spiral(verbose=False)
        trajs.append(np.stack([fu, fv], axis=1).astype(np.float32))  # (T, 2, H, W)
        lambdas.append([cfg.Du, cfg.Dv, cfg.a, cfg.b, cfg.tau])
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{dg.n_trajectories} "
                  f"({(time.time()-t0)/(i+1):.1f}s/traj)")
    trajs = np.stack(trajs)                       # (N, T, 2, H, W)
    lambdas = np.asarray(lambdas, dtype=np.float32)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "spiral_dataset.npz",
                        trajectories=trajs, params=lambdas)
    print(f"Saved dataset {trajs.shape} -> {out/'spiral_dataset.npz'}")
    return trajs, lambdas


# %% 2D FNO
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    """2D spectral convolution retaining the first (m1, m2) Fourier modes."""

    def __init__(self, in_ch, out_ch, m1, m2):
        super().__init__()
        self.m1, self.m2 = m1, m2
        scale = 1.0 / (in_ch * out_ch)
        # two corner blocks of the rFFT spectrum, for low +kx and -kx
        self.w1 = nn.Parameter(scale * torch.randn(in_ch, out_ch, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(in_ch, out_ch, m1, m2, dtype=torch.cfloat))

    def _mul(self, x, w):
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        b, c, h, w = x.shape
        xf = torch.fft.rfft2(x, norm="ortho")
        out = torch.zeros(b, self.w1.shape[1], h, w // 2 + 1,
                          dtype=torch.cfloat, device=x.device)
        out[:, :, : self.m1, : self.m2] = self._mul(xf[:, :, : self.m1, : self.m2], self.w1)
        out[:, :, -self.m1:, : self.m2] = self._mul(xf[:, :, -self.m1:, : self.m2], self.w2)
        return torch.fft.irfft2(out, s=(h, w), norm="ortho")


class FiLMEncoder(nn.Module):
    """Maps lambda=(Du,Dv,a,b,tau) -> per-layer (gamma, beta) modulation."""

    def __init__(self, n_params, width, n_layers):
        super().__init__()
        self.width, self.n_layers = width, n_layers
        self.net = nn.Sequential(
            nn.Linear(n_params, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 2 * width * n_layers),
        )

    def forward(self, lam):
        h = self.net(lam).view(-1, self.n_layers, 2, self.width)
        return h[:, :, 0], h[:, :, 1]            # gamma, beta, each (B, L, width)


class FiLMFNO2d(nn.Module):
    def __init__(self, n_params=5, width=32, modes=12, n_layers=4, in_ch=2, out_ch=2):
        super().__init__()
        self.width, self.n_layers = width, n_layers
        self.lift = nn.Conv2d(in_ch + 2, width, 1)         # +2 for the coord channels
        self.spectral = nn.ModuleList(
            [SpectralConv2d(width, width, modes, modes) for _ in range(n_layers)])
        self.local = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.encoder = FiLMEncoder(n_params, width, n_layers)
        self.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(),
                                  nn.Conv2d(128, out_ch, 1))

    def _coords(self, x):
        b, _, h, w = x.shape
        ys = torch.linspace(0, 1, h, device=x.device)
        xs = torch.linspace(0, 1, w, device=x.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        g = torch.stack([gx, gy])[None].expand(b, -1, -1, -1)
        return g

    def forward(self, x, lam):
        x = torch.cat([x, self._coords(x)], dim=1)
        z = self.lift(x)
        gamma, beta = self.encoder(lam)
        for i in range(self.n_layers):
            zs = self.spectral[i](z) + self.local[i](z)
            g = 1.0 + gamma[:, i][:, :, None, None]
            bta = beta[:, i][:, :, None, None]
            z = F.gelu(g * zs + bta)
        return self.proj(z)


# %% training and rollout
def make_pairs(trajs):
    """(N,T,2,H,W) -> consecutive-frame pairs (inputs, targets, traj_index)."""
    N, T = trajs.shape[:2]
    x = trajs[:, :-1].reshape(-1, *trajs.shape[2:])
    y = trajs[:, 1:].reshape(-1, *trajs.shape[2:])
    idx = np.repeat(np.arange(N), T - 1)
    return x, y, idx


def train_fno(trajs, lambdas, epochs=30, batch=16, lr=1e-3, val_frac=0.2, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # per-channel normalization
    mu = trajs.mean((0, 1, 3, 4), keepdims=True)
    sd = trajs.std((0, 1, 3, 4), keepdims=True) + 1e-6
    trajs_n = (trajs - mu) / sd
    lam_mu, lam_sd = lambdas.mean(0), lambdas.std(0) + 1e-6
    lam_n = (lambdas - lam_mu) / lam_sd

    N = trajs.shape[0]
    n_val = max(1, int(val_frac * N))
    tr, va = slice(n_val, N), slice(0, n_val)

    xt, yt, it = make_pairs(trajs_n[tr]); lt = lam_n[tr]
    xv, yv, iv = make_pairs(trajs_n[va]); lv = lam_n[va]
    to_t = lambda z: torch.tensor(z, dtype=torch.float32)
    xt, yt = to_t(xt).to(device), to_t(yt).to(device)
    lt_full = to_t(lt[it]).to(device)
    xv, yv = to_t(xv).to(device), to_t(yv).to(device)
    lv_full = to_t(lv[iv]).to(device)

    model = FiLMFNO2d().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = xt.shape[0]

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for s in range(0, n, batch):
            j = perm[s:s + batch]
            opt.zero_grad()
            pred = model(xt[j], lt_full[j])
            loss = F.mse_loss(pred, yt[j])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(j)
        sched.step()
        if (ep + 1) % 5 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                pv = model(xv, lv_full)
                rel = (torch.linalg.norm((pv - yv).reshape(len(pv), -1), dim=1) /
                       torch.linalg.norm(yv.reshape(len(pv), -1), dim=1)).mean().item()
            print(f"epoch {ep+1:3d}  train MSE {tot/n:.3e}  val relL2 {rel:.3e}")

    return model, (mu, sd, lam_mu, lam_sd)


def rollout(model, traj, lam, norm, steps=40, device=None):
    """Autoregressive rollout from frame 0; returns predicted (steps,2,H,W)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mu, sd, lam_mu, lam_sd = norm
    model.eval()
    x = torch.tensor((traj[0:1] - mu[0]) / sd[0], dtype=torch.float32, device=device)
    lam_n = torch.tensor((lam[None] - lam_mu) / lam_sd, dtype=torch.float32, device=device)
    preds = []
    with torch.no_grad():
        for _ in range(steps):
            x = model(x, lam_n)
            preds.append(x.cpu().numpy()[0])
    preds = np.stack(preds) * sd[0] + mu[0]
    return preds


# %% entry point
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="generate + animate one spiral")
    ap.add_argument("--generate", action="store_true", help="build the FNO dataset")
    ap.add_argument("--train", action="store_true", help="train the 2D FiLM-FNO")
    ap.add_argument("--n", type=int, default=200, help="trajectories for --generate")
    ap.add_argument("--out", type=str, default="research_outputs/spiral_waves")
    args = ap.parse_args()
    if not (args.demo or args.generate or args.train):
        args.demo = True

    out = Path(args.out)

    if args.demo:
        cfg = SpiralConfig()
        solver = FHN2DSolver(cfg)
        t0 = time.time()
        fu, fv = solver.make_spiral()
        print(f"Solver: {time.time()-t0:.1f}s for {cfg.total_steps} steps")
        plot_spiral_snapshots(fu, out / "spiral_snapshots.png")
        animate_spiral(fu, out / "spiral.gif")

    trajs = lambdas = None
    if args.generate or args.train:
        dg = DataGenConfig(n_trajectories=args.n)
        ds_path = Path("research_outputs/spiral_data/spiral_dataset.npz")
        if ds_path.exists() and not args.generate:
            d = np.load(ds_path)
            trajs, lambdas = d["trajectories"], d["params"]
            print(f"Loaded cached dataset {trajs.shape}")
        else:
            trajs, lambdas = generate_dataset(dg)

    if args.train:
        model, norm = train_fno(trajs, lambdas)
        # rollout check on the first (held-out) trajectory
        preds = rollout(model, trajs[0], lambdas[0], norm, steps=trajs.shape[1] - 1)
        plot_spiral_snapshots(preds[:, 0], out / "spiral_fno_rollout.png")
        plot_spiral_snapshots(trajs[0, 1:, 0], out / "spiral_truth.png")
        torch.save(model.state_dict(), out / "film_fno2d.pt")
        print(f"Saved model -> {out/'film_fno2d.pt'}")


if __name__ == "__main__":
    main()
