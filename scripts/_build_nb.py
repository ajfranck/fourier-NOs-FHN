"""Builds spiral_control_and_superres_colab.ipynb."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.splitlines(keepends=True)})


md(r"""# Spiral-wave control and zero-shot super-resolution for the FHN-FNO

Two experiments on the already-trained parameter-conditioned FNO from the full
re-evaluation (`checkpoints/parametric_s42.pt`, the 8000-trajectory 1D dataset).

Part 1 is zero-shot super-resolution: the FNO is discretization-invariant, so a
model trained at `nx_train` runs at 2x/4x finer grids with no retraining.

Part 2 builds a 2D FHN solver and FNO, trains a 2D FNO with a stimulus channel
`I_ext`, then freezes it and optimizes a stimulus field through the surrogate to
annihilate a spiral. The discovered stimulus is checked in the true solver.

Drive layout (same as the re-eval notebook):
```
MyDrive/fhn_full_reeval/
    fhn_1d_8000.h5
    checkpoints/parametric_s42.pt
    super_resolution/   Part 1 outputs
    spiral_control/     Part 2 outputs
```
Set the runtime to GPU. Part 2 trains a small 2D model in-notebook.
""")

code(r"""!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
import torch
print('CUDA:', torch.cuda.is_available(), '|',
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')""")

code(r"""!pip install -q h5py
from google.colab import drive
drive.mount('/content/drive')""")

code(r"""import os, math, time, json, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device:', device)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if device == 'cuda':
    torch.cuda.manual_seed_all(SEED)""")

code(r"""import glob

# Drive mounts as either 'MyDrive' or 'My Drive'.
DRIVE_FOLDER_CANDIDATES = ['/content/drive/MyDrive/fhn_full_reeval',
                           '/content/drive/My Drive/fhn_full_reeval']
DRIVE_FOLDER = next((d for d in DRIVE_FOLDER_CANDIDATES if os.path.isdir(d)),
                    DRIVE_FOLDER_CANDIDATES[0])
if not os.path.isdir(DRIVE_FOLDER):
    raise FileNotFoundError(
        f'Folder not found: {DRIVE_FOLDER}\n'
        f'Drive root contains: {sorted(os.listdir("/content/drive/MyDrive"))[:50]}')

print('Drive folder:', DRIVE_FOLDER)
print('contents   :', sorted(os.listdir(DRIVE_FOLDER)))

DATASET_NAME = 'fhn_1d_8000.h5'
DRIVE_DATA = os.path.join(DRIVE_FOLDER, DATASET_NAME)
# exact name, else the first .h5 in the folder
if not os.path.exists(DRIVE_DATA):
    h5s = sorted(glob.glob(os.path.join(DRIVE_FOLDER, '*.h5')))
    assert h5s, f'No .h5 file in {DRIVE_FOLDER}. Found: {sorted(os.listdir(DRIVE_FOLDER))}'
    DRIVE_DATA = h5s[0]
    DATASET_NAME = os.path.basename(DRIVE_DATA)
    print('Exact name not found; using dataset:', DRIVE_DATA)
LOCAL_DATA = f'/content/{DATASET_NAME}'

# parametric_s42.pt, else any .pt under the folder
CKPT_PATH = os.path.join(DRIVE_FOLDER, 'checkpoints', 'parametric_s42.pt')
if not os.path.exists(CKPT_PATH):
    pts = sorted(glob.glob(os.path.join(DRIVE_FOLDER, '**', '*.pt'), recursive=True))
    assert pts, f'No .pt checkpoint under {DRIVE_FOLDER}. Expected parametric_s42.pt.'
    named = [p for p in pts if os.path.basename(p) == 'parametric_s42.pt']
    CKPT_PATH = named[0] if named else pts[0]
    print('Default checkpoint path not found; using:', CKPT_PATH)

OUT_SR = os.path.join(DRIVE_FOLDER, 'super_resolution')
OUT_SP = os.path.join(DRIVE_FOLDER, 'spiral_control')
os.makedirs(OUT_SR, exist_ok=True)
os.makedirs(OUT_SP, exist_ok=True)

if not os.path.exists(LOCAL_DATA):
    print('Copying dataset to local disk...')
    shutil.copy(DRIVE_DATA, LOCAL_DATA)
print('dataset   :', LOCAL_DATA, f'({os.path.getsize(LOCAL_DATA)/1e9:.2f} GB)')
print('checkpoint:', CKPT_PATH)
print('outputs   :', OUT_SR, '|', OUT_SP)""")

md(r"""# Part 1 - Zero-shot super-resolution

Reuses the `ParametricFNO`, FD solver, GRF initial condition, and normalization
from the re-evaluation notebook. The new piece is `spectral_upsample`: draw a
band-limited field at `nx_train`, FFT zero-pad it to each finer grid so every
resolution sees the same continuous function, then run the frozen FNO. The error
should stay roughly flat across resolutions.""")

code(r'''# Model definitions from full_reeval_colab.ipynb (state_dict compatible).
class DataConfig:
    Du_range  = (0.01, 0.1)
    Dv_range  = (0.005, 0.05)
    a_range   = (-0.1, 0.1)
    b_range   = (0.1, 0.5)
    tau_range = (1.0, 20.0)

class SpectralConv1d(nn.Module):
    def __init__(self, in_c, out_c, modes):
        super().__init__()
        self.in_c, self.out_c, self.modes = in_c, out_c, modes
        scale = 1 / (in_c * out_c)
        self.weights = nn.Parameter(scale * torch.randn(in_c, out_c, modes, dtype=torch.cfloat))
    def forward(self, x):
        B, _, nx = x.shape
        xf = torch.fft.rfft(x, dim=-1)
        out = torch.zeros(B, self.out_c, nx // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :self.modes] = torch.einsum('bix,iox->box', xf[:, :, :self.modes], self.weights)
        return torch.fft.irfft(out, n=nx, dim=-1)

class FourierLayer(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.w = nn.Conv1d(width, width, 1)
        self.norm = nn.InstanceNorm1d(width, affine=True)
        self.residual_weight = nn.Parameter(torch.ones(1))
    def forward(self, x):
        residual = x
        out = F.gelu(self.spectral(x) + self.w(x))
        out = self.norm(out)
        return out + self.residual_weight * residual

class ParameterEncoder(nn.Module):
    def __init__(self, n_params, width):
        super().__init__()
        h = width * 2
        self.encoder = nn.Sequential(
            nn.Linear(n_params, h), nn.GELU(), nn.LayerNorm(h),
            nn.Linear(h, width), nn.GELU(),
            nn.Linear(width, width), nn.LayerNorm(width))
    def forward(self, p): return self.encoder(p)

class ParametricFNO(nn.Module):
    def __init__(self, modes=16, width=64, n_layers=6, n_params=5):
        super().__init__()
        self.modes, self.width, self.n_layers = modes, width, n_layers
        self.param_encoder = ParameterEncoder(n_params, width)
        self.gamma_h = nn.ModuleList([nn.Linear(width, width) for _ in range(n_layers)])
        self.beta_h  = nn.ModuleList([nn.Linear(width, width) for _ in range(n_layers)])
        self.lift = nn.Sequential(
            nn.Conv1d(2, width * 2, 1), nn.GELU(),
            nn.Conv1d(width * 2, width, 1))
        self.proj = nn.Sequential(
            nn.Conv1d(width, width, 1), nn.GELU(),
            nn.Conv1d(width, 2, 1))
        self.layers = nn.ModuleList([FourierLayer(width, modes) for _ in range(n_layers)])
        self.global_residual = nn.Parameter(torch.ones(1) * 0.1)
    def forward(self, x, params):
        x0 = x.clone()
        x = self.lift(x)
        pf = self.param_encoder(params)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            g = self.gamma_h[i](pf).unsqueeze(-1)
            b = self.beta_h[i](pf).unsqueeze(-1)
            x = g * x + b
        return self.proj(x) + self.global_residual * x0

print('ParametricFNO defined')''')

code(r"""import h5py
# frozen checkpoint + normalization stats
ck = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model1d = ParametricFNO().to(device)
model1d.load_state_dict(ck['state_dict'])
model1d.eval()
for p in model1d.parameters():
    p.requires_grad_(False)
UM, US, VM, VS = float(ck['um']), float(ck['us']), float(ck['vm']), float(ck['vs'])
PM = np.asarray(ck['pm'], dtype=np.float32)
PS = np.asarray(ck['ps'], dtype=np.float32)
with h5py.File(LOCAL_DATA, 'r') as f:
    NX_TRAIN = int(f['u_traj'].shape[-1])
    N_TIMES  = int(f['u_traj'].shape[1])
print(f'nx_train={NX_TRAIN}  frames/traj={N_TIMES}')
print(f'u: mean={UM:.4f} std={US:.4f} | v: mean={VM:.4f} std={VS:.4f}')
print('pm:', PM, '\nps:', PS)""")

code(r'''# FD solver, GRF IC, spectral upsampling.
# fd_traj_nx is the re-eval fd_traj taking nx as an argument (dx = 1/nx, unit domain).
def fd_traj_nx(u0, v0, params, T, dt, n_save, nx):
    B = u0.shape[0]
    dx = 1.0 / nx
    lap = torch.zeros(nx, nx, device=device, dtype=torch.float32)
    c = 1.0 / (dx * dx)
    for i in range(nx):
        lap[i, i] = -2 * c
        lap[i, (i + 1) % nx] = c
        lap[i, (i - 1) % nx] = c
    Du = params[:, 0].view(B, 1, 1); Dv = params[:, 1].view(B, 1, 1)
    a  = params[:, 2].view(B, 1);    b  = params[:, 3].view(B, 1); tau = params[:, 4].view(B, 1)
    I = torch.eye(nx, device=device).unsqueeze(0).expand(B, nx, nx)
    A_u = I - dt * Du * lap.unsqueeze(0).expand(B, nx, nx)
    A_v = I - dt * Dv * lap.unsqueeze(0).expand(B, nx, nx)
    LU_u, piv_u = torch.linalg.lu_factor(A_u)
    LU_v, piv_v = torch.linalg.lu_factor(A_v)
    n_steps = int(T / dt)
    save_interval = max(1, n_steps // n_save)
    u, v = u0.clone(), v0.clone()
    u_hist = [u.clone()]; v_hist = [v.clone()]
    for step in range(n_steps):
        reaction_u = u - u**3 / 3 - v
        reaction_v = (u + a - b * v) / tau
        rhs_u = u + dt * reaction_u
        rhs_v = v + dt * reaction_v
        u = torch.linalg.lu_solve(LU_u, piv_u, rhs_u.unsqueeze(-1)).squeeze(-1)
        v = torch.linalg.lu_solve(LU_v, piv_v, rhs_v.unsqueeze(-1)).squeeze(-1)
        if (step + 1) % save_interval == 0:
            u_hist.append(u.clone()); v_hist.append(v.clone())
    return torch.stack(u_hist, 1), torch.stack(v_hist, 1)

def sample_grf_ic(B, nx, alpha=2.0, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    k = torch.fft.fftfreq(nx, d=1.0 / nx).to(device)
    k_abs = k.abs(); k_abs[0] = 1e-10
    power = (1 + k_abs**2) ** (-alpha / 2)
    def make(B, scale):
        u_hat = (torch.randn(B, nx, generator=g, device=device)
                 + 1j * torch.randn(B, nx, generator=g, device=device)) * power.unsqueeze(0) * scale
        u = torch.fft.ifft(u_hat).real
        u = (u - u.mean(dim=-1, keepdim=True)) / (u.std(dim=-1, keepdim=True) + 1e-8)
        return u
    return make(B, 1.0), make(B, 0.5)

def spectral_upsample(arr, nx_hi):
    """FFT zero-pad upsample of periodic 1D signals. arr:(B,nx_lo)."""
    nx_lo = arr.shape[-1]
    if nx_hi == nx_lo:
        return arr.clone()
    ft = torch.fft.rfft(arr, dim=-1)
    ft_hi = torch.zeros(arr.shape[0], nx_hi // 2 + 1, dtype=ft.dtype, device=arr.device)
    keep = min(ft.shape[-1], ft_hi.shape[-1])
    ft_hi[:, :keep] = ft[:, :keep]
    return torch.fft.irfft(ft_hi, n=nx_hi, dim=-1) * (nx_hi / nx_lo)

# Frame spacing the operator was trained on: T=1, dt=0.01, n_save=50, so
# dt_save=0.02 on the unit domain. We mirror that and refine nx.
SR_T, SR_DT, SR_NSAVE = 1.0, 0.01, 50
print(f'frame spacing dt_save = {(int(SR_T/SR_DT)//SR_NSAVE)*SR_DT:.3f} on unit domain')''')

code(r'''def evaluate_superres(resolutions, n_traj=12, rollout_steps=20, seed=0):
    cfg = DataConfig()
    ranges = [cfg.Du_range, cfg.Dv_range, cfg.a_range, cfg.b_range, cfg.tau_range]
    rng = np.random.RandomState(seed)
    ic_seeds = rng.randint(0, 2**31 - 1, size=n_traj)
    params_np = np.array([[rng.uniform(*r) for r in ranges] for _ in range(n_traj)], dtype=np.float32)
    pm_t = torch.tensor(PM, device=device); ps_t = torch.tensor(PS, device=device)
    params = torch.tensor(params_np, device=device)
    p_norm = (params - pm_t) / ps_t

    results = {nx: {} for nx in resolutions}
    qualitative = {}
    for nx in resolutions:
        # same field at every grid: sample at nx_train, then upsample
        u0 = torch.cat([spectral_upsample(sample_grf_ic(1, NX_TRAIN, seed=int(s))[0], nx)
                        for s in ic_seeds], 0)
        v0 = torch.cat([spectral_upsample(sample_grf_ic(1, NX_TRAIN, seed=int(s))[1], nx)
                        for s in ic_seeds], 0)
        t0 = time.time()
        u_gt, v_gt = fd_traj_nx(u0, v0, params, SR_T, SR_DT, SR_NSAVE, nx)   # (n_traj, n_save+1, nx)

        # single-step error averaged over consecutive frames
        ss_u = torch.zeros(n_traj, device=device); ss_v = torch.zeros(n_traj, device=device)
        with torch.no_grad():
            for t in range(SR_NSAVE):
                x = torch.stack([(u_gt[:, t] - UM) / US, (v_gt[:, t] - VM) / VS], dim=1)
                pred = model1d(x, p_norm)
                pu = pred[:, 0] * US + UM; pv = pred[:, 1] * VS + VM
                ss_u += torch.linalg.vector_norm(pu - u_gt[:, t + 1], dim=-1) / (
                        torch.linalg.vector_norm(u_gt[:, t + 1], dim=-1) + 1e-12)
                ss_v += torch.linalg.vector_norm(pv - v_gt[:, t + 1], dim=-1) / (
                        torch.linalg.vector_norm(v_gt[:, t + 1], dim=-1) + 1e-12)
                if t == 0:
                    qualitative[nx] = (u_gt[0, 1].cpu().numpy(), pu[0].cpu().numpy())
        ss_u /= SR_NSAVE; ss_v /= SR_NSAVE

        # autoregressive rollout from frame 0
        x = torch.stack([(u_gt[:, 0] - UM) / US, (v_gt[:, 0] - VM) / VS], dim=1)
        roll_u, roll_v = [], []
        with torch.no_grad():
            for _ in range(rollout_steps):
                x = model1d(x, p_norm)
                roll_u.append(x[:, 0] * US + UM); roll_v.append(x[:, 1] * VS + VM)
        roll_u = torch.stack(roll_u, 1); roll_v = torch.stack(roll_v, 1)
        gt_u = u_gt[:, 1:rollout_steps + 1]; gt_v = v_gt[:, 1:rollout_steps + 1]
        ro_u = (torch.linalg.vector_norm(roll_u - gt_u, dim=-1) /
                (torch.linalg.vector_norm(gt_u, dim=-1) + 1e-12)).mean(dim=1)
        ro_v = (torch.linalg.vector_norm(roll_v - gt_v, dim=-1) /
                (torch.linalg.vector_norm(gt_v, dim=-1) + 1e-12)).mean(dim=1)

        results[nx] = {'ss_u': ss_u.cpu().numpy(), 'ss_v': ss_v.cpu().numpy(),
                       'ro_u': ro_u.cpu().numpy(), 'ro_v': ro_v.cpu().numpy()}
        print(f'nx={nx:5d}: {time.time()-t0:5.1f}s  single-step u={ss_u.mean():.4e} '
              f'v={ss_v.mean():.4e} | {rollout_steps}-step rollout u={ro_u.mean():.4e}')
    return results, qualitative, params_np

RESOLUTIONS = [NX_TRAIN, 2 * NX_TRAIN, 4 * NX_TRAIN]
ROLLOUT_STEPS = 20
sr_results, sr_qual, sr_params = evaluate_superres(RESOLUTIONS, rollout_steps=ROLLOUT_STEPS)''')

code(r'''res = sorted(sr_results)
ss_u = [sr_results[n]['ss_u'].mean() for n in res]
ss_u_sd = [sr_results[n]['ss_u'].std() for n in res]
ss_v = [sr_results[n]['ss_v'].mean() for n in res]
ro_u = [sr_results[n]['ro_u'].mean() for n in res]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.errorbar(res, ss_u, yerr=ss_u_sd, marker='o', capsize=3, lw=2, label='single-step u')
ax1.plot(res, ss_v, marker='s', lw=2, label='single-step v')
ax1.plot(res, ro_u, marker='^', ls='--', lw=2, label=f'{ROLLOUT_STEPS}-step rollout u')
ax1.axvline(NX_TRAIN, color='gray', ls=':', lw=1.2)
ax1.text(NX_TRAIN * 1.05, ax1.get_ylim()[1] * 0.5, f'train\n$n_x$={NX_TRAIN}', color='gray', fontsize=9)
ax1.set_xscale('log', base=2); ax1.set_yscale('log')
ax1.set_xticks(res); ax1.set_xticklabels(res)
ax1.set_xlabel('evaluation resolution $n_x$'); ax1.set_ylabel('relative $L^2$ error')
ax1.set_title('Zero-shot super-resolution'); ax1.legend(fontsize=9); ax1.grid(True, which='both', alpha=0.3)

nx_hi = res[-1]
u_gt_q, u_pred_q = sr_qual[nx_hi]
xg = np.linspace(0, 1, nx_hi, endpoint=False)
ax2.plot(xg, u_gt_q, 'k', lw=1.8, label='solver (ground truth)')
ax2.plot(xg, u_pred_q, 'tab:red', ls='--', lw=1.2, label='FNO (zero-shot)')
ax2.set_xlabel('x'); ax2.set_ylabel('u(x)')
ax2.set_title(f'Single-step at $n_x$={nx_hi} ({nx_hi // NX_TRAIN}x training)')
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_SR, 'super_resolution.png'), dpi=200, bbox_inches='tight')
plt.show()

sr_json = {str(n): {k: sr_results[n][k].mean().item() for k in sr_results[n]} for n in res}
sr_json['nx_train'] = NX_TRAIN; sr_json['rollout_steps'] = ROLLOUT_STEPS
with open(os.path.join(OUT_SR, 'super_resolution.json'), 'w') as f:
    json.dump(sr_json, f, indent=2)
print('saved super_resolution.png + .json to', OUT_SR)''')

md(r"""# Part 2 - 2D spiral waves and differentiable defibrillation

Here we build a 2D FHN solver and a 2D parameter-conditioned FNO in the same
style as the 1D one (FiLM conditioning, spectral + 1x1 conv + InstanceNorm +
residual blocks, global input skip). The 2D solver is semi-implicit, with
diffusion FFT-diagonalized so it scales and stays differentiable.

The 2D FNO takes a third input channel `I_ext` (a stimulus field), so it learns
the response to forcing. We then show the frozen FNO reproduces a rotating
spiral, optimize a stimulus field through it to minimize spiral activity (with
an energy penalty), and check the result in the true solver.

The control step backprops through `N_CTRL` rollout steps. Defaults fit on a T4;
reduce `N_CTRL` or `NX2D` if needed.""")

code(r"""NX2D = 128
L2D = 4.0
DX2D = L2D / NX2D
DT2D = 0.05
SAVE_EVERY2D = 8
DT_SAVE2D = SAVE_EVERY2D * DT2D            # time between FNO frames

# excitable params lambda = (Du, Dv, a, b, tau)
LAM2D = np.array([0.015, 0.0025, 0.70, 0.80, 12.0], dtype=np.float32)
SPIRAL_RANGES = dict(Du=(0.012, 0.018), Dv=(0.0015, 0.0035),
                     a=(0.65, 0.78), b=(0.72, 0.88), tau=(10.0, 14.0))
U_EXC = 1.5
DV_REFR = 0.7

# dataset
N_TRAJ2D = 64
TRANSIENT_STEPS = 400
REC_FRAMES = 24
P_STIM = 0.6
STIM_AMP = (-1.2, 1.2)
STIM_SIGMA = (0.25, 0.6)
N_BUMPS = (1, 3)

# 2D model
MODES2D = 12
WIDTH2D = 32
NLAYERS2D = 4

# training
EPOCHS2D = 30
BATCH2D = 8
LR2D = 1e-3

# control
N_PULSE = 3
N_CTRL = 12
ENERGY = 1e-3
CTRL_ITERS = 150
CTRL_LR = 0.2
print('Part 2 config set; DT_SAVE2D =', DT_SAVE2D)""")

code(r'''def rest_state(a, b):
    """Most-negative real root of u^3 + (3/b - 3) u + 3a/b = 0, and v* = (u*+a)/b."""
    roots = np.roots([1.0, 0.0, (3.0 / b - 3.0), 3.0 * a / b])
    real = roots[np.abs(roots.imag) < 1e-9].real
    u_star = float(np.min(real))
    return u_star, float((u_star + a) / b)

def is_excitable(a, b, tau):
    u_star, _ = rest_state(a, b)
    tr = (1.0 - u_star**2) - b / tau
    det = (1.0 - b * (1.0 - u_star**2)) / tau
    return (tr < 0.0) and (det > 0.0)

class FHN2DSolver:
    """Periodic 2D FHN. Explicit reaction, implicit FFT-diagonalized diffusion.
    Differentiable in I_ext."""
    def __init__(self, nx, dx, device):
        self.nx, self.dx, self.device = nx, dx, device
        kx = 2 * math.pi * torch.fft.fftfreq(nx, d=dx).to(device)
        ky = 2 * math.pi * torch.fft.rfftfreq(nx, d=dx).to(device)
        KX, KY = torch.meshgrid(kx, ky, indexing='ij')
        self.k2 = (KX**2 + KY**2).unsqueeze(0)            # (1, nx, nx//2+1)
    def step(self, u, v, p, dt, I_ext=None):
        Du = p[:, 0].view(-1, 1, 1); Dv = p[:, 1].view(-1, 1, 1)
        a = p[:, 2].view(-1, 1, 1); b = p[:, 3].view(-1, 1, 1); tau = p[:, 4].view(-1, 1, 1)
        ru = u - u**3 / 3 - v
        if I_ext is not None:
            ru = ru + I_ext
        rv = (u + a - b * v) / tau
        us = u + dt * ru; vs = v + dt * rv
        uh = torch.fft.rfft2(us) / (1 + dt * Du * self.k2)
        vh = torch.fft.rfft2(vs) / (1 + dt * Dv * self.k2)
        return torch.fft.irfft2(uh, s=(self.nx, self.nx)), torch.fft.irfft2(vh, s=(self.nx, self.nx))
    @torch.no_grad()
    def run(self, u0, v0, p, n_steps, save_every, dt, I_ext=None, n_pulse_steps=None):
        u, v = u0.clone(), v0.clone()
        U, V = [u.clone()], [v.clone()]
        for s in range(n_steps):
            stim = I_ext
            if I_ext is not None and n_pulse_steps is not None and s >= n_pulse_steps:
                stim = None
            u, v = self.step(u, v, p, dt, stim)
            if (s + 1) % save_every == 0:
                U.append(u.clone()); V.append(v.clone())
        return torch.stack(U, 1), torch.stack(V, 1)

def spiral_ic(nx, u_rest, v_rest, u_exc, dv_refr):
    """Cross-field IC: excited top half, refractory left half, giving a broken
    wave-front that nucleates a rotating spiral."""
    u = torch.full((1, nx, nx), float(u_rest))
    v = torch.full((1, nx, nx), float(v_rest))
    u[:, :nx // 2, :] = float(u_exc)
    v[:, :, :nx // 2] = float(v_rest + dv_refr)
    return u, v

SOLVER2D = FHN2DSolver(NX2D, DX2D, device)
print('LAM2D excitable:', is_excitable(LAM2D[2], LAM2D[3], LAM2D[4]))''')

code(r"""u_rest, v_rest = rest_state(LAM2D[2], LAM2D[3])
print(f'rest state u*={u_rest:.3f} v*={v_rest:.3f}')
u0, v0 = spiral_ic(NX2D, u_rest, v_rest, U_EXC, DV_REFR)
p1 = torch.tensor(LAM2D, device=device).unsqueeze(0)
DEMO_FRAMES = 60
t0 = time.time()
Ud, Vd = SOLVER2D.run(u0.to(device), v0.to(device), p1,
                      DEMO_FRAMES * SAVE_EVERY2D, SAVE_EVERY2D, DT2D)
print(f'solver demo {time.time()-t0:.1f}s  u:', tuple(Ud.shape))

show = np.linspace(0, DEMO_FRAMES, 6).astype(int)
fig, ax = plt.subplots(1, 6, figsize=(18, 3.2))
for j, fr in enumerate(show):
    ax[j].imshow(Ud[0, fr].cpu().numpy(), origin='lower', cmap='inferno', vmin=-2, vmax=2)
    ax[j].set_title(f't={fr*DT_SAVE2D:.1f}'); ax[j].axis('off')
fig.suptitle('Solver: rotating spiral (u field)')
fig.tight_layout(); fig.savefig(os.path.join(OUT_SP, 'spiral_solver_demo.png'), dpi=140); plt.show()""")

code(r'''def random_stim_field(nx, dx, rng):
    x = torch.linspace(0, (nx - 1) * dx, nx, device=device)
    X, Y = torch.meshgrid(x, x, indexing='ij')
    n = int(rng.integers(N_BUMPS[0], N_BUMPS[1] + 1))
    field = torch.zeros(nx, nx, device=device)
    for _ in range(n):
        amp = rng.uniform(*STIM_AMP); sig = rng.uniform(*STIM_SIGMA)
        cx = rng.uniform(0, (nx - 1) * dx); cy = rng.uniform(0, (nx - 1) * dx)
        field = field + amp * torch.exp(-((X - cx)**2 + (Y - cy)**2) / (2 * sig**2))
    return field

@torch.no_grad()
def generate_spiral_dataset(n_traj, seed):
    rng = np.random.default_rng(seed)
    keys = list(SPIRAL_RANGES)
    U_all, V_all, P_all, I_all = [], [], [], []
    for k in range(n_traj):
        lam = np.array([rng.uniform(*SPIRAL_RANGES[key]) for key in keys], dtype=np.float32)
        while not is_excitable(lam[2], lam[3], lam[4]):
            lam[2] = rng.uniform(*SPIRAL_RANGES['a']); lam[3] = rng.uniform(*SPIRAL_RANGES['b'])
        ur, vr = rest_state(lam[2], lam[3])
        u0, v0 = spiral_ic(NX2D, ur, vr, U_EXC, DV_REFR)
        if rng.random() < 0.5:
            u0 = u0.transpose(-1, -2).contiguous(); v0 = v0.transpose(-1, -2).contiguous()
        roll = (int(rng.integers(0, NX2D)), int(rng.integers(0, NX2D)))
        u0 = torch.roll(u0, roll, dims=(-2, -1)); v0 = torch.roll(v0, roll, dims=(-2, -1))
        p = torch.tensor(lam, device=device).unsqueeze(0)
        I_ext = (random_stim_field(NX2D, DX2D, rng) if rng.random() < P_STIM
                 else torch.zeros(NX2D, NX2D, device=device))
        ut, vt = SOLVER2D.run(u0.to(device), v0.to(device), p,
                              TRANSIENT_STEPS, TRANSIENT_STEPS, DT2D, I_ext.unsqueeze(0))
        Ur, Vr = SOLVER2D.run(ut[:, -1], vt[:, -1], p,
                              REC_FRAMES * SAVE_EVERY2D, SAVE_EVERY2D, DT2D, I_ext.unsqueeze(0))
        U_all.append(Ur.cpu()); V_all.append(Vr.cpu()); P_all.append(lam); I_all.append(I_ext.cpu())
        if (k + 1) % 8 == 0:
            print(f'  generated {k+1}/{n_traj}', end='\r')
    print()
    return (torch.cat(U_all, 0), torch.cat(V_all, 0),
            torch.tensor(np.stack(P_all)), torch.stack(I_all, 0))

t0 = time.time()
U2, V2, P2, I2 = generate_spiral_dataset(N_TRAJ2D, seed=SEED)
print(f'dataset built in {time.time()-t0:.1f}s  U:', tuple(U2.shape), ' I:', tuple(I2.shape))''')

code(r'''# 2D parametric FNO, same FiLM/residual design as the 1D model.
class SpectralConv2d(nn.Module):
    def __init__(self, in_c, out_c, modes):
        super().__init__()
        self.in_c, self.out_c, self.modes = in_c, out_c, modes
        scale = 1 / (in_c * out_c)
        self.w1 = nn.Parameter(scale * torch.randn(in_c, out_c, modes, modes, dtype=torch.cfloat))
        self.w2 = nn.Parameter(scale * torch.randn(in_c, out_c, modes, modes, dtype=torch.cfloat))
    def forward(self, x):
        B, _, H, W = x.shape
        xf = torch.fft.rfft2(x)
        out = torch.zeros(B, self.out_c, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)
        m = self.modes
        out[:, :, :m, :m] = torch.einsum('bixy,ioxy->boxy', xf[:, :, :m, :m], self.w1)
        out[:, :, -m:, :m] = torch.einsum('bixy,ioxy->boxy', xf[:, :, -m:, :m], self.w2)
        return torch.fft.irfft2(out, s=(H, W))

class FourierLayer2d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spectral = SpectralConv2d(width, width, modes)
        self.w = nn.Conv2d(width, width, 1)
        self.norm = nn.InstanceNorm2d(width, affine=True)
        self.residual_weight = nn.Parameter(torch.ones(1))
    def forward(self, x):
        residual = x
        out = F.gelu(self.spectral(x) + self.w(x))
        out = self.norm(out)
        return out + self.residual_weight * residual

class ParameterEncoder2d(nn.Module):
    def __init__(self, n_params, width):
        super().__init__()
        h = width * 2
        self.encoder = nn.Sequential(
            nn.Linear(n_params, h), nn.GELU(), nn.LayerNorm(h),
            nn.Linear(h, width), nn.GELU(),
            nn.Linear(width, width), nn.LayerNorm(width))
    def forward(self, p): return self.encoder(p)

class ParametricFNO2d(nn.Module):
    """in_ch = 3 (u, v, I_ext), out_ch = 2 (u, v). Global skip uses (u, v) only."""
    def __init__(self, modes=MODES2D, width=WIDTH2D, n_layers=NLAYERS2D, n_params=5, in_ch=3):
        super().__init__()
        self.param_encoder = ParameterEncoder2d(n_params, width)
        self.film_gamma = nn.ModuleList([nn.Linear(width, width) for _ in range(n_layers)])
        self.film_beta = nn.ModuleList([nn.Linear(width, width) for _ in range(n_layers)])
        self.lift = nn.Sequential(nn.Conv2d(in_ch, width * 2, 1), nn.GELU(),
                                  nn.Conv2d(width * 2, width, 1))
        self.layers = nn.ModuleList([FourierLayer2d(width, modes) for _ in range(n_layers)])
        self.projection = nn.Sequential(nn.Conv2d(width, width, 1), nn.GELU(),
                                        nn.Conv2d(width, 2, 1))
        self.global_residual = nn.Parameter(torch.ones(1) * 0.1)
    def forward(self, x, params):
        x0 = x[:, :2]
        x = self.lift(x)
        pf = self.param_encoder(params)
        for i, layer in enumerate(self.layers):
            g = self.film_gamma[i](pf).view(-1, pf.shape[-1], 1, 1)
            b = self.film_beta[i](pf).view(-1, pf.shape[-1], 1, 1)
            x = g * layer(x) + b
        return self.projection(x) + self.global_residual * x0

model2d = ParametricFNO2d().to(device)
print('2D FNO params:', sum(p.numel() for p in model2d.parameters()))''')

code(r"""def build_pairs2d(U, V, I, P):
    N, Tn, H, W = U.shape
    X_in = torch.stack([U[:, :-1], V[:, :-1]], dim=2).reshape(-1, 2, H, W)
    X_out = torch.stack([U[:, 1:], V[:, 1:]], dim=2).reshape(-1, 2, H, W)
    pidx = torch.arange(N).view(N, 1).expand(N, Tn - 1).reshape(-1)
    Ich = I[pidx].unsqueeze(1)
    return X_in, X_out, Ich, pidx

Xtr, Ytr, Itr, Ptr = build_pairs2d(U2, V2, I2, P2)
print('train pairs:', Xtr.shape[0])

um2, us2 = U2.mean(), U2.std()
vm2, vs2 = V2.mean(), V2.std()
cm = torch.tensor([um2, vm2]).view(1, 2, 1, 1)
cs = torch.tensor([us2, vs2]).view(1, 2, 1, 1)
pm2, ps2 = P2.mean(0), P2.std(0) + 1e-8
i_scale = float(I2.abs().std() + 1e-6)
def norm_x2(x): return (x - cm.to(x.device)) / cs.to(x.device)
def denorm2(x): return x * cs.to(x.device) + cm.to(x.device)
def norm_p2(p): return (p - pm2.to(p.device)) / ps2.to(p.device)
print('u m/s', float(um2), float(us2), '| v m/s', float(vm2), float(vs2), '| i_scale', i_scale)""")

code(r"""def rel_l2(pred, tgt):
    num = torch.linalg.vector_norm((pred - tgt).flatten(1), dim=-1)
    den = torch.linalg.vector_norm(tgt.flatten(1), dim=-1) + 1e-8
    return (num / den).mean()

opt = torch.optim.AdamW(model2d.parameters(), lr=LR2D, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS2D)
n_pairs = Xtr.shape[0]
Pn2 = norm_p2(P2)
for ep in range(EPOCHS2D):
    model2d.train()
    perm = torch.randperm(n_pairs)
    running = 0.0
    for s in range(0, n_pairs, BATCH2D):
        idx = perm[s:s + BATCH2D]
        x = norm_x2(Xtr[idx]).to(device)
        ich = (Itr[idx] / i_scale).to(device)
        x = torch.cat([x, ich], dim=1)
        y = norm_x2(Ytr[idx]).to(device)
        p = Pn2[Ptr[idx]].to(device)
        opt.zero_grad()
        loss = rel_l2(model2d(x, p), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model2d.parameters(), 1.0)
        opt.step()
        running += loss.item() * idx.numel()
    sched.step()
    if (ep + 1) % 5 == 0 or ep == 0:
        print(f'epoch {ep+1:3d}  train relL2 {running/n_pairs:.4e}')
print('2D training done')""")

code(r"""@torch.no_grad()
def rollout2d(model, u0, v0, p, n_steps):
    x = norm_x2(torch.stack([u0, v0], dim=1)).to(device)
    pn = norm_p2(p).to(device)
    zero = torch.zeros(x.shape[0], 1, x.shape[-2], x.shape[-1], device=device)
    frames = [x]
    for _ in range(n_steps):
        x = model(torch.cat([x, zero], dim=1), pn)
        frames.append(x)
    return denorm2(torch.stack(frames, 1))

# held-out spiral, stimulus-free
lam = LAM2D.copy()
ur, vr = rest_state(lam[2], lam[3])
u0, v0 = spiral_ic(NX2D, ur, vr, U_EXC, DV_REFR)
p1 = torch.tensor(lam, device=device).unsqueeze(0)
ut, vt = SOLVER2D.run(u0.to(device), v0.to(device), p1, TRANSIENT_STEPS, TRANSIENT_STEPS, DT2D)
u_start, v_start = ut[:, -1], vt[:, -1]
ROLL2D = 40
Ug, Vg = SOLVER2D.run(u_start, v_start, p1, ROLL2D * SAVE_EVERY2D, SAVE_EVERY2D, DT2D)
traj = rollout2d(model2d, u_start, v_start, p1, ROLL2D)

err = []
for t in range(ROLL2D + 1):
    num = torch.linalg.vector_norm((traj[0, t, 0] - Ug[0, t]).flatten())
    den = torch.linalg.vector_norm(Ug[0, t].flatten()) + 1e-8
    err.append((num / den).item())

fig, ax = plt.subplots(3, 5, figsize=(15, 9))
show = np.linspace(0, ROLL2D, 5).astype(int)
for j, fr in enumerate(show):
    ax[0, j].imshow(Ug[0, fr].cpu().numpy(), origin='lower', cmap='inferno', vmin=-2, vmax=2)
    ax[0, j].set_title(f'solver t={fr*DT_SAVE2D:.1f}'); ax[0, j].axis('off')
    ax[1, j].imshow(traj[0, fr, 0].cpu().numpy(), origin='lower', cmap='inferno', vmin=-2, vmax=2)
    ax[1, j].set_title('FNO rollout'); ax[1, j].axis('off')
    ax[2, j].imshow((traj[0, fr, 0] - Ug[0, fr]).abs().cpu().numpy(), origin='lower', cmap='magma')
    ax[2, j].set_title('|error|'); ax[2, j].axis('off')
fig.suptitle('2D spiral: solver vs frozen-FNO rollout (u)')
fig.tight_layout(); fig.savefig(os.path.join(OUT_SP, 'spiral_rollout.png'), dpi=140); plt.show()
print(f'rollout relL2 u  step1={err[1]:.3e}  step{ROLL2D}={err[ROLL2D]:.3e}')""")

code(r"""# Freeze the surrogate and optimize a stimulus S applied for the first
# N_PULSE steps to drive the spiral back to rest.
for p in model2d.parameters():
    p.requires_grad_(False)

u_rest_c, v_rest_c = rest_state(LAM2D[2], LAM2D[3])
x0_ctrl = norm_x2(torch.stack([u_start, v_start], dim=1)).to(device)
pn_ctrl = norm_p2(torch.tensor(LAM2D, device=device).unsqueeze(0))

def controlled_rollout(S_phys, n_steps, n_pulse):
    S_norm = (S_phys / i_scale).view(1, 1, NX2D, NX2D)
    zero = torch.zeros_like(S_norm)
    x = x0_ctrl
    frames = []
    for s in range(n_steps):
        ich = S_norm if s < n_pulse else zero
        x = model2d(torch.cat([x, ich], dim=1), pn_ctrl)
        frames.append(x)
    return torch.stack(frames, 1)

S = torch.zeros(NX2D, NX2D, device=device, requires_grad=True)
opt_c = torch.optim.Adam([S], lr=CTRL_LR)
KLAST = 4
hist = []
for it in range(CTRL_ITERS):
    opt_c.zero_grad()
    traj_c = controlled_rollout(S, N_CTRL, N_PULSE)
    u_phys = traj_c[0, -KLAST:, 0] * us2.to(device) + um2.to(device)
    activity = ((u_phys - u_rest_c) ** 2).mean()
    energy = (S ** 2).mean()
    loss = activity + ENERGY * energy
    loss.backward()
    opt_c.step()
    hist.append((activity.item(), energy.item()))
    if (it + 1) % 25 == 0 or it == 0:
        print(f'iter {it+1:4d}  activity {activity.item():.4e}  energy {energy.item():.4e}')
S_opt = S.detach()
print('optimized stimulus  |S|max', float(S_opt.abs().max()),
      ' rms', float((S_opt**2).mean().sqrt()))""")

code(r"""# free vs controlled rollout in the FNO
with torch.no_grad():
    free = controlled_rollout(torch.zeros_like(S_opt), N_CTRL, N_PULSE)
    ctrl = controlled_rollout(S_opt, N_CTRL, N_PULSE)
free_u = (free[0, -1, 0] * us2.to(device) + um2.to(device)).cpu().numpy()
ctrl_u = (ctrl[0, -1, 0] * us2.to(device) + um2.to(device)).cpu().numpy()

# check the stimulus in the true solver
p1 = torch.tensor(LAM2D, device=device).unsqueeze(0)
Sb = S_opt.unsqueeze(0)
Uc, Vc = SOLVER2D.run(u_start, v_start, p1, N_CTRL * SAVE_EVERY2D, SAVE_EVERY2D, DT2D,
                      I_ext=Sb, n_pulse_steps=N_PULSE * SAVE_EVERY2D)
Uf, Vf = SOLVER2D.run(u_start, v_start, p1, N_CTRL * SAVE_EVERY2D, SAVE_EVERY2D, DT2D)
solver_free_u = Uf[0, -1].cpu().numpy()
solver_ctrl_u = Uc[0, -1].cpu().numpy()

def amp(u): return float(((u - u_rest_c) ** 2).mean())
print(f'final activity (FNO)    free={amp(free_u):.4e}  controlled={amp(ctrl_u):.4e}')
print(f'final activity (solver) free={amp(solver_free_u):.4e}  controlled={amp(solver_ctrl_u):.4e}')

fig, ax = plt.subplots(2, 3, figsize=(13, 8.5))
panels = [(free_u, 'FNO: no control'), (ctrl_u, 'FNO: controlled'),
          (S_opt.cpu().numpy(), 'stimulus S'),
          (solver_free_u, 'solver: no control'), (solver_ctrl_u, 'solver: controlled'),
          (None, 'activity')]
for k, (img, title) in enumerate(panels):
    r, c = divmod(k, 3)
    if title == 'activity':
        a = np.array(hist)
        ax[r, c].plot(a[:, 0], lw=2); ax[r, c].set_yscale('log')
        ax[r, c].set_xlabel('iteration'); ax[r, c].set_ylabel('spiral activity')
        ax[r, c].set_title('control objective'); ax[r, c].grid(alpha=0.3)
    elif title == 'stimulus S':
        vmax = float(np.abs(img).max()) + 1e-9
        im = ax[r, c].imshow(img, origin='lower', cmap='seismic', vmin=-vmax, vmax=vmax)
        ax[r, c].set_title(title); ax[r, c].axis('off')
        plt.colorbar(im, ax=ax[r, c], fraction=0.046)
    else:
        ax[r, c].imshow(img, origin='lower', cmap='inferno', vmin=-2, vmax=2)
        ax[r, c].set_title(title); ax[r, c].axis('off')
fig.suptitle('Differentiable defibrillation: stimulus learned through the surrogate annihilates the spiral')
fig.tight_layout(); fig.savefig(os.path.join(OUT_SP, 'spiral_control.png'), dpi=140); plt.show()

torch.save({'model_state': model2d.state_dict(),
            'norm': {'um': float(um2), 'us': float(us2), 'vm': float(vm2), 'vs': float(vs2),
                     'pm': pm2.tolist(), 'ps': ps2.tolist(), 'i_scale': i_scale},
            'config': {'NX2D': NX2D, 'L2D': L2D, 'DT2D': DT2D, 'SAVE_EVERY2D': SAVE_EVERY2D,
                       'MODES2D': MODES2D, 'WIDTH2D': WIDTH2D, 'NLAYERS2D': NLAYERS2D,
                       'LAM2D': LAM2D.tolist(), 'N_PULSE': N_PULSE, 'N_CTRL': N_CTRL}},
           os.path.join(OUT_SP, 'spiral_fno2d.pt'))
with open(os.path.join(OUT_SP, 'control_results.json'), 'w') as f:
    json.dump({'activity_fno_free': amp(free_u), 'activity_fno_ctrl': amp(ctrl_u),
               'activity_solver_free': amp(solver_free_u), 'activity_solver_ctrl': amp(solver_ctrl_u),
               'rollout_relL2_u': err, 'S_rms': float((S_opt**2).mean().sqrt())}, f, indent=2)
print('saved spiral_fno2d.pt, spiral_control.png, spiral_rollout.png, control_results.json')""")

md(r"""## Notes and tuning

Super-resolution reuses the trained 1D checkpoint with no retraining; the frame
spacing (`dt_save = 0.02`) matches the re-eval, so the single-step numbers are
comparable to Table 5.

If you do not see a sustained spiral in the solver demo, increase
`TRANSIENT_STEPS`, nudge `Du` up, or widen `DV_REFR`.

`ENERGY` trades off annihilation against stimulus energy. `N_PULSE` sets how
long the stimulus is applied; `N_CTRL` is the optimization horizon (longer means
stronger gradient but more memory).""")

out = Path(__file__).resolve().parent / "spiral_control_and_superres_colab.ipynb"
nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} with {len(cells)} cells")
