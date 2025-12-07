import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# note: remove the param conditioning it is useless, that paper was pointless


class SpectralConv1d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, nx = x.shape

        # FFT
        x_ft = torch.fft.rfft(x, dim=-1)

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            nx // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, : self.modes], self.weights
        )

        out = torch.fft.irfft(out_ft, n=nx, dim=-1)

        return out


class SpectralConv2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale
            * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale
            * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, nx, ny = x.shape

        x_ft = torch.fft.rfft2(x, dim=(-2, -1))

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            nx,
            ny // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        out_ft[:, :, : self.modes1, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )

        out_ft[:, :, -self.modes1 :, : self.modes2] = torch.einsum(
            "bixy,ioxy->boxy", x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        out = torch.fft.irfft2(out_ft, s=(nx, ny), dim=(-2, -1))

        return out


class FourierLayer(nn.Module):

    def __init__(self, width: int, modes: int, dim: int = 1):
        super().__init__()
        self.width = width
        self.modes = modes
        self.dim = dim

        if dim == 1:
            self.spectral = SpectralConv1d(width, width, modes)
            self.w = nn.Conv1d(width, width, 1)
            self.norm = nn.InstanceNorm1d(width, affine=True)
        elif dim == 2:
            self.spectral = SpectralConv2d(width, width, modes, modes)
            self.w = nn.Conv2d(width, width, 1)
            self.norm = nn.InstanceNorm2d(width, affine=True)
        else:
            raise ValueError(f"Unsupported dimension: {dim}")

        self.residual_weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out1 = self.spectral(x)

        out2 = self.w(x)

        out = F.gelu(out1 + out2)

        out = self.norm(out)

        out = out + self.residual_weight * residual

        return out


class ParameterEncoder(nn.Module):

    def __init__(self, n_params: int, width: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or width * 2

        self.encoder = nn.Sequential(
            nn.Linear(n_params, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, width),
            nn.GELU(),
            nn.Linear(width, width),
            nn.LayerNorm(width),
        )

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        return self.encoder(params)


class FNO(nn.Module):

    def __init__(
        self,
        modes: int = 16,
        width: int = 64,
        n_layers: int = 4,
        in_channels: int = 2,
        out_channels: int = 2,
        dim: int = 1,
        use_positional: bool = False,
        use_param_conditioning: bool = True,
        n_params: int = 5,
    ):

        super().__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.use_positional = use_positional
        self.use_param_conditioning = use_param_conditioning

        if use_param_conditioning:
            self.param_encoder = ParameterEncoder(n_params, width)
            self.param_film_gamma = nn.ModuleList(
                [nn.Linear(width, width) for _ in range(n_layers)]
            )
            self.param_film_beta = nn.ModuleList(
                [nn.Linear(width, width) for _ in range(n_layers)]
            )

        lift_in_channels = in_channels + dim if use_positional else in_channels

        if dim == 1:
            self.lift = nn.Sequential(
                nn.Conv1d(lift_in_channels, width * 2, 1),
                nn.GELU(),
                nn.Conv1d(width * 2, width, 1),
            )
            self.projection = nn.Sequential(
                nn.Conv1d(width, width, 1), nn.GELU(), nn.Conv1d(width, out_channels, 1)
            )
        elif dim == 2:
            self.lift = nn.Sequential(
                nn.Conv2d(lift_in_channels, width * 2, 1),
                nn.GELU(),
                nn.Conv2d(width * 2, width, 1),
            )
            self.projection = nn.Sequential(
                nn.Conv2d(width, width, 1), nn.GELU(), nn.Conv2d(width, out_channels, 1)
            )
        else:
            raise ValueError(f"Unsupported dimension: {dim}")

        self.fourier_layers = nn.ModuleList(
            [FourierLayer(width, modes, dim) for _ in range(n_layers)]
        )

        self.global_residual = nn.Parameter(torch.ones(1) * 0.1)

    def forward(
        self, x: torch.Tensor, params: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        input_x = x.clone()

        if self.use_positional:
            if self.dim == 1:
                nx = x.shape[-1]
                pos = torch.linspace(0, 1, nx, device=x.device).view(1, 1, nx)
                pos = pos.expand(batch_size, 1, nx)
                x = torch.cat([x, pos], dim=1)
            elif self.dim == 2:
                nx, ny = x.shape[-2:]
                x_pos = torch.linspace(0, 1, nx, device=x.device).view(1, 1, nx, 1)
                y_pos = torch.linspace(0, 1, ny, device=x.device).view(1, 1, 1, ny)
                x_pos = x_pos.expand(batch_size, 1, nx, ny)
                y_pos = y_pos.expand(batch_size, 1, nx, ny)
                x = torch.cat([x, x_pos, y_pos], dim=1)

        x = self.lift(x)

        if self.use_param_conditioning and params is not None:
            param_features = self.param_encoder(params)  # (batch, width)

        for i, layer in enumerate(self.fourier_layers):
            if self.use_param_conditioning and params is not None:
                gamma = self.param_film_gamma[i](param_features)
                beta = self.param_film_beta[i](param_features)

                if self.dim == 1:
                    gamma = gamma.unsqueeze(-1)
                    beta = beta.unsqueeze(-1)
                else:
                    gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                    beta = beta.unsqueeze(-1).unsqueeze(-1)

                x = layer(x)
                x = gamma * x + beta
            else:
                x = layer(x)

        x = self.projection(x)

        x = x + self.global_residual * input_x

        return x

    def rollout(
        self, x0: torch.Tensor, n_steps: int, params: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        trajectory = [x0]
        x = x0

        for _ in range(n_steps):
            x = self.forward(x, params)
            trajectory.append(x)

        return torch.stack(trajectory, dim=1)
