from __future__ import annotations

from itertools import combinations

import torch
from torch import nn
import torch.nn.functional as F


class CloudTraceMVAD(nn.Module):
    def __init__(self, view_dims: list[int], hidden_dim: int = 96, latent_dim: int = 32):
        super().__init__()
        self.view_dims = list(view_dims)
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, latent_dim),
                nn.LayerNorm(latent_dim),
            )
            for dim in self.view_dims
        ])
        gate_in = latent_dim * len(self.view_dims)
        self.gate = nn.Sequential(
            nn.Linear(gate_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, len(self.view_dims)),
        )
        self.decoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )
            for dim in self.view_dims
        ])

    def forward(self, views: list[torch.Tensor]) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        z = [enc(x) for enc, x in zip(self.encoders, views)]
        logits = self.gate(torch.cat(z, dim=1))
        gate = torch.softmax(logits, dim=1)
        fused = torch.zeros_like(z[0])
        for i, zi in enumerate(z):
            fused = fused + gate[:, i:i + 1] * zi
        recon = [dec(fused) for dec in self.decoders]
        return {"latents": z, "gate": gate, "fused": fused, "reconstructions": recon}


def corrupt_inputs(views: list[torch.Tensor], mask_probability: float, gaussian_noise: float) -> list[torch.Tensor]:
    out = []
    for x in views:
        mask = torch.rand_like(x) >= mask_probability
        noisy = x * mask + gaussian_noise * torch.randn_like(x)
        out.append(noisy)
    return out


def agreement_loss(latents: list[torch.Tensor]) -> torch.Tensor:
    if len(latents) < 2:
        return latents[0].new_tensor(0.0)
    vals = [F.mse_loss(a, b) for a, b in combinations(latents, 2)]
    return torch.stack(vals).mean()


def variance_guard(latents: list[torch.Tensor], target_std: float = 1.0) -> torch.Tensor:
    penalties = []
    for z in latents:
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        penalties.append(torch.relu(target_std - std).mean())
    return torch.stack(penalties).mean()


def training_loss(
    model: CloudTraceMVAD,
    clean_views: list[torch.Tensor],
    mask_probability: float,
    gaussian_noise: float,
    agreement_weight: float,
    variance_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    noisy = corrupt_inputs(clean_views, mask_probability, gaussian_noise)
    out = model(noisy)
    reconstruction = torch.stack([
        F.mse_loss(pred, target) for pred, target in zip(out["reconstructions"], clean_views)
    ]).mean()
    agree = agreement_loss(out["latents"])
    var = variance_guard(out["latents"])
    total = reconstruction + agreement_weight * agree + variance_weight * var
    parts = {"loss": float(total.detach()), "reconstruction": float(reconstruction.detach()),
             "agreement": float(agree.detach()), "variance": float(var.detach())}
    return total, parts


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def approximate_linear_flops(model: nn.Module) -> int:
    total = 0
    for module in model.modules():
        if isinstance(module, nn.Linear):
            total += 2 * module.in_features * module.out_features
    return int(total)
