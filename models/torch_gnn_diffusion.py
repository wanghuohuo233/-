"""PyTorch GNN-conditioned diffusion model.

This is the stronger training backend used when a CUDA PyTorch installation is
available. It keeps dependencies light by implementing graph convolution layers
directly with dense padded adjacency matrices instead of requiring PyG wheels.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TorchDiffusionConfig:
    descriptor_dim: int
    node_feature_dim: int = 8
    condition_dim: int = 3
    graph_hidden_dim: int = 96
    denoiser_hidden_dim: int = 256
    timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.035
    learning_rate: float = 1e-3
    seed: int = 7


class DenseGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_proj = nn.Linear(in_dim, out_dim)
        self.neigh_proj = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        degree = adj.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neigh = torch.bmm(adj, x) / degree
        h = self.self_proj(x) + self.neigh_proj(neigh)
        h = self.norm(h)
        h = F.silu(h)
        return h * mask.unsqueeze(-1)


class GraphConditionEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        self.conv1 = DenseGraphConv(node_dim, hidden_dim)
        self.conv2 = DenseGraphConv(hidden_dim, hidden_dim)
        self.conv3 = DenseGraphConv(hidden_dim, hidden_dim)
        self.out = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim))

    def forward(self, node_features: torch.Tensor, adj: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.conv1(node_features, adj, mask)
        h = self.conv2(h, adj, mask)
        h = self.conv3(h, adj, mask)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = h.sum(dim=1) / denom
        centered = (h - mean.unsqueeze(1)) * mask.unsqueeze(-1)
        std = torch.sqrt((centered * centered).sum(dim=1) / denom + 1e-6)
        return self.out(torch.cat([mean, std], dim=-1))


class TorchGNNConditionedDiffusion(nn.Module):
    def __init__(self, config: TorchDiffusionConfig):
        super().__init__()
        torch.manual_seed(config.seed)
        self.config = config
        self.graph_encoder = GraphConditionEncoder(config.node_feature_dim, config.graph_hidden_dim)
        in_dim = config.descriptor_dim + config.graph_hidden_dim + config.condition_dim + 2
        self.denoiser = nn.Sequential(
            nn.Linear(in_dim, config.denoiser_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.denoiser_hidden_dim),
            nn.Linear(config.denoiser_hidden_dim, config.denoiser_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.denoiser_hidden_dim, config.descriptor_dim),
        )
        betas = torch.linspace(config.beta_start, config.beta_end, config.timesteps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("x_mean", torch.zeros(config.descriptor_dim))
        self.register_buffer("x_std", torch.ones(config.descriptor_dim))
        self.register_buffer("c_mean", torch.zeros(config.condition_dim))
        self.register_buffer("c_std", torch.ones(config.condition_dim))

    def fit_normalizers(self, descriptors: torch.Tensor, conditions: torch.Tensor) -> None:
        self.x_mean.copy_(descriptors.mean(dim=0))
        self.x_std.copy_(descriptors.std(dim=0).clamp_min(1e-6))
        self.c_mean.copy_(conditions.mean(dim=0))
        self.c_std.copy_(conditions.std(dim=0).clamp_min(1e-6))

    def normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean) / self.x_std

    def denormalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.x_std + self.x_mean

    def normalize_condition(self, c: torch.Tensor) -> torch.Tensor:
        return (c - self.c_mean) / self.c_std

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        phase = t.float() / max(1, self.config.timesteps - 1)
        return torch.stack([torch.sin(torch.pi * phase), torch.cos(torch.pi * phase)], dim=-1)

    def predict_noise(
        self,
        noisy_x: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor,
        node_features: torch.Tensor,
        adj: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        graph_emb = self.graph_encoder(node_features, adj, mask)
        inp = torch.cat([noisy_x, graph_emb, condition, self.time_embedding(t)], dim=-1)
        return self.denoiser(inp)

    def training_loss(self, descriptors, conditions, node_features, adj, mask) -> torch.Tensor:
        clean = self.normalize_x(descriptors)
        cond = self.normalize_condition(conditions)
        batch = clean.shape[0]
        t = torch.randint(0, self.config.timesteps, (batch,), device=clean.device)
        noise = torch.randn_like(clean)
        alpha_bar = self.alpha_bars[t].unsqueeze(-1)
        noisy = torch.sqrt(alpha_bar) * clean + torch.sqrt(1.0 - alpha_bar) * noise
        pred = self.predict_noise(noisy, cond, t, node_features, adj, mask)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample(self, condition, node_features, adj, mask, guidance_scale: float = 0.05) -> torch.Tensor:
        device = self.x_mean.device
        if not torch.is_tensor(condition):
            condition = torch.tensor(condition, dtype=torch.float32, device=device)
        if condition.ndim == 1:
            condition = condition.unsqueeze(0).repeat(node_features.shape[0], 1)
        cond = self.normalize_condition(condition.to(device))
        x = torch.randn(node_features.shape[0], self.config.descriptor_dim, device=device)
        for step in reversed(range(self.config.timesteps)):
            t = torch.full((x.shape[0],), step, dtype=torch.long, device=device)
            pred = self.predict_noise(x, cond, t, node_features, adj, mask)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            beta = self.betas[step]
            x = (x - beta * pred / torch.sqrt(1.0 - alpha_bar)) / torch.sqrt(alpha)
            if step > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)
            x = x + guidance_scale * torch.tanh(-x)
        return self.denormalize_x(x)


def save_torch_checkpoint(model: TorchGNNConditionedDiffusion, path: str | Path, extra: Dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "extra": extra or {},
        },
        path,
    )


def load_torch_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> TorchGNNConditionedDiffusion:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = TorchDiffusionConfig(**payload["config"])
    model = TorchGNNConditionedDiffusion(config).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model
