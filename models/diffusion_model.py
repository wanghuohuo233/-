"""Conditional graph diffusion model implemented with NumPy.

For a production research pipeline this module can be replaced by a PyTorch
Geometric equivariant GNN diffusion backbone. The present implementation keeps
the repository small, reproducible, and runnable in a clean interview machine:
the graph information is encoded into message-passing descriptors, and the
diffusion network learns to denoise those descriptors conditioned on target
HER/stability/synthesis properties.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass
class DiffusionConfig:
    input_dim: int
    condition_dim: int = 3
    hidden_dim: int = 128
    timesteps: int = 60
    beta_start: float = 1e-4
    beta_end: float = 0.045
    learning_rate: float = 2e-3
    seed: int = 7


class GraphMessageEncoder:
    """Tiny message-passing encoder for atom features and adjacency matrices."""

    def __init__(self, feature_dim: int, hidden_dim: int = 16, seed: int = 7):
        rng = np.random.default_rng(seed)
        self.w_self = rng.normal(0.0, 0.15, (feature_dim, hidden_dim))
        self.w_neigh = rng.normal(0.0, 0.15, (feature_dim, hidden_dim))
        self.bias = np.zeros(hidden_dim, dtype=float)

    def encode(self, atom_features: np.ndarray, adjacency: np.ndarray) -> np.ndarray:
        degree = adjacency.sum(axis=1, keepdims=True)
        neighbor_mean = (adjacency @ atom_features) / np.maximum(degree, 1.0)
        hidden = np.tanh(atom_features @ self.w_self + neighbor_mean @ self.w_neigh + self.bias)
        return np.concatenate([hidden.mean(axis=0), hidden.std(axis=0)])


class ConditionalGraphDiffusion:
    """Denoising diffusion model over graph/material descriptors."""

    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.betas = np.linspace(config.beta_start, config.beta_end, config.timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = np.cumprod(self.alphas)

        in_dim = config.input_dim + config.condition_dim + 2
        scale = 1.0 / np.sqrt(in_dim)
        self.w1 = self.rng.normal(0.0, scale, (in_dim, config.hidden_dim))
        self.b1 = np.zeros(config.hidden_dim, dtype=float)
        self.w2 = self.rng.normal(0.0, 1.0 / np.sqrt(config.hidden_dim), (config.hidden_dim, config.input_dim))
        self.b2 = np.zeros(config.input_dim, dtype=float)

        self.x_mean = np.zeros(config.input_dim, dtype=float)
        self.x_std = np.ones(config.input_dim, dtype=float)
        self.c_mean = np.zeros(config.condition_dim, dtype=float)
        self.c_std = np.ones(config.condition_dim, dtype=float)

    def fit_normalizers(self, x: np.ndarray, condition: np.ndarray) -> None:
        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0) + 1e-6
        self.c_mean = condition.mean(axis=0)
        self.c_std = condition.std(axis=0) + 1e-6

    def normalize_x(self, x: np.ndarray) -> np.ndarray:
        return (x - self.x_mean) / self.x_std

    def denormalize_x(self, x: np.ndarray) -> np.ndarray:
        return x * self.x_std + self.x_mean

    def normalize_condition(self, condition: np.ndarray) -> np.ndarray:
        return (condition - self.c_mean) / self.c_std

    def _time_embedding(self, t: np.ndarray) -> np.ndarray:
        phase = t.astype(float) / max(1, self.config.timesteps - 1)
        return np.stack([np.sin(np.pi * phase), np.cos(np.pi * phase)], axis=1)

    def _forward(self, noisy_x: np.ndarray, condition: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        inp = np.concatenate([noisy_x, condition, self._time_embedding(t)], axis=1)
        hidden_raw = inp @ self.w1 + self.b1
        hidden = np.tanh(hidden_raw)
        pred = hidden @ self.w2 + self.b2
        return pred, hidden, inp

    def train_epoch(self, x: np.ndarray, condition: np.ndarray, batch_size: int = 16) -> float:
        x_norm = self.normalize_x(x)
        c_norm = self.normalize_condition(condition)
        n = x_norm.shape[0]
        order = self.rng.permutation(n)
        losses = []
        lr = self.config.learning_rate

        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            clean = x_norm[idx]
            cond = c_norm[idx]
            t = self.rng.integers(0, self.config.timesteps, size=len(idx))
            eps = self.rng.normal(0.0, 1.0, clean.shape)
            alpha_bar = self.alpha_bars[t][:, None]
            noisy = np.sqrt(alpha_bar) * clean + np.sqrt(1.0 - alpha_bar) * eps
            pred, hidden, inp = self._forward(noisy, cond, t)
            err = pred - eps
            loss = float(np.mean(err * err))
            losses.append(loss)

            grad_pred = 2.0 * err / err.size
            grad_w2 = hidden.T @ grad_pred
            grad_b2 = grad_pred.sum(axis=0)
            grad_hidden = grad_pred @ self.w2.T
            grad_hidden_raw = grad_hidden * (1.0 - hidden * hidden)
            grad_w1 = inp.T @ grad_hidden_raw
            grad_b1 = grad_hidden_raw.sum(axis=0)

            self.w2 -= lr * grad_w2
            self.b2 -= lr * grad_b2
            self.w1 -= lr * grad_w1
            self.b1 -= lr * grad_b1

        return float(np.mean(losses))

    def train(self, x: np.ndarray, condition: np.ndarray, epochs: int = 220, batch_size: int = 16) -> Dict[str, list]:
        self.fit_normalizers(x, condition)
        history = {"loss": []}
        for _ in range(epochs):
            history["loss"].append(self.train_epoch(x, condition, batch_size=batch_size))
        return history

    def sample(self, condition: np.ndarray, n_samples: int = 16, guidance_scale: float = 0.12) -> np.ndarray:
        if condition.ndim == 1:
            condition = np.repeat(condition[None, :], n_samples, axis=0)
        elif len(condition) != n_samples:
            condition = np.resize(condition, (n_samples, condition.shape[-1]))

        cond = self.normalize_condition(condition)
        x = self.rng.normal(0.0, 1.0, (n_samples, self.config.input_dim))

        for step in reversed(range(self.config.timesteps)):
            t = np.full(n_samples, step, dtype=int)
            pred_eps, _, _ = self._forward(x, cond, t)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            beta = self.betas[step]
            x = (x - beta * pred_eps / np.sqrt(1.0 - alpha_bar)) / np.sqrt(alpha)
            if step > 0:
                x += np.sqrt(beta) * self.rng.normal(0.0, 1.0, x.shape)
            x += guidance_scale * self._property_guidance(x, condition)

        return self.denormalize_x(x)

    def _property_guidance(self, x_norm: np.ndarray, target_condition: np.ndarray) -> np.ndarray:
        """Heuristic classifier-free-style guidance in descriptor space.

        Descriptor positions are defined in utils.geo_utils.graph_descriptor.
        Index 0:25 composition histogram; following geometry positions hold
        lattice/thickness/bond proxies. The push favors transition-metal
        chalcogenide/MXene regions associated with HER-active 2D catalysts.
        """
        target = np.zeros_like(x_norm)
        comp_start = 0
        comp_end = 25
        geometry_start = comp_end
        proto_start = geometry_start + 7

        # Emphasize Mo/W/V/Nb/Ta/Pt/Pd and S/Se/C/N species.
        preferred = [2, 3, 6, 7, 11, 12, 19, 20, 21, 22, 23, 24]
        target[:, comp_start + np.array(preferred)] = 0.10
        target[:, geometry_start + 3] = 0.60
        target[:, geometry_start + 4] = 0.66
        target[:, proto_start : proto_start + 3] = np.array([0.45, 0.35, 0.35])
        target_norm = self.normalize_x(target)
        return target_norm - x_norm

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            config=np.array(
                [
                    self.config.input_dim,
                    self.config.condition_dim,
                    self.config.hidden_dim,
                    self.config.timesteps,
                    self.config.beta_start,
                    self.config.beta_end,
                    self.config.learning_rate,
                    self.config.seed,
                ],
                dtype=float,
            ),
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            x_mean=self.x_mean,
            x_std=self.x_std,
            c_mean=self.c_mean,
            c_std=self.c_std,
        )

    @classmethod
    def load(cls, path: str | Path) -> "ConditionalGraphDiffusion":
        data = np.load(path, allow_pickle=True)
        raw = data["config"]
        config = DiffusionConfig(
            input_dim=int(raw[0]),
            condition_dim=int(raw[1]),
            hidden_dim=int(raw[2]),
            timesteps=int(raw[3]),
            beta_start=float(raw[4]),
            beta_end=float(raw[5]),
            learning_rate=float(raw[6]),
            seed=int(raw[7]),
        )
        model = cls(config)
        for name in ["w1", "b1", "w2", "b2", "x_mean", "x_std", "c_mean", "c_std"]:
            setattr(model, name, data[name])
        return model
