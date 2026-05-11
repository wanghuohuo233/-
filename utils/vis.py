"""Visualization utilities for training and generated materials."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np

from dataset.material_dataset import MaterialRecord
from utils.geo_utils import METALS


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_loss_curve(history: Dict[str, List[float]], path: str | Path) -> None:
    path = _ensure_parent(path)
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=160)
    loss = history.get("loss", [])
    ax.plot(np.arange(1, len(loss) + 1), loss, color="#2563eb", linewidth=2.2)
    ax.set_title("Conditional Diffusion Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Noise prediction MSE")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_her_performance(records: Iterable[MaterialRecord], path: str | Path) -> None:
    path = _ensure_parent(path)
    records = list(records)
    values = np.array([record.properties["delta_g_h"] for record in records], dtype=float)
    names = [f"#{idx:02d} {record.formula}" for idx, record in enumerate(records, start=1)]
    order = np.argsort(np.abs(values))
    values = values[order]
    names = [names[i] for i in order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6), dpi=160, gridspec_kw={"width_ratios": [1.1, 1.5]})
    ax1.hist(values, bins=12, color="#14b8a6", edgecolor="white")
    ax1.axvline(0.0, color="#111827", linestyle="--", linewidth=1.4)
    ax1.set_title("HER ΔG_H Distribution")
    ax1.set_xlabel("ΔG_H (eV), target = 0")
    ax1.set_ylabel("Count")
    ax1.grid(axis="y", alpha=0.25)

    top_n = min(10, len(values))
    colors = ["#22c55e" if abs(v) <= 0.10 else "#f97316" for v in values[:top_n]]
    ax2.barh(np.arange(top_n), values[:top_n], color=colors)
    ax2.axvline(0.0, color="#111827", linestyle="--", linewidth=1.2)
    ax2.set_yticks(np.arange(top_n), names[:top_n], fontsize=8)
    ax2.invert_yaxis()
    ax2.set_title("Top Generated Candidates")
    ax2.set_xlabel("ΔG_H (eV)")
    ax2.set_ylabel("Candidate rank and formula")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_stability_curve(records: Iterable[MaterialRecord], path: str | Path) -> None:
    path = _ensure_parent(path)
    records = list(records)
    xs = np.arange(1, len(records) + 1)
    stability = np.array([record.properties["stability_score"] for record in records])
    synthesis = np.array([record.properties["synthesis_score"] for record in records])
    thermo = np.array([record.properties["thermodynamic_stability"] for record in records])
    kinetic = np.array([record.properties["kinetic_stability"] for record in records])

    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=160)
    ax.plot(xs, stability, label="Combined stability", color="#2563eb", linewidth=2.0)
    ax.plot(xs, thermo, label="Thermodynamic", color="#7c3aed", linewidth=1.6, alpha=0.85)
    ax.plot(xs, kinetic, label="Kinetic", color="#06b6d4", linewidth=1.6, alpha=0.85)
    ax.plot(xs, synthesis, label="Synthesis score", color="#f97316", linewidth=2.0)
    ax.set_ylim(0.0, 1.03)
    ax.set_xlabel("Ranked generated material")
    ax.set_ylabel("Score")
    ax.set_title("Stability and Synthesizability")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_generated_structures(records: Iterable[MaterialRecord], path: str | Path, max_items: int = 10) -> None:
    path = _ensure_parent(path)
    records = list(records)[:max_items]
    cols = 5
    rows = int(np.ceil(len(records) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.35), dpi=170)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    color_map = {
        "Mo": "#2563eb", "W": "#1d4ed8", "V": "#0f766e", "Nb": "#0891b2", "Ta": "#4f46e5",
        "Ti": "#0284c7", "Pt": "#475569", "Pd": "#64748b", "Ni": "#16a34a", "Co": "#15803d",
        "Fe": "#b45309", "S": "#facc15", "Se": "#fb923c", "Te": "#ef4444", "C": "#111827",
        "N": "#38bdf8", "P": "#a855f7", "B": "#f472b6",
    }

    for ax in axes.flat:
        ax.axis("off")

    for ax, record in zip(axes.flat, records):
        positions = np.asarray(record.positions)
        xy = positions[:, :2]
        xy = xy - xy.mean(axis=0)
        if np.ptp(xy[:, 0]) < 1e-6:
            xy[:, 0] += np.linspace(-0.2, 0.2, len(xy))
        if np.ptp(xy[:, 1]) < 1e-6:
            xy[:, 1] += np.linspace(-0.2, 0.2, len(xy))

        for i in range(len(record.elements)):
            for j in range(i + 1, len(record.elements)):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < 3.1:
                    ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]], color="#94a3b8", linewidth=1.1, zorder=1)

        for element, (x, y) in zip(record.elements, xy):
            size = 260 if element in METALS else 180
            ax.scatter(x, y, s=size, c=color_map.get(element, "#64748b"), edgecolor="white", linewidth=1.2, zorder=2)
            ax.text(x, y, element, ha="center", va="center", fontsize=7, color="white" if element in METALS or element == "C" else "#111827", zorder=3)

        ax.set_title(
            f"{record.formula}\nΔG={record.properties['delta_g_h']:.2f} eV, S={record.properties['stability_score']:.2f}",
            fontsize=8,
        )
        pad = 0.9
        ax.set_xlim(float(xy[:, 0].min() - pad), float(xy[:, 0].max() + pad))
        ax.set_ylim(float(xy[:, 1].min() - pad), float(xy[:, 1].max() + pad))
        ax.set_aspect("equal")

    fig.suptitle("Generated 2D Material Structures", fontsize=12, y=0.995)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
