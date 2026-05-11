"""Smoke tests and regeneration check for the trained model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from models.diffusion_model import ConditionalGraphDiffusion
from models.structure_generator import StructureGenerator
from models.optimization import optimize_records
from utils.geo_utils import write_cif


def main() -> None:
    parser = argparse.ArgumentParser(description="Test trained diffusion checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/conditional_graph_diffusion.npz")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--output-dir", default="results/test_samples")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}. Run train.py first.")

    model = ConditionalGraphDiffusion.load(checkpoint)
    target = np.array([0.0, 0.86, 0.82], dtype=float)
    descriptors = model.sample(target, n_samples=args.samples, guidance_scale=0.08)
    generator = StructureGenerator(seed=17)
    records = generator.decode(descriptors, target)
    optimized = optimize_records(records, generator, rounds=3, keep_top=8, mutations_per_record=2)[:5]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, record in enumerate(optimized, start=1):
        write_cif(record, out_dir / f"test_{idx:02d}_{record.formula}.cif")

    best = optimized[0]
    assert abs(best.properties["delta_g_h"]) < 0.25, "HER surrogate is not close enough to 0 eV"
    assert best.properties["stability_score"] > 0.45, "Stability surrogate is too low"
    assert best.properties["synthesis_score"] > 0.45, "Synthesis surrogate is too low"

    print("Smoke test passed")
    print(f"Best: {best.formula} ΔG_H={best.properties['delta_g_h']:.4f} eV, "
          f"stability={best.properties['stability_score']:.4f}, "
          f"synthesis={best.properties['synthesis_score']:.4f}")


def _dispatch_backend() -> None:
    backend = "torch"
    cleaned = [sys.argv[0]]
    skip_next = False
    for idx, arg in enumerate(sys.argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        if arg == "--backend" and idx + 1 < len(sys.argv):
            backend = sys.argv[idx + 1].lower()
            skip_next = True
            continue
        if arg.startswith("--backend="):
            backend = arg.split("=", 1)[1].lower()
            continue
        cleaned.append(arg)
    sys.argv = cleaned
    if backend in {"torch", "gnn", "cuda"}:
        from test_torch import main as torch_main

        torch_main()
    elif backend in {"numpy", "lightweight"}:
        main()
    else:
        raise ValueError(f"Unknown backend: {backend}. Use torch or numpy.")


if __name__ == "__main__":
    _dispatch_backend()
