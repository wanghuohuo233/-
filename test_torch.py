"""Smoke test for the GPU GNN diffusion checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dataset.graph_dataset import build_graph_batch
from dataset.material_dataset import load_material_dataset
from models.optimization import optimize_records
from models.structure_generator import StructureGenerator
from models.torch_gnn_diffusion import load_torch_checkpoint
from utils.geo_utils import write_cif


def main() -> None:
    parser = argparse.ArgumentParser(description="Test trained Torch GNN diffusion checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/torch_gnn_diffusion.pt")
    parser.add_argument("--data", default="data/c2dm_public_2d.json")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--output-dir", default="results/test_torch_samples")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}. Run train_torch.py first.")

    records = load_material_dataset(args.data)
    batch = build_graph_batch(records, max_atoms=8)
    model = load_torch_checkpoint(checkpoint, device=device)
    rng = np.random.default_rng(17)
    idx = rng.choice(len(records), size=args.samples, replace=True)
    nodes = torch.tensor(batch.node_features[idx], dtype=torch.float32, device=device)
    adj = torch.tensor(batch.adjacency[idx], dtype=torch.float32, device=device)
    mask = torch.tensor(batch.mask[idx], dtype=torch.float32, device=device)
    target = torch.tensor([0.0, 0.88, 0.84], dtype=torch.float32, device=device)
    with torch.no_grad():
        descriptors = model.sample(target, nodes, adj, mask, guidance_scale=0.04).detach().cpu().numpy()

    generator = StructureGenerator(seed=17)
    generated = generator.decode(descriptors, target.detach().cpu().numpy())
    optimized = optimize_records(generated, generator, rounds=3, keep_top=10, mutations_per_record=2)[:5]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("*.cif"):
        stale.unlink()
    for i, record in enumerate(optimized, start=1):
        write_cif(record, out_dir / f"torch_test_{i:02d}_{record.formula}.cif")

    best = optimized[0]
    assert abs(best.properties["delta_g_h"]) < 0.25
    assert best.properties["stability_score"] > 0.45
    assert best.properties["synthesis_score"] > 0.45
    print("Torch GNN smoke test passed")
    print(f"Device: {device}")
    print(f"Best: {best.formula} ΔG_H={best.properties['delta_g_h']:.4f} eV, "
          f"stability={best.properties['stability_score']:.4f}, "
          f"synthesis={best.properties['synthesis_score']:.4f}")


if __name__ == "__main__":
    main()
