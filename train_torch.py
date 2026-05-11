"""GPU PyTorch training entry for the GNN-conditioned diffusion model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Dict, List

import numpy as np
import torch

from dataset.graph_dataset import build_graph_batch
from dataset.material_dataset import load_material_dataset
from models.optimization import loss_terms, objective, optimize_records, select_diverse_records
from models.structure_generator import StructureGenerator
from models.torch_gnn_diffusion import (
    TorchDiffusionConfig,
    TorchGNNConditionedDiffusion,
    save_torch_checkpoint,
)
from utils.geo_utils import records_to_json, write_cif, write_xyz
from utils.vis import plot_generated_structures, plot_her_performance, plot_loss_curve, plot_stability_curve


def summarize(records) -> Dict[str, float]:
    return {
        "avg_abs_delta_g_h": float(np.mean([abs(r.properties["delta_g_h"]) for r in records])),
        "avg_stability_score": float(np.mean([r.properties["stability_score"] for r in records])),
        "avg_synthesis_score": float(np.mean([r.properties["synthesis_score"] for r in records])),
        "best_objective": float(max(objective(r) for r in records)),
    }


def save_structures(records, output_dir: Path, count: int = 10) -> List[Dict[str, str]]:
    structure_dir = output_dir / "generated_structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    for stale in list(structure_dir.glob("*.cif")) + list(structure_dir.glob("*.xyz")):
        stale.unlink()
    saved = []
    for idx, record in enumerate(records[:count], start=1):
        safe_name = f"{idx:02d}_{record.formula}_{record.name}".replace("/", "_")
        cif_path = structure_dir / f"{safe_name}.cif"
        xyz_path = structure_dir / f"{safe_name}.xyz"
        write_cif(record, cif_path)
        write_xyz(record, xyz_path)
        saved.append({"name": record.name, "formula": record.formula, "cif": str(cif_path), "xyz": str(xyz_path)})
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GPU GNN-conditioned diffusion for 2D HER material generation")
    parser.add_argument("--data", default="data/c2dm_public_2d.json")
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--samples", type=int, default=160)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--checkpoint", default="checkpoints/torch_gnn_diffusion.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--graph-hidden-dim", type=int, default=96)
    parser.add_argument("--denoiser-hidden-dim", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--guidance-scale", type=float, default=0.04)
    parser.add_argument("--target-stability", type=float, default=0.88)
    parser.add_argument("--target-synthesis", type=float, default=0.84)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_material_dataset(args.data, seed=args.seed)
    if len(records) < 1000:
        print(
            "warning: dataset has fewer than 1000 records; this is enough for a runnable "
            "pipeline, but expanded JARVIS/C2DB data is recommended before increasing model size."
        )
    batch = build_graph_batch(records, max_atoms=8)
    descriptors = torch.tensor(batch.descriptors, dtype=torch.float32, device=device)
    conditions = torch.tensor(batch.conditions, dtype=torch.float32, device=device)
    node_features = torch.tensor(batch.node_features, dtype=torch.float32, device=device)
    adjacency = torch.tensor(batch.adjacency, dtype=torch.float32, device=device)
    mask = torch.tensor(batch.mask, dtype=torch.float32, device=device)

    model = TorchGNNConditionedDiffusion(
        TorchDiffusionConfig(
            descriptor_dim=descriptors.shape[1],
            graph_hidden_dim=args.graph_hidden_dim,
            denoiser_hidden_dim=args.denoiser_hidden_dim,
            timesteps=args.timesteps,
            learning_rate=args.lr,
            seed=args.seed,
        )
    ).to(device)
    model.fit_normalizers(descriptors, conditions)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.config.learning_rate, weight_decay=args.weight_decay)

    n = descriptors.shape[0]
    history = {"loss": []}
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(n, device=device)
        epoch_losses = []
        model.train()
        for start in range(0, n, args.batch_size):
            idx = order[start : start + args.batch_size]
            loss = model.training_loss(descriptors[idx], conditions[idx], node_features[idx], adjacency[idx], mask[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        history["loss"].append(float(np.mean(epoch_losses)))
        if epoch % max(1, args.epochs // 5) == 0:
            print(f"epoch {epoch:04d}/{args.epochs} loss={history['loss'][-1]:.6f}")

    model.eval()
    save_torch_checkpoint(
        model,
        args.checkpoint,
        extra={"dataset": args.data, "device": str(device), "epochs": args.epochs, "records": len(records)},
    )

    rng = np.random.default_rng(args.seed)
    seed_idx = rng.choice(n, size=args.samples, replace=True)
    sample_nodes = node_features[torch.tensor(seed_idx, device=device)]
    sample_adj = adjacency[torch.tensor(seed_idx, device=device)]
    sample_mask = mask[torch.tensor(seed_idx, device=device)]
    target_condition = torch.tensor([0.0, args.target_stability, args.target_synthesis], dtype=torch.float32, device=device)
    with torch.no_grad():
        sampled = model.sample(target_condition, sample_nodes, sample_adj, sample_mask, guidance_scale=args.guidance_scale)
    descriptors_np = sampled.detach().cpu().numpy()

    generator = StructureGenerator(seed=args.seed)
    generated = generator.decode(descriptors_np, target_condition.detach().cpu().numpy())
    optimized = optimize_records(generated, generator, rounds=8, keep_top=24, mutations_per_record=4)
    top_records = select_diverse_records(optimized, limit=24, max_per_formula=2)

    plot_loss_curve(history, output_dir / "loss_curve.png")
    plot_her_performance(top_records, output_dir / "her_performance.png")
    plot_stability_curve(top_records, output_dir / "stability_curve.png")
    plot_generated_structures(top_records, output_dir / "generated_structures.png", max_items=10)
    saved_structures = save_structures(top_records, output_dir, count=10)
    records_to_json(top_records, output_dir / "generated_materials.json")

    baseline_top = sorted(records, key=lambda record: abs(record.properties["delta_g_h"]))[:24]
    report = {
        "backend": "torch_gnn_cuda" if device.type == "cuda" else "torch_gnn_cpu",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dataset_path": args.data,
        "dataset_size": len(records),
        "descriptor_dim": int(descriptors.shape[1]),
        "target_condition": {"delta_g_h": 0.0, "stability_score": 0.88, "synthesis_score": 0.84},
        "baseline_seed_summary": summarize(baseline_top),
        "ours_summary": summarize(top_records),
        "best_candidates": [
            {
                "rank": idx,
                "name": record.name,
                "formula": record.formula,
                "prototype": record.prototype,
                "properties": record.properties,
                "loss_terms": loss_terms(record.properties),
            }
            for idx, record in enumerate(top_records[:10], start=1)
        ],
        "saved_structures": saved_structures,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "final_loss": float(history["loss"][-1]),
            "checkpoint": str(Path(args.checkpoint)),
            "graph_hidden_dim": args.graph_hidden_dim,
            "denoiser_hidden_dim": args.denoiser_hidden_dim,
            "timesteps": args.timesteps,
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "guidance_scale": args.guidance_scale,
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("GPU GNN diffusion training complete")
    print(f"Device: {device}")
    print(f"Dataset records: {len(records)}")
    print(f"Final loss: {history['loss'][-1]:.6f}")
    print(f"Top avg |ΔG_H|: {report['ours_summary']['avg_abs_delta_g_h']:.4f} eV")
    print(f"Top avg stability: {report['ours_summary']['avg_stability_score']:.4f}")
    print(f"Top avg synthesis: {report['ours_summary']['avg_synthesis_score']:.4f}")


if __name__ == "__main__":
    main()
