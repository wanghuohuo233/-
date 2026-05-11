"""Train conditional diffusion model and generate HER-active 2D materials."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from dataset.material_dataset import condition_matrix, descriptor_matrix, load_material_dataset
from models.diffusion_model import ConditionalGraphDiffusion, DiffusionConfig
from models.optimization import loss_terms, optimize_records, objective, select_diverse_records
from models.structure_generator import StructureGenerator
from utils.geo_utils import descriptor_dim, records_to_json, write_cif, write_xyz
from utils.vis import plot_generated_structures, plot_her_performance, plot_loss_curve, plot_stability_curve


def summarize(records) -> Dict[str, float]:
    abs_dg = [abs(record.properties["delta_g_h"]) for record in records]
    stability = [record.properties["stability_score"] for record in records]
    synthesis = [record.properties["synthesis_score"] for record in records]
    return {
        "avg_abs_delta_g_h": float(np.mean(abs_dg)),
        "avg_stability_score": float(np.mean(stability)),
        "avg_synthesis_score": float(np.mean(synthesis)),
        "best_objective": float(max(objective(record) for record in records)),
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
    parser = argparse.ArgumentParser(description="Conditional diffusion for HER-active 2D material generation")
    parser.add_argument("--data", default=None, help="Optional JSON/CSV dataset from 2DMatPedia, C2DB, MP, NOMAD, etc.")
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--checkpoint", default="checkpoints/conditional_graph_diffusion.npz")
    args = parser.parse_args()

    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_material_dataset(args.data, seed=args.seed)
    x = descriptor_matrix(records)
    condition = condition_matrix(records)

    model = ConditionalGraphDiffusion(
        DiffusionConfig(
            input_dim=descriptor_dim(),
            hidden_dim=144,
            timesteps=70,
            learning_rate=1.8e-3,
            seed=args.seed,
        )
    )
    history = model.train(x, condition, epochs=args.epochs, batch_size=16)
    model.save(args.checkpoint)

    target_condition = np.array([0.0, 0.86, 0.82], dtype=float)
    generated_descriptors = model.sample(target_condition, n_samples=args.samples, guidance_scale=0.08)
    generator = StructureGenerator(seed=args.seed)
    generated = generator.decode(generated_descriptors, target_condition)
    optimized = optimize_records(generated, generator, rounds=8, keep_top=18, mutations_per_record=4)
    top_records = select_diverse_records(optimized, limit=24, max_per_formula=2)

    plot_loss_curve(history, output_dir / "loss_curve.png")
    plot_her_performance(top_records, output_dir / "her_performance.png")
    plot_stability_curve(top_records, output_dir / "stability_curve.png")
    plot_generated_structures(top_records, output_dir / "generated_structures.png", max_items=10)

    saved_structures = save_structures(top_records, output_dir, count=10)
    records_to_json(top_records, output_dir / "generated_materials.json")

    baseline_top = sorted(records, key=lambda record: abs(record.properties["delta_g_h"]))[:24]
    report = {
        "dataset_size": len(records),
        "descriptor_dim": int(x.shape[1]),
        "target_condition": {
            "delta_g_h": 0.0,
            "stability_score": 0.86,
            "synthesis_score": 0.82,
        },
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
            "final_loss": float(history["loss"][-1]),
            "checkpoint": str(Path(args.checkpoint)),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Training complete")
    print(f"Dataset records: {len(records)}")
    print(f"Final loss: {history['loss'][-1]:.6f}")
    print(f"Top avg |ΔG_H|: {report['ours_summary']['avg_abs_delta_g_h']:.4f} eV")
    print(f"Top avg stability: {report['ours_summary']['avg_stability_score']:.4f}")
    print(f"Top avg synthesis: {report['ours_summary']['avg_synthesis_score']:.4f}")
    print(f"Saved figures and structures to: {output_dir.resolve()}")


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
        from train_torch import main as torch_main

        torch_main()
    elif backend in {"numpy", "lightweight"}:
        main()
    else:
        raise ValueError(f"Unknown backend: {backend}. Use torch or numpy.")


if __name__ == "__main__":
    _dispatch_backend()
