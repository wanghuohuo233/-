"""Multi-objective optimization for generated 2D materials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from dataset.material_dataset import MaterialRecord
from models.structure_generator import StructureGenerator
from utils.geo_utils import evaluate_material


@dataclass
class ObjectiveWeights:
    her: float = 0.48
    stability: float = 0.34
    synthesis: float = 0.18


def objective(record: MaterialRecord, weights: ObjectiveWeights = ObjectiveWeights()) -> float:
    metrics = record.properties or evaluate_material(record)
    her_score = float(np.exp(-abs(metrics["delta_g_h"]) / 0.12))
    return (
        weights.her * her_score
        + weights.stability * metrics["stability_score"]
        + weights.synthesis * metrics["synthesis_score"]
    )


def loss_terms(metrics: Dict[str, float], target_delta_g: float = 0.0) -> Dict[str, float]:
    her_loss = abs(metrics["delta_g_h"] - target_delta_g)
    stability_loss = 1.0 - metrics["stability_score"]
    synthesis_loss = 1.0 - metrics["synthesis_score"]
    total = 0.48 * her_loss + 0.34 * stability_loss + 0.18 * synthesis_loss
    return {
        "her_loss": float(her_loss),
        "stability_loss": float(stability_loss),
        "synthesis_loss": float(synthesis_loss),
        "total_loss": float(total),
    }


def pareto_rank(records: Iterable[MaterialRecord]) -> List[Tuple[MaterialRecord, float]]:
    ranked = []
    for record in records:
        record.properties = evaluate_material(record)
        ranked.append((record, objective(record)))
    return sorted(ranked, key=lambda item: item[1], reverse=True)


def select_diverse_records(records: Iterable[MaterialRecord], limit: int = 24, max_per_formula: int = 2) -> List[MaterialRecord]:
    """Select high-scoring records while avoiding one-formula mode collapse."""
    ranked = pareto_rank(records)
    selected: List[MaterialRecord] = []
    formula_counts: Dict[str, int] = {}
    prototype_counts: Dict[str, int] = {}

    for record, _ in ranked:
        formula_count = formula_counts.get(record.formula, 0)
        prototype_count = prototype_counts.get(record.prototype, 0)
        if formula_count < max_per_formula and prototype_count < max(3, limit // 2):
            selected.append(record)
            formula_counts[record.formula] = formula_count + 1
            prototype_counts[record.prototype] = prototype_count + 1
        if len(selected) >= limit:
            return selected

    for record, _ in ranked:
        if record not in selected:
            selected.append(record)
        if len(selected) >= limit:
            break
    return selected


def optimize_records(
    records: List[MaterialRecord],
    generator: StructureGenerator,
    rounds: int = 7,
    keep_top: int = 16,
    mutations_per_record: int = 4,
) -> List[MaterialRecord]:
    """Evolutionary local search over generated structures."""
    pool = list(records)
    counter = 1
    for _ in range(rounds):
        parents = select_diverse_records(pool, limit=keep_top, max_per_formula=2)
        children: List[MaterialRecord] = []
        for parent in parents:
            for _ in range(mutations_per_record):
                children.append(generator.mutate_record(parent, counter, temperature=0.28))
                counter += 1
        pool = parents + children
    return [record for record, _ in pareto_rank(pool)]
