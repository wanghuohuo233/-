"""Decode diffusion descriptors into simple 2D crystal structures."""

from __future__ import annotations

from dataclasses import replace
import random
from typing import Iterable, List, Sequence

import numpy as np

from dataset.material_dataset import MaterialRecord, build_structure_from_prototype
from utils.geo_utils import ELEMENTS, METALS, ANIONS, PROTOTYPES, evaluate_material, formula_from_elements


PREFERRED_METALS = ["Mo", "W", "V", "Nb", "Ta", "Ti", "Pt", "Pd", "Ni", "Co", "Fe", "Cr", "Mn", "Cu"]
PREFERRED_ANIONS = ["S", "Se", "N", "C", "O", "P", "Te", "B"]


class StructureGenerator:
    """Converts continuous model samples into chemically plausible prototypes."""

    def __init__(self, seed: int = 7):
        self.rng = random.Random(seed)

    def _pick_elements(self, descriptor: np.ndarray) -> tuple[str, str, str]:
        comp = np.asarray(descriptor[: len(ELEMENTS)], dtype=float)
        comp = np.nan_to_num(comp, nan=0.0, posinf=0.0, neginf=0.0)
        comp = np.maximum(comp, 0.0)
        if comp.sum() <= 1e-8:
            comp = np.ones_like(comp)

        ranked = [ELEMENTS[i] for i in np.argsort(comp)[::-1]]
        metals = [e for e in ranked if e in METALS and e in PREFERRED_METALS]
        anions = [e for e in ranked if e in ANIONS and e in PREFERRED_ANIONS]
        if not metals:
            metals = list(PREFERRED_METALS)
        if not anions:
            anions = list(PREFERRED_ANIONS)

        metal = metals[0]
        anion_a = anions[0]
        anion_b = anions[1] if len(anions) > 1 else anion_a
        return metal, anion_a, anion_b

    def _pick_prototype(self, descriptor: np.ndarray, target_condition: Sequence[float]) -> str:
        proto_start = len(ELEMENTS) + 7
        proto_scores = descriptor[proto_start : proto_start + len(PROTOTYPES)]
        if len(proto_scores) != len(PROTOTYPES) or np.allclose(proto_scores, proto_scores[0]):
            target_dg = abs(float(target_condition[0])) if len(target_condition) else 0.0
            return "Janus-MXY" if target_dg < 0.08 else "MX2"
        prototype = PROTOTYPES[int(np.argmax(proto_scores))]
        if prototype == "Other":
            prototype = "MX2"
        return prototype

    def decode_one(self, descriptor: np.ndarray, target_condition: Sequence[float], index: int = 0) -> MaterialRecord:
        metal, anion_a, anion_b = self._pick_elements(descriptor)
        prototype = self._pick_prototype(descriptor, target_condition)
        geometry_start = len(ELEMENTS)
        geom = descriptor[geometry_start : geometry_start + 7]

        a = float(np.clip(abs(geom[0]) * 6.0 if len(geom) > 0 else 3.2, 2.4, 4.2))
        if prototype == "MXene":
            a = float(np.clip(a, 2.8, 3.4))
        thickness = float(np.clip(abs(geom[3]) * 5.0 if len(geom) > 3 else 3.0, 0.2, 4.6))
        if prototype in {"MX2", "Janus-MXY"}:
            thickness = float(np.clip(thickness, 2.5, 3.8))
        if prototype == "MXene":
            thickness = float(np.clip(thickness, 1.8, 2.8))

        record = build_structure_from_prototype(
            prototype=prototype,
            metal=metal,
            anion_a=anion_a,
            anion_b=anion_b,
            a=a,
            thickness=thickness,
            name=f"gen_{index:03d}_{metal}_{anion_a}_{anion_b}_{prototype.replace('-', '')}",
            source="conditional_diffusion",
        )
        record.properties = evaluate_material(record)
        record.formula = formula_from_elements(record.elements)
        return record

    def decode(self, descriptors: np.ndarray, target_condition: Sequence[float]) -> List[MaterialRecord]:
        return [self.decode_one(row, target_condition, i) for i, row in enumerate(descriptors, start=1)]

    def mutate_record(self, record: MaterialRecord, index: int, temperature: float = 0.18) -> MaterialRecord:
        """Small local search mutation used by the optimizer."""
        elements = list(record.elements)
        metal_positions = [i for i, element in enumerate(elements) if element in METALS]
        anion_positions = [i for i, element in enumerate(elements) if element in ANIONS]
        if metal_positions and self.rng.random() < temperature:
            elements[metal_positions[0]] = self.rng.choice(PREFERRED_METALS[:8])
        if anion_positions and self.rng.random() < temperature:
            elements[anion_positions[-1]] = self.rng.choice(PREFERRED_ANIONS[:5])

        new_record = replace(record)
        new_record.name = f"opt_{index:03d}_{record.name}"
        new_record.elements = elements
        new_record.positions = [[float(v) for v in pos] for pos in record.positions]
        new_record.lattice = [[float(v) for v in vec] for vec in record.lattice]

        strain = 1.0 + self.rng.uniform(-0.05, 0.05)
        z_scale = 1.0 + self.rng.uniform(-0.07, 0.07)
        for vec in new_record.lattice[:2]:
            vec[0] *= strain
            vec[1] *= strain
        for pos in new_record.positions:
            pos[0] *= strain
            pos[1] *= strain
            pos[2] *= z_scale

        new_record.formula = formula_from_elements(new_record.elements)
        new_record.properties = evaluate_material(new_record)
        return new_record
