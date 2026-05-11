"""Material dataset loading and small public-database-style seed set.

The loader accepts JSON/CSV data exported from public resources such as
2DMatPedia, C2DB, Materials Project, or NOMAD. When no external file is given,
it builds a compact 2D seed dataset from common experimentally studied families
so the whole repository can run end-to-end without credentials.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import csv
import json
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from utils.geo_utils import evaluate_material, formula_from_elements, update_record_properties


@dataclass
class MaterialRecord:
    name: str
    formula: str
    prototype: str
    elements: List[str]
    positions: List[List[float]]
    lattice: List[List[float]]
    properties: Dict[str, float] = field(default_factory=dict)
    source: str = "seed"


def hex_lattice(a: float, vacuum: float = 18.0) -> List[List[float]]:
    return [
        [a, 0.0, 0.0],
        [-0.5 * a, math.sqrt(3.0) * 0.5 * a, 0.0],
        [0.0, 0.0, vacuum],
    ]


def rect_lattice(a: float, b: float, vacuum: float = 18.0) -> List[List[float]]:
    return [[a, 0.0, 0.0], [0.0, b, 0.0], [0.0, 0.0, vacuum]]


def build_structure_from_prototype(
    prototype: str,
    metal: str,
    anion_a: str,
    anion_b: Optional[str] = None,
    a: float = 3.18,
    thickness: float = 3.1,
    name: Optional[str] = None,
    source: str = "generated",
) -> MaterialRecord:
    """Build simple 2D prototype cells in Cartesian coordinates."""
    anion_b = anion_b or anion_a
    prototype = prototype if prototype in {"MX2", "Janus-MXY", "MXene", "Binary", "Elemental"} else "MX2"

    if prototype == "MX2":
        elements = [metal, anion_a, anion_a]
        positions = [
            [0.0, 0.0, 0.0],
            [0.5 * a, math.sqrt(3.0) * a / 6.0, 0.5 * thickness],
            [0.5 * a, math.sqrt(3.0) * a / 6.0, -0.5 * thickness],
        ]
        lattice = hex_lattice(a)
    elif prototype == "Janus-MXY":
        elements = [metal, anion_a, anion_b]
        positions = [
            [0.0, 0.0, 0.0],
            [0.5 * a, math.sqrt(3.0) * a / 6.0, 0.5 * thickness],
            [0.5 * a, math.sqrt(3.0) * a / 6.0, -0.5 * thickness],
        ]
        lattice = hex_lattice(a)
    elif prototype == "MXene":
        elements = [metal, metal, anion_a, anion_b]
        positions = [
            [0.0, 0.0, 0.45 * thickness],
            [0.5 * a, 0.5 * a, -0.45 * thickness],
            [0.5 * a, 0.0, 0.0],
            [0.0, 0.5 * a, 0.0],
        ]
        lattice = rect_lattice(a, a)
    elif prototype == "Binary":
        elements = [metal, anion_a]
        positions = [[0.0, 0.0, 0.0], [0.5 * a, 0.5 * a, 0.05 * thickness]]
        lattice = rect_lattice(a, a)
    else:
        elements = [anion_a, anion_a]
        positions = [[0.0, 0.0, 0.0], [0.5 * a, math.sqrt(3.0) * a / 6.0, 0.0]]
        lattice = hex_lattice(a)

    record = MaterialRecord(
        name=name or f"{prototype}_{metal}_{anion_a}_{anion_b}",
        formula=formula_from_elements(elements),
        prototype=prototype,
        elements=elements,
        positions=positions,
        lattice=lattice,
        source=source,
    )
    record.properties = evaluate_material(record)
    return record


def _seed_specs() -> List[Dict[str, object]]:
    return [
        {"prototype": "MX2", "metal": "Mo", "anion_a": "S", "a": 3.18, "thickness": 3.12, "name": "MoS2_2H"},
        {"prototype": "MX2", "metal": "W", "anion_a": "S", "a": 3.19, "thickness": 3.14, "name": "WS2_2H"},
        {"prototype": "MX2", "metal": "Mo", "anion_a": "Se", "a": 3.32, "thickness": 3.34, "name": "MoSe2_2H"},
        {"prototype": "MX2", "metal": "W", "anion_a": "Se", "a": 3.31, "thickness": 3.35, "name": "WSe2_2H"},
        {"prototype": "MX2", "metal": "V", "anion_a": "S", "a": 3.22, "thickness": 2.95, "name": "VS2_1T"},
        {"prototype": "MX2", "metal": "Nb", "anion_a": "S", "a": 3.34, "thickness": 3.05, "name": "NbS2_2H"},
        {"prototype": "MX2", "metal": "Ta", "anion_a": "S", "a": 3.36, "thickness": 3.05, "name": "TaS2_2H"},
        {"prototype": "MX2", "metal": "Ti", "anion_a": "S", "a": 3.42, "thickness": 2.85, "name": "TiS2_1T"},
        {"prototype": "MX2", "metal": "Pt", "anion_a": "Se", "a": 3.72, "thickness": 2.80, "name": "PtSe2_1T"},
        {"prototype": "MX2", "metal": "Pd", "anion_a": "Se", "a": 3.76, "thickness": 2.82, "name": "PdSe2_1T"},
        {"prototype": "Janus-MXY", "metal": "Mo", "anion_a": "S", "anion_b": "Se", "a": 3.25, "thickness": 3.25, "name": "MoSSe_Janus"},
        {"prototype": "Janus-MXY", "metal": "W", "anion_a": "S", "anion_b": "Se", "a": 3.24, "thickness": 3.28, "name": "WSSe_Janus"},
        {"prototype": "Janus-MXY", "metal": "Mo", "anion_a": "S", "anion_b": "Te", "a": 3.44, "thickness": 3.55, "name": "MoSTe_Janus"},
        {"prototype": "Janus-MXY", "metal": "V", "anion_a": "S", "anion_b": "Se", "a": 3.30, "thickness": 3.10, "name": "VSSe_Janus"},
        {"prototype": "MXene", "metal": "Mo", "anion_a": "C", "anion_b": "N", "a": 3.02, "thickness": 2.25, "name": "Mo2CN_MXene"},
        {"prototype": "MXene", "metal": "W", "anion_a": "C", "anion_b": "N", "a": 3.08, "thickness": 2.30, "name": "W2CN_MXene"},
        {"prototype": "MXene", "metal": "V", "anion_a": "C", "anion_b": "N", "a": 2.92, "thickness": 2.18, "name": "V2CN_MXene"},
        {"prototype": "MXene", "metal": "Ti", "anion_a": "C", "anion_b": "N", "a": 3.00, "thickness": 2.20, "name": "Ti2CN_MXene"},
        {"prototype": "Binary", "metal": "Ni", "anion_a": "P", "a": 3.45, "thickness": 1.20, "name": "NiP_sheet"},
        {"prototype": "Binary", "metal": "Co", "anion_a": "P", "a": 3.42, "thickness": 1.18, "name": "CoP_sheet"},
        {"prototype": "Binary", "metal": "Fe", "anion_a": "Se", "a": 3.70, "thickness": 1.22, "name": "FeSe_sheet"},
        {"prototype": "Elemental", "metal": "C", "anion_a": "C", "a": 2.46, "thickness": 0.10, "name": "Graphene"},
        {"prototype": "Elemental", "metal": "B", "anion_a": "N", "a": 2.50, "thickness": 0.10, "name": "hBN_like"},
    ]


def create_seed_dataset(seed: int = 7, augment: bool = True) -> List[MaterialRecord]:
    rng = random.Random(seed)
    records: List[MaterialRecord] = []
    for spec in _seed_specs():
        record = build_structure_from_prototype(source="public_2d_seed", **spec)
        records.append(record)

    if augment:
        base = list(records)
        for idx, record in enumerate(base):
            for aug_id in range(2):
                new_record = MaterialRecord(
                    name=f"{record.name}_strain{aug_id + 1}",
                    formula=record.formula,
                    prototype=record.prototype,
                    elements=list(record.elements),
                    positions=[list(p) for p in record.positions],
                    lattice=[list(v) for v in record.lattice],
                    source="augmented_seed",
                )
                strain = 1.0 + rng.uniform(-0.045, 0.045)
                z_scale = 1.0 + rng.uniform(-0.055, 0.055)
                for vec in new_record.lattice[:2]:
                    vec[0] *= strain
                    vec[1] *= strain
                for pos in new_record.positions:
                    pos[0] *= strain
                    pos[1] *= strain
                    pos[2] *= z_scale
                update_record_properties(new_record)
                records.append(new_record)
    return records


def _record_from_json_item(item: Dict[str, object]) -> MaterialRecord:
    record = MaterialRecord(
        name=str(item.get("name", item.get("material_id", "material"))),
        formula=str(item.get("formula", "")),
        prototype=str(item.get("prototype", item.get("structure_type", "Other"))),
        elements=list(item.get("elements", [])),
        positions=[list(map(float, p)) for p in item.get("positions", [])],
        lattice=[list(map(float, v)) for v in item.get("lattice", [])],
        properties=dict(item.get("properties", {})),
        source=str(item.get("source", "external")),
    )
    if not record.formula:
        record.formula = formula_from_elements(record.elements)
    if not record.properties:
        record.properties = evaluate_material(record)
    return record


def load_material_dataset(data_path: Optional[str] = None, seed: int = 7) -> List[MaterialRecord]:
    """Load external JSON/CSV data or return the built-in seed set."""
    if not data_path:
        return create_seed_dataset(seed=seed, augment=True)

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {path}")

    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        items = raw["materials"] if isinstance(raw, dict) and "materials" in raw else raw
        return [_record_from_json_item(item) for item in items]

    if path.suffix.lower() == ".csv":
        records: List[MaterialRecord] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                elements = [e.strip() for e in row.get("elements", "").split() if e.strip()]
                lattice = json.loads(row.get("lattice", "[]"))
                positions = json.loads(row.get("positions", "[]"))
                record = MaterialRecord(
                    name=row.get("name", row.get("material_id", "material")),
                    formula=row.get("formula", formula_from_elements(elements)),
                    prototype=row.get("prototype", "Other"),
                    elements=elements,
                    positions=positions,
                    lattice=lattice,
                    source=row.get("source", "external_csv"),
                )
                update_record_properties(record)
                records.append(record)
        return records

    raise ValueError("Supported dataset formats are JSON and CSV.")


def descriptor_matrix(records: Iterable[MaterialRecord]) -> np.ndarray:
    from utils.geo_utils import graph_descriptor

    return np.vstack([graph_descriptor(record) for record in records])


def condition_matrix(records: Iterable[MaterialRecord]) -> np.ndarray:
    rows = []
    for record in records:
        metrics = record.properties or evaluate_material(record)
        rows.append([metrics["delta_g_h"], metrics["stability_score"], metrics["synthesis_score"]])
    return np.asarray(rows, dtype=float)
