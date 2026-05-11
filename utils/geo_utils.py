"""Geometry, descriptor, and surrogate evaluation utilities.

The project is intentionally lightweight and runnable on a laptop. The HER,
stability, and synthesizability values are interpretable surrogate scores, not
DFT outputs. The code is structured so a later workflow can replace these
functions with DFT, phonon, AIMD, or database-backed labels.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json
import math
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


ELEMENTS: List[str] = [
    "B", "C", "N", "O", "F", "P", "S", "Se", "Te", "Cl", "I",
    "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Nb", "Mo",
    "Ta", "W", "Pd", "Pt",
]

METALS = {"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Nb", "Mo", "Ta", "W", "Pd", "Pt"}
ANIONS = {"B", "C", "N", "O", "F", "P", "S", "Se", "Te", "Cl", "I"}
PROTOTYPES = ["MX2", "Janus-MXY", "MXene", "Binary", "Elemental", "Other"]

ATOMIC_NUMBER: Dict[str, int] = {
    "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16,
    "Cl": 17, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Se": 34, "Nb": 41, "Mo": 42,
    "Pd": 46, "Te": 52, "I": 53, "Ta": 73, "W": 74, "Pt": 78,
}

ELECTRONEGATIVITY: Dict[str, float] = {
    "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44, "F": 3.98,
    "P": 2.19, "S": 2.58, "Cl": 3.16, "Ti": 1.54, "V": 1.63,
    "Cr": 1.66, "Mn": 1.55, "Fe": 1.83, "Co": 1.88, "Ni": 1.91,
    "Cu": 1.90, "Se": 2.55, "Nb": 1.60, "Mo": 2.16, "Pd": 2.20,
    "Te": 2.10, "I": 2.66, "Ta": 1.50, "W": 2.36, "Pt": 2.28,
}

COVALENT_RADIUS: Dict[str, float] = {
    "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Ti": 1.60, "V": 1.53,
    "Cr": 1.39, "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24,
    "Cu": 1.32, "Se": 1.20, "Nb": 1.64, "Mo": 1.54, "Pd": 1.39,
    "Te": 1.38, "I": 1.39, "Ta": 1.70, "W": 1.62, "Pt": 1.36,
}

VALENCE: Dict[str, int] = {
    "B": 3, "C": 4, "N": 5, "O": 6, "F": 7, "P": 5, "S": 6,
    "Cl": 7, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8,
    "Co": 9, "Ni": 10, "Cu": 11, "Se": 6, "Nb": 5, "Mo": 6,
    "Pd": 10, "Te": 6, "I": 7, "Ta": 5, "W": 6, "Pt": 10,
}

METAL_HER_CENTER: Dict[str, float] = {
    "Pt": 0.02, "Pd": 0.06, "Mo": 0.08, "W": 0.11, "V": -0.03,
    "Nb": -0.02, "Ta": 0.03, "Ni": 0.18, "Co": 0.20, "Fe": 0.24,
    "Cr": -0.11, "Ti": -0.18, "Cu": 0.28, "Mn": -0.16,
}

ANION_HER_SHIFT: Dict[str, float] = {
    "S": 0.00, "Se": 0.04, "Te": 0.08, "N": -0.04, "C": -0.05,
    "P": 0.03, "B": 0.05, "O": -0.10, "F": 0.16, "Cl": 0.12,
    "I": 0.18,
}


def _clip01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-value)))


def element_feature(element: str) -> np.ndarray:
    """Return a compact normalized atom feature vector."""
    return np.array(
        [
            ATOMIC_NUMBER.get(element, 30) / 80.0,
            ELECTRONEGATIVITY.get(element, 2.0) / 4.0,
            COVALENT_RADIUS.get(element, 1.2) / 2.0,
            VALENCE.get(element, 6) / 11.0,
            1.0 if element in METALS else 0.0,
        ],
        dtype=float,
    )


def formula_from_elements(elements: Sequence[str]) -> str:
    counts: Dict[str, int] = {}
    for element in elements:
        counts[element] = counts.get(element, 0) + 1
    parts = []
    for element in sorted(counts, key=lambda x: (x not in METALS, x)):
        count = counts[element]
        parts.append(element if count == 1 else f"{element}{count}")
    return "".join(parts)


def lattice_lengths(lattice: Sequence[Sequence[float]]) -> Tuple[float, float, float]:
    arr = np.asarray(lattice, dtype=float)
    return tuple(float(np.linalg.norm(arr[i])) for i in range(3))


def pairwise_distances(positions: Sequence[Sequence[float]]) -> np.ndarray:
    pos = np.asarray(positions, dtype=float)
    if len(pos) == 0:
        return np.zeros((0, 0), dtype=float)
    diff = pos[:, None, :] - pos[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def build_adjacency(positions: Sequence[Sequence[float]], cutoff: float = 3.1) -> np.ndarray:
    dists = pairwise_distances(positions)
    adjacency = ((dists > 1e-8) & (dists < cutoff)).astype(float)
    return adjacency


def _bond_stats(elements: Sequence[str], positions: Sequence[Sequence[float]]) -> Tuple[float, float, float]:
    dists = pairwise_distances(positions)
    bonds: List[float] = []
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            radius_sum = COVALENT_RADIUS.get(elements[i], 1.2) + COVALENT_RADIUS.get(elements[j], 1.2)
            if 0.55 * radius_sum <= dists[i, j] <= 1.55 * radius_sum + 0.55:
                bonds.append(float(dists[i, j]))
    if not bonds:
        upper = dists[np.triu_indices(len(elements), 1)] if len(elements) > 1 else np.array([2.4])
        bonds = [float(np.mean(upper))]
    return float(np.mean(bonds)), float(np.std(bonds)), float(len(bonds) / max(1, len(elements)))


def graph_descriptor(record, max_atoms: int = 6) -> np.ndarray:
    """Encode a material graph into one numeric vector.

    Descriptor layout:
    composition histogram | lattice/thickness/bond stats | prototype one-hot |
    atom features padded to max_atoms | message-passing graph summary.
    """
    elements = list(record.elements)
    positions = np.asarray(record.positions, dtype=float)
    lattice = np.asarray(record.lattice, dtype=float)

    composition = np.zeros(len(ELEMENTS), dtype=float)
    for element in elements:
        if element in ELEMENTS:
            composition[ELEMENTS.index(element)] += 1.0
    composition /= max(1.0, composition.sum())

    a, b, c = lattice_lengths(lattice)
    z_values = positions[:, 2] if len(positions) else np.array([0.0])
    thickness = float(np.max(z_values) - np.min(z_values))
    bond_mean, bond_std, mean_degree = _bond_stats(elements, positions)
    geometry = np.array(
        [a / 6.0, b / 6.0, c / 25.0, thickness / 5.0, bond_mean / 4.0, bond_std / 2.0, mean_degree / 4.0],
        dtype=float,
    )

    proto = np.zeros(len(PROTOTYPES), dtype=float)
    proto_name = record.prototype if record.prototype in PROTOTYPES else "Other"
    proto[PROTOTYPES.index(proto_name)] = 1.0

    atom_features = np.zeros((max_atoms, 8), dtype=float)
    for idx, element in enumerate(elements[:max_atoms]):
        xy = positions[idx, :2] / 6.0
        z = np.array([positions[idx, 2] / 5.0])
        atom_features[idx, :] = np.concatenate([element_feature(element), xy, z])

    adjacency = build_adjacency(positions)
    raw_features = np.array([element_feature(e) for e in elements], dtype=float)
    if len(elements) > 0:
        degree = adjacency.sum(axis=1, keepdims=True)
        neighbor_mean = (adjacency @ raw_features) / np.maximum(degree, 1.0)
        graph_summary = np.concatenate(
            [
                raw_features.mean(axis=0),
                raw_features.std(axis=0),
                neighbor_mean.mean(axis=0),
                np.array([float(adjacency.sum() / max(1, len(elements))), float(np.std(degree))]),
            ]
        )
    else:
        graph_summary = np.zeros(17, dtype=float)

    return np.concatenate([composition, geometry, proto, atom_features.reshape(-1), graph_summary])


def descriptor_dim(max_atoms: int = 6) -> int:
    return len(ELEMENTS) + 7 + len(PROTOTYPES) + max_atoms * 8 + 17


def dominant_species(elements: Sequence[str]) -> Tuple[str, List[str]]:
    metal_counts: Dict[str, int] = {}
    anion_counts: Dict[str, int] = {}
    for element in elements:
        if element in METALS:
            metal_counts[element] = metal_counts.get(element, 0) + 1
        else:
            anion_counts[element] = anion_counts.get(element, 0) + 1
    metal = max(metal_counts, key=metal_counts.get) if metal_counts else "Mo"
    anions = sorted(anion_counts, key=anion_counts.get, reverse=True) or ["S"]
    return metal, anions


def estimate_delta_g_h(record) -> float:
    """Estimate HER hydrogen adsorption free energy in eV.

    The target for HER is close to 0 eV: too negative binds H too strongly,
    too positive binds H too weakly.
    """
    metal, anions = dominant_species(record.elements)
    base = METAL_HER_CENTER.get(metal, 0.22)
    anion_shift = float(np.mean([ANION_HER_SHIFT.get(a, 0.08) for a in anions]))
    bond_mean, bond_std, _ = _bond_stats(record.elements, record.positions)
    ideal = COVALENT_RADIUS.get(metal, 1.5) + np.mean([COVALENT_RADIUS.get(a, 1.1) for a in anions])
    strain = abs(bond_mean - ideal) / max(ideal, 1e-6)
    janus_bonus = -0.04 if record.prototype == "Janus-MXY" and len(set(anions)) > 1 else 0.0
    mxene_bonus = -0.03 if record.prototype == "MXene" and ("C" in anions or "N" in anions) else 0.0
    disorder_penalty = min(0.12, bond_std * 0.06)
    return float(base + anion_shift + 0.22 * strain + janus_bonus + mxene_bonus + disorder_penalty)


def thermodynamic_stability(record) -> float:
    metal, anions = dominant_species(record.elements)
    known_2d_family = 0.0
    if record.prototype in {"MX2", "Janus-MXY"} and metal in {"Mo", "W", "Nb", "Ta", "V", "Ti"}:
        known_2d_family += 0.24
    if record.prototype == "MXene" and metal in {"Ti", "V", "Nb", "Ta", "Mo", "W"}:
        known_2d_family += 0.28
    if record.prototype == "Elemental" and set(record.elements) <= {"C", "B", "N"}:
        known_2d_family += 0.22

    en_gap = np.mean([abs(ELECTRONEGATIVITY.get(a, 2.3) - ELECTRONEGATIVITY.get(metal, 1.8)) for a in anions])
    bond_mean, bond_std, mean_degree = _bond_stats(record.elements, record.positions)
    thickness = float(np.ptp(np.asarray(record.positions)[:, 2])) if record.positions else 0.0
    compactness = math.exp(-abs(thickness - 3.0) / 4.0)
    bond_order = math.exp(-bond_std)
    score = 0.35 + known_2d_family + 0.10 * min(en_gap, 2.0) + 0.12 * compactness + 0.12 * bond_order
    score += 0.04 * min(mean_degree, 3.0)
    if any(a in {"F", "Cl", "I"} for a in anions):
        score -= 0.08
    if bond_mean < 1.2 or bond_mean > 3.4:
        score -= 0.18
    return _clip01(score)


def kinetic_stability(record) -> float:
    positions = np.asarray(record.positions, dtype=float)
    thickness = float(np.ptp(positions[:, 2])) if len(positions) else 0.0
    _, bond_std, mean_degree = _bond_stats(record.elements, record.positions)
    coordination = math.exp(-abs(mean_degree - 2.7) / 2.2)
    vibration_proxy = math.exp(-bond_std * 1.4)
    layer_proxy = math.exp(-abs(thickness - 2.8) / 3.5)
    score = 0.18 + 0.35 * coordination + 0.30 * vibration_proxy + 0.17 * layer_proxy
    return _clip01(score)


def synthesis_score(record) -> float:
    metal, anions = dominant_species(record.elements)
    family_bonus = {
        "MX2": 0.34,
        "Janus-MXY": 0.25,
        "MXene": 0.30,
        "Binary": 0.22,
        "Elemental": 0.28,
        "Other": 0.10,
    }.get(record.prototype, 0.10)
    known_elements = 0.18 if metal in {"Mo", "W", "Ti", "V", "Nb", "Ta", "Ni", "Pt", "Pd"} else 0.08
    anion_bonus = float(np.mean([0.16 if a in {"S", "Se", "C", "N", "B"} else 0.08 for a in anions]))
    rarity_penalty = 0.05 if metal in {"Pt", "Pd", "Ta"} else 0.0
    volatile_penalty = 0.06 if any(a in {"Te", "I", "F", "Cl"} for a in anions) else 0.0
    score = 0.22 + family_bonus + known_elements + anion_bonus - rarity_penalty - volatile_penalty
    return _clip01(score)


def evaluate_material(record) -> Dict[str, float]:
    delta_g = estimate_delta_g_h(record)
    thermo = thermodynamic_stability(record)
    kinetic = kinetic_stability(record)
    synth = synthesis_score(record)
    stability = 0.55 * thermo + 0.45 * kinetic
    quality = 0.46 * math.exp(-abs(delta_g) / 0.16) + 0.34 * stability + 0.20 * synth
    return {
        "delta_g_h": float(delta_g),
        "thermodynamic_stability": float(thermo),
        "kinetic_stability": float(kinetic),
        "stability_score": float(stability),
        "synthesis_score": float(synth),
        "quality_score": float(quality),
    }


def update_record_properties(record):
    record.properties = evaluate_material(record)
    record.formula = formula_from_elements(record.elements)
    return record


def cart_to_fractional(positions: Sequence[Sequence[float]], lattice: Sequence[Sequence[float]]) -> np.ndarray:
    lat = np.asarray(lattice, dtype=float)
    pos = np.asarray(positions, dtype=float)
    return pos @ np.linalg.inv(lat)


def write_xyz(record, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{len(record.elements)}\n")
        handle.write(
            f"{record.name} formula={record.formula} dgH={record.properties.get('delta_g_h', 0):.4f} "
            f"stability={record.properties.get('stability_score', 0):.4f} synthesis={record.properties.get('synthesis_score', 0):.4f}\n"
        )
        for element, (x, y, z) in zip(record.elements, record.positions):
            handle.write(f"{element:2s} {x: .6f} {y: .6f} {z: .6f}\n")


def write_cif(record, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    a, b, c = lattice_lengths(record.lattice)
    frac = cart_to_fractional(record.positions, record.lattice)
    gamma = 120.0 if record.prototype in {"MX2", "Janus-MXY", "Elemental"} else 90.0
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"data_{record.name}\n")
        handle.write("_symmetry_space_group_name_H-M 'P 1'\n")
        handle.write("_symmetry_Int_Tables_number 1\n")
        handle.write(f"_cell_length_a {a:.6f}\n")
        handle.write(f"_cell_length_b {b:.6f}\n")
        handle.write(f"_cell_length_c {c:.6f}\n")
        handle.write("_cell_angle_alpha 90.000000\n")
        handle.write("_cell_angle_beta 90.000000\n")
        handle.write(f"_cell_angle_gamma {gamma:.6f}\n")
        handle.write("loop_\n")
        handle.write("_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n")
        for idx, (element, coords) in enumerate(zip(record.elements, frac), start=1):
            fx, fy, fz = coords % 1.0
            handle.write(f"{element}{idx} {element} {fx:.6f} {fy:.6f} {fz:.6f}\n")


def records_to_json(records: Iterable, path: Path) -> None:
    serializable = []
    for record in records:
        item = asdict(record)
        item["positions"] = np.asarray(record.positions, dtype=float).tolist()
        item["lattice"] = np.asarray(record.lattice, dtype=float).tolist()
        serializable.append(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
