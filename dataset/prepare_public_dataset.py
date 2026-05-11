"""Normalize public crystal datasets into this repository's JSON schema.

Supported practical paths:

1. Generic JSON/JSONL exported from 2DMatPedia, C2DB, NOMAD, Materials Project,
   or a Kaggle-style dump, as long as each item carries elements, lattice and
   Cartesian/fractional coordinates.
2. Materials Project API export when `mp-api` is installed and MP_API_KEY is set.
3. JARVIS-Tools Figshare datasets such as dft_2d, dft_2d_2021, c2db,
   and twod_matpd when `jarvis-tools` is installed.

The command deliberately avoids hard-coded private download links. Many public
materials databases require accepting terms, logging in, or using an API key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import math

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.material_dataset import MaterialRecord
from utils.geo_utils import evaluate_material, formula_from_elements, records_to_json


ELEMENT_KEYS = ("elements", "species", "atom_types", "symbols")
LATTICE_KEYS = ("lattice", "cell", "lattice_matrix")
POSITION_KEYS = ("positions", "cart_coords", "cartesian_positions", "coords")
FRACTIONAL_KEYS = ("frac_coords", "fractional_positions", "scaled_positions")


def _first_key(item: Dict, keys: Iterable[str]):
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def _as_matrix(value, rows: Optional[int] = None) -> List[List[float]]:
    arr = np.asarray(value, dtype=object)
    if arr.ndim == 1 and len(arr) > 0 and isinstance(arr[0], (list, tuple, np.ndarray)):
        arr = np.vstack([np.asarray(row, dtype=float) for row in arr])
    else:
        arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D matrix")
    if rows is not None and arr.shape[0] != rows:
        raise ValueError(f"Expected {rows} rows, got {arr.shape[0]}")
    return arr.tolist()


def _formula_from_mp_composition(composition) -> str:
    if hasattr(composition, "reduced_formula"):
        return composition.reduced_formula
    return str(composition)


def record_from_generic_item(item: Dict, source: str = "public_dataset") -> Optional[MaterialRecord]:
    """Convert one heterogeneous public-dataset item into MaterialRecord."""
    elements = _first_key(item, ELEMENT_KEYS)
    lattice = _first_key(item, LATTICE_KEYS)
    positions = _first_key(item, POSITION_KEYS)
    frac_positions = _first_key(item, FRACTIONAL_KEYS)

    if elements is None and "structure" in item:
        return record_from_generic_item(item["structure"], source=source)
    if elements is None or lattice is None or (positions is None and frac_positions is None):
        return None

    elements = [str(element) for element in elements]
    lattice = _as_matrix(lattice, rows=3)
    if positions is None:
        frac = np.asarray(frac_positions, dtype=float)
        positions = (frac @ np.asarray(lattice, dtype=float)).tolist()
    else:
        positions = _as_matrix(positions)

    name = str(item.get("name", item.get("material_id", item.get("id", "public_material"))))
    prototype = str(item.get("prototype", item.get("structure_type", item.get("spacegroup", "Other"))))
    formula = str(item.get("formula", item.get("pretty_formula", formula_from_elements(elements))))
    properties = dict(item.get("properties", {}))

    record = MaterialRecord(
        name=name,
        formula=formula,
        prototype=prototype,
        elements=elements,
        positions=positions,
        lattice=lattice,
        properties=properties,
        source=source,
    )
    record.properties.update(evaluate_material(record))
    return record


def _numeric_property(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _jarvis_properties(item: Dict) -> Dict[str, object]:
    keys = (
        "energy_above_hull",
        "ehull",
        "formation_energy_peratom",
        "formation_energy_per_atom",
        "optb88vdw_total_energy",
        "optb88vdw_bandgap",
        "mbj_bandgap",
        "spillage",
        "magmom_oszicar",
    )
    return {key: _numeric_property(item[key]) for key in keys if key in item and item[key] not in (None, "")}


def record_from_jarvis_item(item: Dict, source: str = "jarvis") -> Optional[MaterialRecord]:
    """Convert one JARVIS-Tools item into MaterialRecord.

    JARVIS atom dictionaries commonly use lattice_mat + coords + elements.
    Some mirrors/exported JSON files use generic keys, so this function keeps a
    few fallbacks instead of assuming a single versioned schema.
    """
    atoms = item.get("atoms", item.get("structure", {}))
    if not isinstance(atoms, dict):
        return record_from_generic_item(item, source=source)

    elements = _first_key(atoms, ("elements", "species", "atom_types", "symbols"))
    lattice = _first_key(atoms, ("lattice_mat", "lattice", "cell", "lattice_matrix"))
    coords = _first_key(atoms, ("coords", "positions", "cart_coords", "cartesian_positions"))
    frac_coords = _first_key(atoms, ("frac_coords", "fractional_positions", "scaled_positions"))

    if elements is None or lattice is None or (coords is None and frac_coords is None):
        return record_from_generic_item(item, source=source)

    elements = [str(element) for element in elements]
    lattice = _as_matrix(lattice, rows=3)
    if coords is None:
        frac = np.asarray(frac_coords, dtype=float)
        positions = (frac @ np.asarray(lattice, dtype=float)).tolist()
    else:
        coords_array = np.asarray(coords, dtype=float)
        cartesian = bool(atoms.get("cartesian", atoms.get("coords_are_cartesian", True)))
        positions = coords_array.tolist() if cartesian else (coords_array @ np.asarray(lattice, dtype=float)).tolist()

    name = str(item.get("jid", item.get("id", item.get("name", "jarvis_material"))))
    formula = str(item.get("formula", item.get("formula_pretty", formula_from_elements(elements))))
    prototype = str(item.get("prototype", item.get("spacegroup", item.get("spg_number", "JARVIS-2D"))))
    properties = _jarvis_properties(item)

    record = MaterialRecord(
        name=name,
        formula=formula,
        prototype=prototype,
        elements=elements,
        positions=positions,
        lattice=lattice,
        properties=properties,
        source=source,
    )
    record.properties.update(evaluate_material(record))
    return record


def load_jarvis_dataset(dataset_name: str, store_dir: Optional[Path], max_entries: Optional[int] = None) -> List[MaterialRecord]:
    """Download/read a JARVIS-Tools Figshare dataset and normalize it.

    If Figshare is blocked by the local network, manually download the zip from
    the official JARVIS link and place it in `store_dir`; jarvis-tools will read
    the local zip on the next run.
    """
    try:
        from jarvis.db.figshare import data
    except ImportError as exc:
        raise RuntimeError("Install JARVIS first: pip install jarvis-tools") from exc

    try:
        raw_items = data(dataset_name, store_dir=str(store_dir) if store_dir else None)
    except Exception as exc:
        target = str(store_dir or "<jarvis cache>")
        if store_dir and store_dir.exists():
            for cached_zip in store_dir.glob("*.zip"):
                if cached_zip.stat().st_size < 1024:
                    cached_zip.unlink()
        raise RuntimeError(
            "JARVIS download/read failed. If the network returns 403, download the official "
            f"JARVIS Figshare zip in a browser and place it under {target}, then rerun."
        ) from exc

    records: List[MaterialRecord] = []
    for item in raw_items:
        record = record_from_jarvis_item(item, source=f"jarvis:{dataset_name}")
        if record is not None:
            records.append(record)
        if max_entries and len(records) >= max_entries:
            break
    return records


def load_generic_public_file(path: Path, source: str = "public_dataset") -> List[MaterialRecord]:
    records: List[MaterialRecord] = []
    if path.suffix.lower() == ".parquet":
        return load_colabfit_parquet(path, source=source)
    if path.suffix.lower() == ".jsonl":
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            for key in ("materials", "data", "structures", "entries", "items"):
                if key in raw and isinstance(raw[key], list):
                    raw = raw[key]
                    break
        items = raw if isinstance(raw, list) else [raw]

    for item in items:
        record = record_from_generic_item(item, source=source)
        if record is not None:
            records.append(record)
    return records


def _symbols_from_atomic_numbers(values: Iterable[int]) -> List[str]:
    try:
        from ase.data import chemical_symbols
    except ImportError as exc:
        raise RuntimeError("Install ASE first: pip install ase") from exc
    return [chemical_symbols[int(value)] for value in values]


def load_colabfit_parquet(path: Path, source: str = "colabfit_parquet", max_entries: Optional[int] = None) -> List[MaterialRecord]:
    """Load ColabFit Exchange parquet rows.

    This supports the Hugging Face ColabFit parquet schema used by datasets such
    as colabfit/JARVIS_C2DB, where each configuration row contains `cell`,
    `positions`, and per-atom `atomic_numbers`.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Install pandas/pyarrow first: pip install pandas pyarrow") from exc

    df = pd.read_parquet(path)
    records: List[MaterialRecord] = []
    for row in df.itertuples(index=False):
        data = row._asdict()
        atomic_numbers = data.get("atomic_numbers")
        lattice = data.get("cell")
        positions = data.get("positions")
        if atomic_numbers is None or lattice is None or positions is None:
            continue
        elements = _symbols_from_atomic_numbers(atomic_numbers)
        properties = {}
        for key in ("energy", "electronic_band_gap", "formation_energy", "energy_above_hull", "adsorption_energy"):
            value = data.get(key)
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                properties[key] = float(value)
        names = data.get("names")
        name = names[0] if isinstance(names, list) and names else data.get("configuration_id", "colabfit_material")
        record = MaterialRecord(
            name=str(name),
            formula=str(data.get("chemical_formula_reduced") or formula_from_elements(elements)),
            prototype=str(data.get("chemical_formula_anonymous") or "ColabFit"),
            elements=elements,
            positions=_as_matrix(positions),
            lattice=_as_matrix(lattice, rows=3),
            properties=properties,
            source=source,
        )
        record.properties.update(evaluate_material(record))
        records.append(record)
        if max_entries and len(records) >= max_entries:
            break
    return records


def load_ase_database(path: Path, source: str = "ase_public_db", max_entries: Optional[int] = None) -> List[MaterialRecord]:
    try:
        from ase.db import connect
    except ImportError as exc:
        raise RuntimeError("Install ASE first: pip install ase") from exc

    records: List[MaterialRecord] = []
    db = connect(str(path))
    seen = set()
    for row in db.select():
        atoms = row.toatoms()
        kv = dict(row.key_value_pairs)
        xc = str(kv.get("xc", ""))
        key = (str(kv.get("name", row.formula)), str(kv.get("phase", "")), xc or "unknown")
        if key in seen:
            continue
        seen.add(key)
        symbols = atoms.get_chemical_symbols()
        positions = atoms.positions.tolist()
        lattice = atoms.cell.array.tolist()
        properties = {
            "hform": float(kv["hform"]) if "hform" in kv else None,
            "hform_fere": float(kv["hform_fere"]) if "hform_fere" in kv else None,
            "band_gap": float(kv["ind_gap"]) if "ind_gap" in kv else None,
            "direct_gap": float(kv["dir_gap"]) if "dir_gap" in kv else None,
            "xc": xc,
        }
        properties = {k: v for k, v in properties.items() if v is not None}
        record = MaterialRecord(
            name=str(kv.get("name", row.formula)),
            formula=row.formula,
            prototype=str(kv.get("phase", "Other")),
            elements=symbols,
            positions=positions,
            lattice=lattice,
            properties=properties,
            source=source,
        )
        record.properties.update(evaluate_material(record))
        records.append(record)
        if max_entries and len(records) >= max_entries:
            break
    return records


def export_from_materials_project(output: Path, max_entries: int = 500) -> List[MaterialRecord]:
    """Fetch 2D-like entries from Materials Project when credentials exist."""
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("Set MP_API_KEY before using --source materials-project")
    try:
        from mp_api.client import MPRester
    except ImportError as exc:
        raise RuntimeError("Install mp-api first: pip install mp-api") from exc

    records: List[MaterialRecord] = []
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            fields=["material_id", "formula_pretty", "structure", "energy_above_hull"],
            num_chunks=1,
            chunk_size=max_entries,
        )
        for doc in docs:
            structure = doc.structure
            lattice = structure.lattice.matrix.tolist()
            positions = structure.cart_coords.tolist()
            elements = [site.specie.symbol for site in structure.sites]
            c_len = float(np.linalg.norm(np.asarray(lattice)[2]))
            z_span = float(np.ptp(np.asarray(positions)[:, 2])) if positions else 0.0
            if c_len < 12.0 or z_span > 6.0:
                continue
            record = MaterialRecord(
                name=str(doc.material_id),
                formula=_formula_from_mp_composition(doc.formula_pretty),
                prototype="MP-screened-2D-like",
                elements=elements,
                positions=positions,
                lattice=lattice,
                properties={"energy_above_hull": float(doc.energy_above_hull or 0.0)},
                source="Materials Project",
            )
            record.properties.update(evaluate_material(record))
            records.append(record)
    records_to_json(records, output)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare public 2D/crystal data for training")
    parser.add_argument("--input", help="Generic JSON/JSONL export from a public dataset")
    parser.add_argument(
        "--source",
        default="public_dataset",
        choices=["public_dataset", "2dmatpedia", "c2db", "nomad", "materials-project", "jarvis"],
    )
    parser.add_argument("--jarvis-dataset", default="dft_2d", help="JARVIS dataset name, e.g. dft_2d, dft_2d_2021, c2db, twod_matpd")
    parser.add_argument("--store-dir", default="data/jarvis_cache", help="Local cache for JARVIS zip files")
    parser.add_argument("--output", default="data/public_2d_materials.json")
    parser.add_argument("--max-entries", type=int, default=500)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.source == "materials-project":
        records = export_from_materials_project(output, max_entries=args.max_entries)
    elif args.source == "jarvis":
        records = load_jarvis_dataset(args.jarvis_dataset, Path(args.store_dir), max_entries=args.max_entries)
        records_to_json(records, output)
    else:
        if not args.input:
            raise ValueError("--input is required unless --source materials-project")
        input_path = Path(args.input)
        if input_path.suffix.lower() == ".db":
            records = load_ase_database(input_path, source=args.source, max_entries=args.max_entries)
        elif input_path.suffix.lower() == ".parquet":
            records = load_colabfit_parquet(input_path, source=args.source, max_entries=args.max_entries)
        else:
            records = load_generic_public_file(input_path, source=args.source)
        records_to_json(records, output)

    print(f"Prepared {len(records)} records -> {output}")


if __name__ == "__main__":
    main()
