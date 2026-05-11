"""Merge multiple normalized material JSON files for diffusion retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.material_dataset import MaterialRecord, load_material_dataset
from utils.geo_utils import evaluate_material, lattice_lengths, records_to_json


def _signature(record: MaterialRecord, decimals: int = 2) -> Tuple:
    lengths = tuple(round(value, decimals) for value in lattice_lengths(record.lattice))
    positions = np.asarray(record.positions, dtype=float)
    z_span = round(float(np.ptp(positions[:, 2])) if len(positions) else 0.0, decimals)
    reduced_elements = tuple(sorted(record.elements))
    return (record.formula, record.prototype, len(record.elements), reduced_elements, lengths, z_span)


def merge_records(paths: Iterable[Path], dedupe: bool = True) -> list[MaterialRecord]:
    merged: list[MaterialRecord] = []
    seen = set()
    for path in paths:
        records = load_material_dataset(str(path))
        for record in records:
            record.properties.update(evaluate_material(record))
            sig = _signature(record)
            if dedupe and sig in seen:
                continue
            seen.add(sig)
            merged.append(record)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge normalized material datasets into one training JSON")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSON files created by prepare_public_dataset.py")
    parser.add_argument("--output", default="data/expanded_2d_materials.json")
    parser.add_argument("--metadata", default="data/expanded_2d_materials_metadata.json")
    parser.add_argument("--no-dedupe", action="store_true", help="Keep duplicate structures")
    args = parser.parse_args()

    paths = [Path(item) for item in args.inputs]
    records = merge_records(paths, dedupe=not args.no_dedupe)
    output = Path(args.output)
    metadata = Path(args.metadata)
    records_to_json(records, output)

    source_counts: dict[str, int] = {}
    for record in records:
        source_counts[record.source] = source_counts.get(record.source, 0) + 1
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text(
        json.dumps(
            {
                "output": str(output),
                "records": len(records),
                "dedupe": not args.no_dedupe,
                "inputs": [str(path) for path in paths],
                "source_counts": source_counts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Merged {len(records)} records -> {output}")
    print(f"Metadata -> {metadata}")


if __name__ == "__main__":
    main()
