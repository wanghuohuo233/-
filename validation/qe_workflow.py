"""Generate Quantum ESPRESSO inputs for DFT validation.

This script writes standard input decks for:
- structure relaxation
- SCF energy
- H adsorption geometry for HER ΔG_H screening
- Gamma-point phonon check
- short AIMD stability check

It can optionally run `pw.x` / `ph.x` if they are available on PATH. The current
machine does not need QE installed for the repository tests; generated inputs
are still valid handoff artifacts for a compute workstation or cluster.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_PSEUDO_DIR = "./pseudos"
PSEUDO_FILES = {
    "H": "H.pbe-rrkjus_psl.1.0.0.UPF",
    "B": "B.pbe-n-rrkjus_psl.1.0.0.UPF",
    "C": "C.pbe-n-rrkjus_psl.1.0.0.UPF",
    "N": "N.pbe-n-rrkjus_psl.1.0.0.UPF",
    "O": "O.pbe-n-rrkjus_psl.1.0.0.UPF",
    "S": "S.pbe-n-rrkjus_psl.1.0.0.UPF",
    "Se": "Se.pbe-n-rrkjus_psl.1.0.0.UPF",
    "V": "V.pbe-spnl-rrkjus_psl.1.0.0.UPF",
    "Nb": "Nb.pbe-spn-rrkjus_psl.1.0.0.UPF",
    "Ta": "Ta.pbe-spn-rrkjus_psl.1.0.0.UPF",
    "Mo": "Mo.pbe-spn-rrkjus_psl.1.0.0.UPF",
    "W": "W.pbe-spn-rrkjus_psl.1.0.0.UPF",
}


def _atomic_species(elements: Iterable[str], pseudo_dir: str = DEFAULT_PSEUDO_DIR) -> str:
    unique = sorted(set(elements))
    lines = []
    masses = {
        "H": 1.008, "B": 10.81, "C": 12.011, "N": 14.007, "O": 15.999,
        "S": 32.06, "Se": 78.971, "Te": 127.60, "P": 30.974, "V": 50.942,
        "Nb": 92.906, "Ta": 180.948, "Mo": 95.95, "W": 183.84, "Ti": 47.867,
        "Cr": 51.996, "Mn": 54.938, "Fe": 55.845, "Co": 58.933, "Ni": 58.693,
        "Cu": 63.546, "Pd": 106.42, "Pt": 195.084,
    }
    for element in unique:
        pseudo_file = PSEUDO_FILES.get(element, f"{element}.UPF")
        lines.append(f"  {element:2s} {masses.get(element, 50.0):10.4f} {pseudo_file}")
    return "\n".join(lines)


def _cell_parameters(lattice: List[List[float]]) -> str:
    return "\n".join("  " + " ".join(f"{v:14.8f}" for v in row) for row in lattice)


def _atomic_positions(elements: List[str], positions: List[List[float]]) -> str:
    return "\n".join(
        f"  {element:2s} {x:14.8f} {y:14.8f} {z:14.8f}"
        for element, (x, y, z) in zip(elements, positions)
    )


def make_pw_input(
    record: Dict,
    calculation: str,
    prefix: str,
    pseudo_dir: str = DEFAULT_PSEUDO_DIR,
    k_grid: str = "4 4 1 0 0 0",
    ecutwfc: int = 35,
    ecutrho: int = 280,
) -> str:
    elements = record["elements"]
    positions = record["positions"]
    lattice = record["lattice"]
    return f"""&CONTROL
  calculation = '{calculation}',
  prefix = '{prefix}',
  pseudo_dir = '{pseudo_dir}',
  outdir = './tmp',
  tstress = .true.,
  tprnfor = .true.,
/
&SYSTEM
  ibrav = 0,
  nat = {len(elements)},
  ntyp = {len(set(elements))},
  ecutwfc = {ecutwfc},
  ecutrho = {ecutrho},
  occupations = 'smearing',
  smearing = 'mp',
  degauss = 0.02,
  assume_isolated = '2D',
/
&ELECTRONS
  conv_thr = 1.0d-7,
  mixing_beta = 0.35,
/
&IONS
  ion_dynamics = 'bfgs',
/
&CELL
  cell_dofree = '2Dxy',
/
ATOMIC_SPECIES
{_atomic_species(elements, pseudo_dir)}
CELL_PARAMETERS angstrom
{_cell_parameters(lattice)}
ATOMIC_POSITIONS angstrom
{_atomic_positions(elements, positions)}
K_POINTS automatic
  {k_grid}
"""


def make_h_adsorbed_record(record: Dict, height: float = 1.10) -> Dict:
    positions = [list(p) for p in record["positions"]]
    elements = list(record["elements"])
    top_idx = max(range(len(positions)), key=lambda i: positions[i][2])
    x, y, z = positions[top_idx]
    elements.append("H")
    positions.append([x, y, z + height])
    new_record = dict(record)
    new_record["elements"] = elements
    new_record["positions"] = positions
    return new_record


def make_ph_input(prefix: str) -> str:
    return f"""&INPUTPH
  prefix = '{prefix}',
  outdir = './tmp',
  tr2_ph = 1.0d-14,
  ldisp = .false.,
  epsil = .false.,
/
0.0 0.0 0.0
"""


def make_aimd_input(record: Dict, prefix: str, pseudo_dir: str = DEFAULT_PSEUDO_DIR) -> str:
    text = make_pw_input(record, calculation="md", prefix=prefix, pseudo_dir=pseudo_dir, k_grid="2 2 1 0 0 0")
    text = text.replace("&IONS\n  ion_dynamics = 'bfgs',\n/", "&IONS\n  ion_dynamics = 'verlet',\n  tempw = 300,\n/")
    text = text.replace("&CONTROL", "&CONTROL\n  nstep = 1000,\n  dt = 20.0,")
    return text


def write_inputs(records: List[Dict], output_dir: Path, pseudo_dir: str, top_k: int = 5) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for idx, record in enumerate(records[:top_k], start=1):
        prefix = f"candidate_{idx:02d}_{record['formula']}"
        folder = output_dir / prefix
        folder.mkdir(parents=True, exist_ok=True)
        pristine_relax = folder / "01_relax.in"
        pristine_scf = folder / "02_scf.in"
        h_relax = folder / "03_h_ads_relax.in"
        phonon = folder / "04_gamma_phonon.in"
        aimd = folder / "05_aimd_300K.in"
        pristine_relax.write_text(make_pw_input(record, "relax", prefix, pseudo_dir), encoding="utf-8")
        pristine_scf.write_text(make_pw_input(record, "scf", prefix, pseudo_dir), encoding="utf-8")
        h_record = make_h_adsorbed_record(record)
        h_relax.write_text(make_pw_input(h_record, "relax", prefix + "_H", pseudo_dir), encoding="utf-8")
        phonon.write_text(make_ph_input(prefix), encoding="utf-8")
        aimd.write_text(make_aimd_input(record, prefix + "_aimd", pseudo_dir), encoding="utf-8")
        written.extend([pristine_relax, pristine_scf, h_relax, phonon, aimd])
    return written


def run_if_available(input_files: Iterable[Path], cwd: Path) -> List[Dict[str, object]]:
    pw = shutil.which("pw.exe") or shutil.which("pw.x") or shutil.which("pw")
    ph = shutil.which("ph.exe") or shutil.which("ph.x") or shutil.which("ph")
    results = []
    for path in input_files:
        exe = ph if "phonon" in path.name else pw
        if exe is None:
            print(f"SKIP {path}: required executable is not on PATH")
            results.append({"input": str(path), "status": "missing_executable"})
            continue
        out = path.with_suffix(".out")
        with path.open("r", encoding="utf-8") as stdin, out.open("w", encoding="utf-8") as stdout:
            proc = subprocess.run([exe], stdin=stdin, stdout=stdout, stderr=subprocess.STDOUT, check=False, cwd=cwd)
        text = out.read_text(encoding="utf-8", errors="ignore")
        converged = "JOB DONE" in text
        energy = None
        for line in text.splitlines():
            if line.strip().startswith("!") and "total energy" in line:
                try:
                    energy = float(line.split("=")[1].split()[0])
                except Exception:
                    energy = None
        status = "done" if converged else "failed"
        print(f"{status.upper()} {path} -> {out}")
        results.append({"input": str(path), "output": str(out), "status": status, "returncode": proc.returncode, "energy_ry": energy})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Quantum ESPRESSO validation inputs")
    parser.add_argument("--materials", default="results/generated_materials.json")
    parser.add_argument("--output-dir", default="validation_inputs/qe")
    parser.add_argument("--pseudo-dir", default=DEFAULT_PSEUDO_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--run", action="store_true", help="Run pw.x/ph.x if available")
    parser.add_argument("--summary", default="results/qe_validation_summary.json")
    args = parser.parse_args()

    records = json.loads(Path(args.materials).read_text(encoding="utf-8"))
    written = write_inputs(records, Path(args.output_dir), args.pseudo_dir, top_k=args.top_k)
    print(f"Wrote {len(written)} QE input files under {args.output_dir}")
    if args.run:
        Path("tmp").mkdir(exist_ok=True)
        results = run_if_available(written, cwd=Path.cwd())
        Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary).write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
