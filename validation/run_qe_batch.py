"""Run a small QE validation batch from generated candidate input folders.

The script is meant to be executed in a Quantum ESPRESSO work directory that
contains candidate folders plus `pseudos/` and `tmp/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
from typing import Iterable


MASS = {
    "H": 1.0080,
    "C": 12.0110,
    "N": 14.0070,
    "O": 15.9990,
    "S": 32.0600,
    "Se": 78.9710,
    "V": 50.9420,
    "Nb": 92.9060,
}

PSEUDO = {
    "H": "H.pbe-rrkjus_psl.1.0.0.UPF",
    "C": "C.pbe-n-rrkjus_psl.1.0.0.UPF",
    "N": "N.pbe-n-rrkjus_psl.1.0.0.UPF",
    "O": "O.pbe-n-rrkjus_psl.1.0.0.UPF",
    "S": "S.pbe-n-rrkjus_psl.1.0.0.UPF",
    "Se": "Se.pbe-n-rrkjus_psl.1.0.0.UPF",
    "V": "V.pbe-spnl-rrkjus_psl.1.0.0.UPF",
    "Nb": "Nb.pbe-spn-rrkjus_psl.1.0.0.UPF",
}


def run(cmd: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(cmd), ">", output, flush=True)
    with output.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=True)


def completed(path: Path) -> bool:
    return path.exists() and "JOB DONE" in path.read_text(errors="ignore")


def write_status(candidate_dir: Path, status: str, reason: str, extra: dict | None = None) -> None:
    payload = {"candidate": candidate_dir.name, "status": status, "reason": reason}
    if extra:
        payload.update(extra)
    (candidate_dir / "qe_screening_status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_card(lines: list[str], name: str) -> list[str]:
    for idx, line in enumerate(lines):
        if line.strip().upper().startswith(name):
            out = []
            for item in lines[idx + 1 :]:
                stripped = item.strip()
                if not stripped:
                    continue
                if re.match(r"^[A-Z_]+", stripped) and not re.match(r"^[A-Z][a-z]?\s+", stripped):
                    break
                out.append(item)
            return out
    raise ValueError(f"Card not found: {name}")


def parse_species(input_path: Path) -> list[str]:
    species = []
    for line in parse_card(input_path.read_text().splitlines(), "ATOMIC_SPECIES"):
        parts = line.split()
        if len(parts) >= 3:
            species.append(parts[0])
    return species


def parse_cell(input_path: Path) -> list[list[float]]:
    rows = []
    for line in parse_card(input_path.read_text().splitlines(), "CELL_PARAMETERS"):
        parts = line.split()
        if len(parts) == 3:
            rows.append([float(x) for x in parts])
        if len(rows) == 3:
            break
    if len(rows) != 3:
        raise ValueError(f"Could not parse 3 cell rows from {input_path}")
    return rows


def relax_converged(output_path: Path) -> tuple[bool, str]:
    text = output_path.read_text(errors="ignore")
    if "JOB DONE" not in text:
        return False, "job_not_done"
    if "The maximum number of steps has been reached" in text:
        return False, "maximum_relax_steps_reached"
    if "convergence NOT achieved" in text:
        return False, "scf_convergence_not_achieved"
    if "Error in routine" in text or "MPI_ABORT" in text:
        return False, "qe_error"
    if "Begin final coordinates" not in text or "End final coordinates" not in text:
        return False, "no_normal_final_coordinates_block"
    return True, "relax_converged"


def parse_coordinate_line(line: str) -> tuple[str, float, float, float] | None:
    parts = line.split()
    if len(parts) != 4:
        return None
    if not re.fullmatch(r"[A-Z][a-z]?", parts[0]):
        return None
    try:
        return parts[0], float(parts[1]), float(parts[2]), float(parts[3])
    except ValueError:
        return None


def parse_atomic_positions_block(block: list[str]) -> list[tuple[str, float, float, float]]:
    begin = None
    positions: list[tuple[str, float, float, float]] = []
    stop_prefixes = (
        "End",
        "JOB DONE",
        "Writing",
        "Begin final coordinates",
        "CELL_PARAMETERS",
        "K_POINTS",
        "&",
        "=",
    )
    for idx, line in enumerate(block):
        if line.strip().startswith("ATOMIC_POSITIONS"):
            begin = idx
            break
    if begin is None:
        return positions
    for line in block[begin + 1 :]:
        stripped = line.strip()
        if not stripped:
            break
        if any(stripped.startswith(prefix) for prefix in stop_prefixes):
            break
        parsed = parse_coordinate_line(stripped)
        if parsed is None:
            break
        positions.append(parsed)
    return positions


def parse_positions_from_relax(output_path: Path) -> list[tuple[str, float, float, float]]:
    lines = output_path.read_text(errors="ignore").splitlines()
    begin = None
    end = None
    for idx, line in enumerate(lines):
        if "Begin final coordinates" in line:
            begin = idx
        if begin is not None and "End final coordinates" in line:
            end = idx
            break
    if begin is None or end is None:
        raise ValueError(f"No normal final coordinates block in {output_path}")
    positions = parse_atomic_positions_block(lines[begin:end])
    if not positions:
        raise ValueError(f"No positions parsed from {output_path}")
    return positions


def species_block(species: Iterable[str]) -> str:
    lines = ["ATOMIC_SPECIES"]
    for element in species:
        if element not in PSEUDO:
            raise ValueError(f"No pseudo mapping for {element}")
        lines.append(f"  {element:<2s} {MASS.get(element, 50.0):10.4f} {PSEUDO[element]}")
    return "\n".join(lines)


def cell_block(cell: list[list[float]]) -> str:
    rows = ["CELL_PARAMETERS angstrom"]
    rows.extend(f"  {a:14.8f} {b:14.8f} {c:14.8f}" for a, b, c in cell)
    return "\n".join(rows)


def position_block(positions: list[tuple[str, float, float, float]]) -> str:
    rows = ["ATOMIC_POSITIONS angstrom"]
    rows.extend(f"  {el:<2s} {x:16.10f} {y:16.10f} {z:16.10f}" for el, x, y, z in positions)
    return "\n".join(rows)


def write_pw_input(
    path: Path,
    calculation: str,
    prefix: str,
    positions,
    cell,
    species,
    relax: bool = False,
    stable_electrons: bool = False,
) -> None:
    ions = "\n&IONS\n  ion_dynamics = 'bfgs',\n/" if relax else ""
    ecutrho = 420 if stable_electrons else 280
    smearing = "mv" if stable_electrons else "mp"
    degauss = 0.01 if stable_electrons else 0.02
    conv_thr = "1.0d-6" if stable_electrons else "1.0d-7"
    electron_extra = (
        "  electron_maxstep = 300,\n"
        "  mixing_beta = 0.10,\n"
        "  mixing_mode = 'local-TF',\n"
        "  diagonalization = 'david',"
        if stable_electrons
        else "  mixing_beta = 0.35,"
    )
    text = f"""&CONTROL
  calculation = '{calculation}',
  prefix = '{prefix}',
  pseudo_dir = './pseudos',
  outdir = './tmp',
  tstress = .true.,
  tprnfor = .true.,
/
&SYSTEM
  ibrav = 0,
  nat = {len(positions)},
  ntyp = {len(species)},
  ecutwfc = 35,
  ecutrho = {ecutrho},
  occupations = 'smearing',
  smearing = '{smearing}',
  degauss = {degauss},
  assume_isolated = '2D',
/
&ELECTRONS
  conv_thr = {conv_thr},
{electron_extra}
/{ions}
{species_block(species)}
{cell_block(cell)}
{position_block(positions)}
K_POINTS automatic
  4 4 1 0 0 0
"""
    path.write_text(text, encoding="utf-8")


def write_ph_input(path: Path, prefix: str) -> None:
    path.write_text(
        f"""&INPUTPH
  prefix = '{prefix}',
  outdir = './tmp',
  tr2_ph = 1.0d-12,
  ldisp = .false.,
  epsil = .false.,
/
0.0 0.0 0.0
""",
        encoding="utf-8",
    )


def make_h_ads_positions(positions: list[tuple[str, float, float, float]]) -> list[tuple[str, float, float, float]]:
    top = max(positions, key=lambda row: row[3])
    _, x, y, z = top
    return list(positions) + [("H", x, y, z + 1.1)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QE relax/scf/H-ads/Gamma phonon for generated candidates")
    parser.add_argument("--root", default=".")
    parser.add_argument("--candidates", nargs="*", default=None)
    parser.add_argument("--pw", default="/root/bin/micromamba run -n qe pw.x")
    parser.add_argument("--ph", default="/root/bin/micromamba run -n qe ph.x")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--screening-no-scf",
        action="store_true",
        help="Use the final relax energy/save for screening, then run H adsorption and phonon without a separate SCF.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    candidates = [root / item for item in args.candidates] if args.candidates else sorted(root.glob("candidate_*"))
    pw = args.pw.split()
    ph = args.ph.split()

    for cand in candidates:
        if not cand.is_dir():
            continue
        name = cand.name
        print(f"=== {name} ===", flush=True)
        relax_out = cand / "01_relax_cf.out"
        if not (args.skip_existing and completed(relax_out)):
            run(pw + ["-in", str(cand / "01_relax.in")], relax_out)

        ok, reason = relax_converged(relax_out)
        if not ok:
            write_status(cand, "relax_not_converged", reason, {"relax_output": str(relax_out.name)})
            print(f"SKIP {name}: relax_not_converged ({reason})", flush=True)
            continue

        positions = parse_positions_from_relax(relax_out)
        cell = parse_cell(cand / "01_relax.in")
        species = parse_species(cand / "01_relax.in")

        if args.screening_no_scf:
            scf_prefix = name
        else:
            scf_prefix = f"{name}_relaxed"
            scf_in = cand / "02_scf_relaxed.in"
            write_pw_input(scf_in, "scf", scf_prefix, positions, cell, species, relax=False)
            scf_out = cand / "02_scf_relaxed_cf.out"
            if not (args.skip_existing and completed(scf_out)):
                run(pw + ["-in", str(scf_in)], scf_out)

        h_positions = make_h_ads_positions(positions)
        h_species = list(species)
        if "H" not in h_species:
            h_species = ["H"] + h_species
        h_prefix = f"{name}_H_relaxed"
        h_in = cand / "03_h_ads_top_relaxed.in"
        write_pw_input(h_in, "relax", h_prefix, h_positions, cell, h_species, relax=True, stable_electrons=True)
        h_out = cand / "03_h_ads_top_relaxed_cf.out"
        if not (args.skip_existing and completed(h_out)):
            run(pw + ["-in", str(h_in)], h_out)

        ph_in = cand / "04_gamma_phonon_relaxed.in"
        write_ph_input(ph_in, scf_prefix)
        ph_out = cand / "04_gamma_phonon_relaxed_cf.out"
        if not (args.skip_existing and completed(ph_out)):
            run(ph + ["-in", str(ph_in)], ph_out)
        write_status(cand, "screening_completed", "relax_h_adsorption_and_phonon_finished")


if __name__ == "__main__":
    main()
