#!/usr/bin/env python3
"""
SugarCubed Tensor Compilation Script (scripts/build_tensor.py)

Reads sugarcubed-catalog.jsonld, joins GLYCAM 3D structures (.ttl) with
BMRB and GISSMO NMR-STAR (.str) records using Standard InChI (molecular anchor)
and ALATIS (canonical 1-based atom index), and writes AI-ready tensor files into
the git-ignored `training-data/` directory (plus updates `sbom.json` with SHA-256
provenance hashes).

Usage:
    python3 scripts/build_tensor.py
    python3 scripts/build_tensor.py --output-dir training-data
    python3 scripts/build_tensor.py --verify-only
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
import sys
from pathlib import Path

ATOMIC_NUMBERS = {"H": 1, "C": 6, "N": 7, "O": 8, "P": 15, "S": 16}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_alatis_index(atom_id: str) -> int:
    m = re.search(r"(\d+)$", atom_id.strip())
    if not m:
        raise ValueError(f"Cannot extract ALATIS integer index from {atom_id!r}")
    return int(m.group(1))


def parse_glycam_ttl(ttl_path: Path) -> dict:
    text = ttl_path.read_text(encoding="utf-8")
    oligo_block = text.split("\n:model_oligo1\n")[1].split("rdf:type\t\t:Oligosaccharide.")[0]
    inchi = re.search(r':InChICode\s+"""(InChI=1S/.*?)"""', oligo_block).group(1)
    seq_name = re.search(r':oligoSequenceName\s+"""(.*?)"""', oligo_block).group(1)
    smiles = re.search(r':canonicalSMILES\s+"""(.*?)"""', oligo_block).group(1)
    is_frag_match = re.search(r":isFragment\s+(true|false)", oligo_block)
    is_fragment = is_frag_match.group(1) == "true" if is_frag_match else False
    alatis_map = json.loads(
        re.search(r':hasALATISMapping\s+"""(\{.*?\})"""', oligo_block).group(1)
    )

    # Parse Monosaccharide metadata keyed by residue name (e.g., 1GA, 0CU, 0GB, 4GB)
    mono_metadata = {}
    for mono_match in re.finditer(
        r":model_mono\d+\s+(.*?)rdf:type\s+:Monosaccharide\.", text, re.S
    ):
        mblock = mono_match.group(1)
        res_name_m = re.search(r':hasResidueName\s+"""(.*?)"""', mblock)
        if res_name_m:
            rname = res_name_m.group(1)
            mono_metadata[rname] = {
                "residue_name": rname,
                "short_name": (re.search(r':shortName\s+"""(.*?)"""', mblock) or [None, ""])[1],
                "isomer": (re.search(r':isomer\s+"""(.*?)"""', mblock) or [None, ""])[1],
                "configuration": (re.search(r':configuration\s+"""(.*?)"""', mblock) or [None, ""])[1],
                "ring_type": (re.search(r':ringType\s+"""(.*?)"""', mblock) or [None, ""])[1],
                "snfg_name": (re.search(r':hasSNFGName\s+"""(.*?)"""', mblock) or [None, ""])[1],
            }

    mol2_block = text.split('gmmo:Mol2file """')[1].split('""".')[0]
    in_atom, in_bond = False, False
    mol2_idx_to_alatis = {}
    atoms_by_alatis = {}
    bonds_alatis = []

    for line in mol2_block.splitlines():
        if line.startswith("@<TRIPOS>ATOM"):
            in_atom, in_bond = True, False
            continue
        if line.startswith("@<TRIPOS>BOND"):
            in_atom, in_bond = False, True
            continue
        if line.startswith("@<TRIPOS>SUBSTRUCTURE"):
            in_atom, in_bond = False, False
            continue

        if in_atom and line.strip():
            parts = line.split()
            mol2_idx = int(parts[0])
            atom_name, x, y, z, amber_type, res_id, res_name, charge = (
                parts[1],
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                parts[5],
                parts[6],
                parts[7],
                float(parts[8]),
            )
            glycam_key = f"{res_id}_{res_name}_{atom_name}"
            alatis_label = alatis_map[glycam_key]
            alatis_idx = parse_alatis_index(alatis_label)
            elem = re.match(r"([A-Za-z]+)", alatis_label).group(1)
            mol2_idx_to_alatis[mol2_idx] = alatis_idx
            res_info = mono_metadata.get(res_name, {"residue_name": res_name})
            atoms_by_alatis[alatis_idx] = {
                "alatis_index": alatis_idx,
                "tensor_row": alatis_idx - 1,
                "alatis_label": alatis_label,
                "element": elem,
                "atomic_number": ATOMIC_NUMBERS.get(elem, 0),
                "glycam_atom_id": glycam_key,
                "glycam_residue_index": int(res_id),
                "glycam_residue_name": res_name,
                "amber_atom_type": amber_type,
                "partial_charge": charge,
                "xyz": [x, y, z],
                "ontology_residue_features": res_info,
            }
        elif in_bond and line.strip():
            parts = line.split()
            u = mol2_idx_to_alatis[int(parts[1])] - 1
            v = mol2_idx_to_alatis[int(parts[2])] - 1
            bonds_alatis.append([u, v])
            bonds_alatis.append([v, u])  # Bidirectional directed edges for GNN message passing

    num_atoms = len(atoms_by_alatis)
    ordered_atoms = [atoms_by_alatis[i] for i in range(1, num_atoms + 1)]

    # Node feature matrix X: [atomic_number, partial_charge, glycam_residue_index, is_ring_pyranose, is_ring_furanose, is_alpha, is_beta]
    node_features_X = []
    for a in ordered_atoms:
        rf = a["ontology_residue_features"]
        node_features_X.append([
            float(a["atomic_number"]),
            float(a["partial_charge"]),
            float(a["glycam_residue_index"]),
            1.0 if rf.get("ring_type") == "P" else 0.0,
            1.0 if rf.get("ring_type") == "F" else 0.0,
            1.0 if rf.get("configuration") == "alpha" else 0.0,
            1.0 if rf.get("configuration") == "beta" else 0.0,
        ])

    return {
        "inchi": inchi,
        "glycam_sequence": seq_name,
        "canonical_smiles": smiles,
        "is_fragment": is_fragment,
        "source_ttl": ttl_path.name,
        "num_atoms": num_atoms,
        "atoms": ordered_atoms,
        "node_feature_names": [
            "atomic_number",
            "partial_charge",
            "glycam_residue_index",
            "is_pyranose",
            "is_furanose",
            "is_alpha",
            "is_beta",
        ],
        "X_node_features": node_features_X,
        "R_coordinates_xyz": [a["xyz"] for a in ordered_atoms],
        "E_edge_index": bonds_alatis,
        "nmr_sources": [],
        "bmrb_experimental_shifts": {},
        "gissmo_spin_matrix": {"shifts_ppm": {}, "j_couplings_hz": []},
    }


def parse_nmr_star_loop(text: str, prefix: str) -> list[dict]:
    pattern = re.compile(r"loop_\s+((?:\s+" + re.escape(prefix) + r"\.\S+\s*\n)+)(.*?)stop_", re.S)
    m = pattern.search(text)
    if not m:
        return []
    cols = [c.strip() for c in m.group(1).splitlines() if c.strip()]
    rows = []
    for line in m.group(2).splitlines():
        parts = line.split()
        if len(parts) == len(cols):
            rows.append(dict(zip(cols, parts)))
    return rows


def attach_nmr_str_data(records_by_inchi: dict, str_path: Path, partition_id: str) -> None:
    text = str_path.read_text(encoding="utf-8")
    inchi_match = re.search(r"_Chem_comp\.InChI_code\s+(InChI=1S/\S+)", text)
    if not inchi_match or inchi_match.group(1) not in records_by_inchi:
        return
    rec = records_by_inchi[inchi_match.group(1)]
    rec["nmr_sources"].append(str_path.name)

    if "bmrb" in partition_id:
        for row in parse_nmr_star_loop(text, "_Atom_chem_shift"):
            alatis_idx = parse_alatis_index(row["_Atom_chem_shift.Atom_ID"])
            rec["bmrb_experimental_shifts"].setdefault(str(alatis_idx), []).append({
                "alatis_index": alatis_idx,
                "tensor_row": alatis_idx - 1,
                "atom_type": row.get("_Atom_chem_shift.Atom_type", ""),
                "isotope": int(row.get("_Atom_chem_shift.Atom_isotope_number", "0")),
                "val_ppm": float(row["_Atom_chem_shift.Val"]),
                "ambiguity_code": int(row.get("_Atom_chem_shift.Ambiguity_code", "1")),
            })
    elif "gissmo" in partition_id:
        for row in parse_nmr_star_loop(text, "_Atom_chem_shift"):
            alatis_idx = parse_alatis_index(row["_Atom_chem_shift.Atom_ID"])
            rec["gissmo_spin_matrix"]["shifts_ppm"][str(alatis_idx)] = {
                "alatis_index": alatis_idx,
                "tensor_row": alatis_idx - 1,
                "atom_type": row.get("_Atom_chem_shift.Atom_type", "H"),
                "val_ppm": float(row["_Atom_chem_shift.Val"]),
            }
        for row in parse_nmr_star_loop(text, "_Coupling_constant"):
            u = parse_alatis_index(row["_Coupling_constant.Atom_ID_1"])
            v = parse_alatis_index(row["_Coupling_constant.Atom_ID_2"])
            j_hz = float(row["_Coupling_constant.Val"])
            rec["gissmo_spin_matrix"]["j_couplings_hz"].append({
                "alatis_index_1": u,
                "alatis_index_2": v,
                "tensor_row_1": u - 1,
                "tensor_row_2": v - 1,
                "j_hz": j_hz,
            })


def build_dense_observables(rec: dict) -> dict:
    """Builds fixed-length N_atoms vectors and N_atoms x N_atoms J-coupling matrix."""
    n = rec["num_atoms"]
    y_shift_ppm = [None] * n
    y_shift_mask = [0] * n
    y_shift_ambiguity = [0] * n
    y_shift_candidates = [[] for _ in range(n)]

    # Populate from BMRB experimental shifts where available
    for idx_str, entries in rec["bmrb_experimental_shifts"].items():
        row = int(idx_str) - 1
        vals = [e["val_ppm"] for e in entries]
        amb = entries[0]["ambiguity_code"]
        y_shift_candidates[row] = vals
        y_shift_ambiguity[row] = amb
        y_shift_mask[row] = 1
        # Primary scalar target is the mean of unambiguous (code 1) or reported candidates
        y_shift_ppm[row] = round(sum(vals) / len(vals), 5)

    # Populate from GISSMO spin-matrix shifts where available
    for idx_str, entry in rec["gissmo_spin_matrix"]["shifts_ppm"].items():
        row = int(idx_str) - 1
        y_shift_ppm[row] = entry["val_ppm"]
        y_shift_candidates[row] = [entry["val_ppm"]]
        y_shift_ambiguity[row] = 1
        y_shift_mask[row] = 1

    # N x N symmetric J-coupling matrix (Hz)
    j_matrix = [[0.0] * n for _ in range(n)]
    j_mask = [[0] * n for _ in range(n)]
    for jc in rec["gissmo_spin_matrix"]["j_couplings_hz"]:
        r1, r2, val = jc["tensor_row_1"], jc["tensor_row_2"], jc["j_hz"]
        j_matrix[r1][r2] = val
        j_matrix[r2][r1] = val
        j_mask[r1][r2] = 1
        j_mask[r2][r1] = 1

    return {
        "Y_shift_ppm": y_shift_ppm,
        "M_shift_mask": y_shift_mask,
        "Y_shift_ambiguity_code": y_shift_ambiguity,
        "Y_shift_all_candidates_ppm": y_shift_candidates,
        "Y_J_coupling_matrix_hz": j_matrix,
        "M_J_coupling_mask": j_mask,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile SugarCubed AI-ready tensors.")
    parser.add_argument(
        "--output-dir",
        default="training-data",
        help="Directory for compiled AI-ready tensor files (default: training-data)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify compilation in-memory without writing output files",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    catalog_path = repo_root / "sugarcubed-catalog.jsonld"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

    source_hashes = {"sugarcubed-catalog.jsonld": sha256_file(catalog_path)}
    records_by_inchi = {}

    for part in catalog["sugarcubed:dataPartitions"]:
        if part["schema:encodingFormat"] == "text/turtle":
            part_dir = repo_root / part["schema:contentUrl"]
            for ttl_file in sorted(part_dir.glob("*.ttl")):
                rel_p = str(ttl_file.relative_to(repo_root))
                source_hashes[rel_p] = sha256_file(ttl_file)
                rec = parse_glycam_ttl(ttl_file)
                records_by_inchi[rec["inchi"]] = rec

    for part in catalog["sugarcubed:dataPartitions"]:
        if part["schema:encodingFormat"] == "application/x-nmr-star":
            part_dir = repo_root / part["schema:contentUrl"]
            for str_file in sorted(part_dir.glob("*.str")):
                rel_p = str(str_file.relative_to(repo_root))
                source_hashes[rel_p] = sha256_file(str_file)
                attach_nmr_str_data(records_by_inchi, str_file, part["@id"])

    print(f"Compiled {len(records_by_inchi)} canonical ALATIS-indexed molecular graph tensors:")
    xyz_dataset = []
    expt_dataset = []

    for inchi, rec in records_by_inchi.items():
        obs = build_dense_observables(rec)
        rec.update(obs)
        xyz_dataset.append({
            "inchi": rec["inchi"],
            "glycam_sequence": rec["glycam_sequence"],
            "canonical_smiles": rec["canonical_smiles"],
            "is_fragment": rec["is_fragment"],
            "num_atoms": rec["num_atoms"],
            "node_feature_names": rec["node_feature_names"],
            "X_node_features": rec["X_node_features"],
            "R_coordinates_xyz": rec["R_coordinates_xyz"],
            "E_edge_index": rec["E_edge_index"],
            "atoms": rec["atoms"],
        })
        expt_dataset.append({
            "inchi": rec["inchi"],
            "glycam_sequence": rec["glycam_sequence"],
            "nmr_sources": rec["nmr_sources"],
            "num_atoms": rec["num_atoms"],
            "Y_shift_ppm": rec["Y_shift_ppm"],
            "M_shift_mask": rec["M_shift_mask"],
            "Y_shift_ambiguity_code": rec["Y_shift_ambiguity_code"],
            "Y_shift_all_candidates_ppm": rec["Y_shift_all_candidates_ppm"],
            "Y_J_coupling_matrix_hz": rec["Y_J_coupling_matrix_hz"],
            "M_J_coupling_mask": rec["M_J_coupling_mask"],
        })
        print(
            f"  - {rec['glycam_sequence']}: N_atoms={rec['num_atoms']}, "
            f"X shape=({rec['num_atoms']}, {len(rec['node_feature_names'])}), "
            f"R shape=({rec['num_atoms']}, 3), "
            f"directed_edges={len(rec['E_edge_index'])}, "
            f"assigned_shifts={sum(rec['M_shift_mask'])}, "
            f"J_couplings={len(rec['gissmo_spin_matrix']['j_couplings_hz'])}"
        )

    if args.verify_only:
        print("Verification complete (--verify-only).")
        return 0

    out_dir = repo_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    xyz_path = out_dir / "SugarCubed.xyz.json"
    expt_path = out_dir / "SugarCubed.expt.json"
    full_path = out_dir / "sugarcubed_tensors_full.json"

    xyz_path.write_text(json.dumps(xyz_dataset, indent=2) + "\n", encoding="utf-8")
    expt_path.write_text(json.dumps(expt_dataset, indent=2) + "\n", encoding="utf-8")
    full_path.write_text(json.dumps(list(records_by_inchi.values()), indent=2) + "\n", encoding="utf-8")

    output_hashes = {
        str(xyz_path.relative_to(repo_root)): sha256_file(xyz_path),
        str(expt_path.relative_to(repo_root)): sha256_file(expt_path),
        str(full_path.relative_to(repo_root)): sha256_file(full_path),
    }

    # Also write binary NumPy tensor archive (.npz) if numpy is installed
    try:
        import numpy as np

        npz_dict = {}
        for rec in records_by_inchi.values():
            prefix = rec["glycam_sequence"].replace("-", "_")
            npz_dict[f"{prefix}__X"] = np.array(rec["X_node_features"], dtype=np.float32)
            npz_dict[f"{prefix}__R"] = np.array(rec["R_coordinates_xyz"], dtype=np.float32)
            npz_dict[f"{prefix}__edge_index"] = np.array(rec["E_edge_index"], dtype=np.int64).T
            npz_dict[f"{prefix}__Y_shift_ppm"] = np.array(
                [np.nan if v is None else v for v in rec["Y_shift_ppm"]], dtype=np.float32
            )
            npz_dict[f"{prefix}__M_shift_mask"] = np.array(rec["M_shift_mask"], dtype=np.int64)
            npz_dict[f"{prefix}__Y_J_matrix_hz"] = np.array(
                rec["Y_J_coupling_matrix_hz"], dtype=np.float32
            )
        npz_path = out_dir / "sugarcubed_tensors.npz"
        np.savez_compressed(npz_path, **npz_dict)
        output_hashes[str(npz_path.relative_to(repo_root))] = sha256_file(npz_path)
    except ImportError:
        pass

    # Update sbom.json provenance manifest
    sbom_path = repo_root / "sbom.json"
    sbom = {
        "schemaVersion": "1.0.0",
        "generatedAtUtc": datetime.now(timezone.utc).isoformat(),
        "generatorScript": "scripts/build_tensor.py",
        "generatorSha256": sha256_file(Path(__file__).resolve()),
        "validatorScript": "scripts/validation/validate_catalog_and_data.py",
        "validatorSha256": sha256_file(
            repo_root / "scripts/validation/validate_catalog_and_data.py"
        ),
        "moleculeCount": len(records_by_inchi),
        "sourceDataSha256": source_hashes,
        "compiledOutputsSha256": output_hashes,
    }
    sbom_path.write_text(json.dumps(sbom, indent=2) + "\n", encoding="utf-8")

    print(f"\nWrote compiled AI-ready datasets to: {out_dir.relative_to(repo_root)}/")
    for rel_out in output_hashes:
        print(f"  [FILE] {rel_out}")
    print(f"Updated cryptographic provenance manifest: {sbom_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
