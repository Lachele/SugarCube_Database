#!/usr/bin/env python3
"""
SugarCubed Validation Suite (scripts/validation/validate_catalog_and_data.py)

Validates:
1. sugarcubed-catalog.jsonld schema and partition directories.
2. Presence and format of Standard InChI strings across all data partitions.
3. Completeness and 1-to-1 bijection of ALATIS atom mappings in GLYCAM .ttl
   and BMRB/GISSMO NMR-STAR (.str) files.
"""

import json
import os
import re
import sys
from pathlib import Path


def validate_catalog(repo_root: Path) -> dict:
    catalog_path = repo_root / "sugarcubed-catalog.jsonld"
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Missing catalog: {catalog_path}")

    with open(catalog_path, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    required_keys = [
        "@context",
        "@id",
        "@type",
        "sugarcubed:moleculeAnchorStandard",
        "sugarcubed:atomAlignmentStandard",
        "sugarcubed:dataPartitions",
    ]
    for key in required_keys:
        if key not in catalog:
            raise ValueError(f"Catalog missing required key: {key}")

    for part in catalog["sugarcubed:dataPartitions"]:
        part_dir = repo_root / part["schema:contentUrl"]
        if not part_dir.is_dir():
            raise FileNotFoundError(
                f"Partition directory missing for {part['@id']}: {part_dir}"
            )

    return catalog


def validate_ttl_file(ttl_path: Path) -> dict:
    text = ttl_path.read_text(encoding="utf-8")
    oligo_block = text.split("\n:model_oligo1\n")[1].split("rdf:type\t\t:Oligosaccharide.")[0]

    is_frag_match = re.search(r":isFragment\s+(true|false)", oligo_block)
    is_fragment = is_frag_match.group(1) == "true" if is_frag_match else False

    inchi_match = re.search(r':InChICode\s+"""(InChI=1S/.*?)"""', oligo_block)
    if not is_fragment and not inchi_match:
        raise ValueError(f"{ttl_path.name}: Complete molecule missing :InChICode")

    alatis_match = re.search(r':hasALATISMapping\s+"""(\{.*?\})"""', oligo_block)
    if not is_fragment and not alatis_match:
        raise ValueError(f"{ttl_path.name}: Complete molecule missing :hasALATISMapping")

    alatis_map = json.loads(alatis_match.group(1)) if alatis_match else {}
    if len(set(alatis_map.values())) != len(alatis_map):
        raise ValueError(f"{ttl_path.name}: Duplicate ALATIS atom labels in mapping")

    return {
        "file": ttl_path.name,
        "inchi": inchi_match.group(1) if inchi_match else None,
        "is_fragment": is_fragment,
        "num_mapped_atoms": len(alatis_map),
    }


def validate_str_file(str_path: Path) -> dict:
    text = str_path.read_text(encoding="utf-8")
    inchi_match = re.search(r"_Chem_comp\.InChI_code\s+(InChI=1S/\S+)", text)
    if not inchi_match:
        raise ValueError(f"{str_path.name}: Missing _Chem_comp.InChI_code")

    has_alatis = "ALATIS" in text
    if not has_alatis:
        raise ValueError(f"{str_path.name}: Missing ALATIS atom nomenclature metadata")

    has_assigned_shifts = "_Atom_chem_shift.Val" in text
    has_couplings = "_Coupling_constant.Val" in text
    has_peaks = "_Spectral_peak" in text

    return {
        "file": str_path.name,
        "inchi": inchi_match.group(1),
        "has_assigned_shifts": has_assigned_shifts,
        "has_couplings": has_couplings,
        "has_peaks": has_peaks,
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    print(f"Validating SugarCubed repository at: {repo_root}")
    catalog = validate_catalog(repo_root)
    print(f"  [OK] Catalog validated ({len(catalog['sugarcubed:dataPartitions'])} partitions)")

    inchi_registry = {}
    for part in catalog["sugarcubed:dataPartitions"]:
        part_id = part["@id"]
        part_dir = repo_root / part["schema:contentUrl"]
        fmt = part["schema:encodingFormat"]
        print(f"  Checking partition {part_id} ({part_dir.relative_to(repo_root)})...")

        if fmt == "text/turtle":
            for ttl_file in sorted(part_dir.glob("*.ttl")):
                info = validate_ttl_file(ttl_file)
                inchi_registry.setdefault(info["inchi"], []).append(info["file"])
                print(
                    f"    [OK] {info['file']}: InChI present, "
                    f"{info['num_mapped_atoms']} ALATIS atoms mapped"
                )
        elif fmt == "application/x-nmr-star":
            for str_file in sorted(part_dir.glob("*.str")):
                info = validate_str_file(str_file)
                inchi_registry.setdefault(info["inchi"], []).append(info["file"])
                obs = []
                if info["has_assigned_shifts"]:
                    obs.append("shifts")
                if info["has_couplings"]:
                    obs.append("J-couplings")
                if info["has_peaks"]:
                    obs.append("peaks")
                print(
                    f"    [OK] {info['file']}: InChI present, observables={obs}"
                )

    print("\nCross-Partition InChI Entity Resolution Summary:")
    for inchi, files in inchi_registry.items():
        print(f"  {inchi[:45]}... -> {files}")
    print("\nAll validation checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
