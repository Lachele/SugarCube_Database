# SugarCubed Data-to-Tensor Pipeline Documentation

This document explains how the SugarCubed repository validates domain-native carbohydrate structural and NMR datasets and compiles them into **AI-ready molecular graph tensors** inside the git-ignored `training-data/` directory.

---

## 1. Quick Start: Running the Pipeline

From the root of this repository (`Main_Repo_Directory_mockup/`), you can run both the validation suite and the tensor compiler using standard Python 3 (no external packages are required for JSON tensor generation; if `numpy` is installed, a compressed binary `.npz` tensor bundle is also produced automatically).

### Step 1: Validate the Catalog, InChI Strings, and ALATIS Atom Maps
```bash
python3 scripts/validation/validate_catalog_and_data.py
```
**What this checks:**
1. Verifies that `sugarcubed-catalog.jsonld` is valid and that every listed `schema:contentUrl` directory exists under `data/`.
2. Audits every GLYCAM `.ttl` file in `data/glycam-simulation-structures/` to confirm that complete molecules (`:isFragment false`) contain a valid IUPAC Standard InChI (`:InChICode "InChI=1S/..."`) and a complete, 1-to-1 `:hasALATISMapping` dictionary.
3. Audits every NMR-STAR `.str` file in `data/bmrb-nmr-data/` and `data/gissmo-nmr-simulation-data/` to confirm that `_Chem_comp.InChI_code` and `ALATIS` atom nomenclature metadata are present.
4. Prints a cross-partition entity resolution table showing which files across different directories belong to the same carbohydrate molecule.

### Step 2: Compile the AI-Ready Tensors into `training-data/`
```bash
python3 scripts/build_tensor.py
```
*(Optional flags: `--output-dir training-data` to change the output folder, or `--verify-only` to run an in-memory dry-run check in CI without writing files.)*

**What this generates:**
* **`training-data/SugarCubed.xyz.json`** — The structural & ontology graph dataset (`SugarCubed.xyz`), containing canonical ALATIS-ordered 3D coordinate tensors $\mathbf{R}$, node feature matrices $\mathbf{X}$, directed covalent bond edge lists $\mathbf{E}_{\text{index}}$, and per-atom GLYCAM ontology metadata.
* **`training-data/SugarCubed.expt.json`** — The experimental and spin-simulated NMR dataset (`SugarCubed.expt`), containing ALATIS-ordered chemical shift vectors $\mathbf{Y}_{\text{shift}}$, observation masks $\mathbf{M}_{\text{shift}}$, BMRB ambiguity codes, and $N_{\text{atoms}} \times N_{\text{atoms}}$ scalar coupling ($J$-coupling) spin matrices $\mathbf{Y}_{J}$.
* **`training-data/sugarcubed_tensors_full.json`** — The combined multi-modal records joining structure, ontology features, and NMR observables in a single file.
* **`training-data/sugarcubed_tensors.npz`** *(generated automatically when `numpy` is installed)* — Compressed binary NumPy tensor arrays (`__X`, `__R`, `__edge_index`, `__Y_shift_ppm`, `__M_shift_mask`, `__Y_J_matrix_hz`) ready for zero-copy loading into PyTorch, JAX, or DGL.
* **`sbom.json`** — Updated automatically in the repository root with a UTC timestamp and SHA-256 cryptographic hashes of `sugarcubed-catalog.jsonld`, all input `.ttl`/`.str` files, the pipeline scripts, and all compiled output files in `training-data/`.

---

## 2. Repository Layout & Design Rationale

```text
.
├── sugarcubed-catalog.jsonld          # Declarative JSON-LD catalog of active data partitions
├── sbom.json                          # Cryptographic Software Bill of Materials (SHA-256 audit trail)
├── .gitignore                         # Ignores generated `training-data/` locally
├── data/                              # Domain-native source partitions (tracked in Git)
│   ├── glycam-simulation-structures/  # Distilled GLYCAM 3D structures & ontology (.ttl)
│   ├── bmrb-nmr-data/                 # Experimental NMR-STAR files from BMRB (.str)
│   └── gissmo-nmr-simulation-data/    # Simulated spin-matrix NMR-STAR files from GISSMO (.str)
├── scripts/
│   ├── validation/
│   │   └── validate_catalog_and_data.py
│   └── build_tensor.py                # Compiles `data/` -> `training-data/` & updates `sbom.json`
├── training-data/                     # Git-ignored output folder for compiled AI-ready tensors
└── documentation/
    └── README.md                      # This guide
```

### Why `training-data/` is Git-Ignored
Keeping `training-data/` in `.gitignore` follows best practices for reproducible ML data engineering:
1. **No binary bloat in Git history:** Source files in `data/` are human-readable text (`.ttl`, `.str`), while compiled tensor bundles (`training-data/`) are deterministic build artifacts that anyone can regenerate locally with `python3 scripts/build_tensor.py` or download pre-built from GitHub Releases, Zenodo, or Hugging Face.
2. **Customizable local cuts:** Researchers can clone the repository, add their own work-in-progress `.ttl` or `.str` files into `data/`, run `python3 scripts/build_tensor.py`, and immediately train on their custom `training-data/` folder without polluting the upstream git history.
3. **Cryptographic verification via `sbom.json`:** Even though `training-data/` is git-ignored, `sbom.json` records the exact SHA-256 hashes of the source files and compiled outputs for every official release.

---

## 3. How the Two-Level Alignment (InChI + ALATIS) Works

1. **Molecule-Level Join (`InChI`):**
   * Every complete carbohydrate structure has an identical IUPAC Standard InChI string across all partitions:
     * In GLYCAM `.ttl`: `:InChICode """InChI=1S/C12H22O11/..."""`
     * In BMRB & GISSMO `.str`: `_Chem_comp.InChI_code InChI=1S/C12H22O11/...`
2. **Atom-Level Join (`ALATIS` 1-Based Canonical Indexing):**
   * Standard InChI numbers every atom in a molecule canonically from $1 \dots N_{\text{atoms}}$ (heavy atoms first in connection-table order, followed by hydrogens). ALATIS uses this exact $1 \dots N_{\text{atoms}}$ numbering (`C1`..`C12`, `O13`..`O23`, `H24`..`H45`).
   * Each GLYCAM `.ttl` file stores `:hasALATISMapping`, which maps every GLYCAM residue-atom identifier (e.g., `"1_1GA_C1": "C11"`) to its ALATIS label.
   * `scripts/build_tensor.py` places **ALATIS atom index $k$ ($1 \le k \le N_{\text{atoms}}$) at 0-based tensor row $k - 1$ ($0 \le \text{row} < N_{\text{atoms}}$)** across every tensor:
     * Row $k-1$ of `X_node_features` $\rightarrow$ features of ALATIS atom $k$
     * Row $k-1$ of `R_coordinates_xyz` $\rightarrow$ $[x, y, z]$ coordinates ($\text{\AA}$) of ALATIS atom $k$
     * Entry $k-1$ of `Y_shift_ppm` $\rightarrow$ chemical shift (ppm) of ALATIS atom $k$
     * Entry $(i-1, j-1)$ of `Y_J_coupling_matrix_hz` $\rightarrow$ scalar $J$-coupling (Hz) between ALATIS atoms $i$ and $j$

---

## 4. Loading `training-data/` in Python / PyTorch Geometric

### Option A: Loading the JSON Tensors (Standard Python / PyTorch)
```python
import json
import torch

with open("training-data/sugarcubed_tensors_full.json") as f:
    dataset = json.load(f)

mol = dataset[0]  # e.g., Sucrose (DFrufb2-1DGlcpa)
X = torch.tensor(mol["X_node_features"], dtype=torch.float32)          # (N_atoms, 7)
R = torch.tensor(mol["R_coordinates_xyz"], dtype=torch.float32)        # (N_atoms, 3)
edge_index = torch.tensor(mol["E_edge_index"], dtype=torch.long).t()   # (2, N_directed_edges)
J_matrix = torch.tensor(mol["Y_J_coupling_matrix_hz"], dtype=torch.float32)  # (N_atoms, N_atoms)
```

### Option B: Loading the Binary NumPy Archive (`sugarcubed_tensors.npz`)
```python
import numpy as np

archive = np.load("training-data/sugarcubed_tensors.npz")
X = archive["DFrufb2_1DGlcpa__X"]              # float32 (45, 7)
R = archive["DFrufb2_1DGlcpa__R"]              # float32 (45, 3)
edge_index = archive["DFrufb2_1DGlcpa__edge_index"]  # int64 (2, 92)
Y_shift = archive["DFrufb2_1DGlcpa__Y_shift_ppm"]    # float32 (45,)
```

---

## 5. Adding New Data or Partitions

1. **Adding molecules to an existing partition:** Drop new `.ttl` files (with `:InChICode` and `:hasALATISMapping`) into `data/glycam-simulation-structures/` or new `.str` files into `data/bmrb-nmr-data/` or `data/gissmo-nmr-simulation-data/`, then run `python3 scripts/validation/validate_catalog_and_data.py` followed by `python3 scripts/build_tensor.py`.
2. **Adding a new data source directory (e.g., `data/hmdb-nmr-data/` or `data/casper-predicted-shifts/`):**
   * Create the subdirectory under `data/`.
   * Add a corresponding partition entry to `sugarcubed:dataPartitions` in `sugarcubed-catalog.jsonld` specifying its `schema:contentUrl`, `schema:encodingFormat`, `sugarcubed:moleculeAnchorField`, and `sugarcubed:observables`.
