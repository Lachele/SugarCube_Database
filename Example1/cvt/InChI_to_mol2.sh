#!/bin/bash
# Build a 3D mol from an InChI string, then reorder its atoms with ALATIS.
# ALATIS reads a structure file (not an InChI) and writes a V2000 mol whose
# atom order is the ALATIS canonical NMR order.

set -u

if [ -z "${1:-}" ]; then
    echo "Usage: $0 '<InChI_String>' [output_filename.mol]"
    exit 1
fi

INCHI_STRING="$1"
OUTPUT_FILE="${2:-alatis_3D_optimized.mol}"

if [[ "$INCHI_STRING" != InChI=* ]]; then
    echo "Error: argument is not an InChI string (expected it to start with 'InChI=')."
    exit 1
fi

# Resolve the output path before creating the temporary work directory.
if [[ "$OUTPUT_FILE" != /* ]]; then
    OUTPUT_FILE="$(pwd)/$OUTPUT_FILE"
fi
mkdir -p "$(dirname "$OUTPUT_FILE")"

MATLAB_BIN="${MATLAB_BIN:-/usr/local/MATLAB/R2026b/bin/matlab}"
ALATIS_DIR="${ALATIS_DIR:-/home/rajan/Desktop/GlycoNMR/NMR_STAR_GMMO/cvt_3D_to_2D/ALATIS/alatis}"

if [ ! -x "$MATLAB_BIN" ]; then
    if command -v matlab >/dev/null 2>&1; then
        MATLAB_BIN="$(command -v matlab)"
    else
        echo "Error: MATLAB not found at $MATLAB_BIN"
        exit 1
    fi
fi

if [ ! -f "$ALATIS_DIR/alatis_command.m" ] || [ ! -f "$ALATIS_DIR/inchi-1" ]; then
    echo "Error: ALATIS source not found at $ALATIS_DIR"
    exit 1
fi

if command -v obabel >/dev/null 2>&1; then
    OBABEL="$(command -v obabel)"
elif [ -x /home/rajan/miniconda3/envs/general_env1/bin/obabel ]; then
    OBABEL="/home/rajan/miniconda3/envs/general_env1/bin/obabel"
else
    echo "Error: obabel is not on PATH and was not found in the general_env1 conda environment."
    exit 1
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/alatis_inchi.XXXXXX")"
cleanup() {
    rm -rf "$WORK"
}
trap cleanup EXIT

# inchi-1 is shipped mode 644, and alatis_command looks it up as the bare
# command "inchi-1" after changing into the input directory.
cp "$ALATIS_DIR/inchi-1" "$WORK/inchi-1"
chmod +x "$WORK/inchi-1"
export PATH="$WORK:$(dirname "$OBABEL"):$PATH"

echo "Step 1: Building a 3D V2000 mol with explicit hydrogens from the InChI..."
printf '%s\n' "$INCHI_STRING" > "$WORK/input.inchi"
if ! "$OBABEL" -iinchi "$WORK/input.inchi" -omol -O "$WORK/input.mol" -h --gen3d >/dev/null; then
    echo "Error: Open Babel could not build a 3D mol from the InChI."
    exit 1
fi
if ! grep -q 'V2000' "$WORK/input.mol"; then
    echo "Error: Open Babel did not write a V2000 mol. ALATIS only reads Mol V2000."
    exit 1
fi

echo "Step 2: Running ALATIS to assign canonical NMR atom order..."
# alatis_command(path, type, project3D, addHydrogen, send_email)
# 3D coordinates and hydrogens are already in the mol, so both flags stay '0'.
# addpath is required because alatis_command then cd's into the work directory.
if ! "$MATLAB_BIN" -batch "addpath('$ALATIS_DIR'); addpath('$ALATIS_DIR/alatis_modules'); alatis_command('$WORK/input.mol', 'mol', '0', '0', '0');"; then
    echo "Error: MATLAB exited while running ALATIS."
    exit 1
fi

ALATIS_MOL="$WORK/alatis_output_input.mol"
if [ ! -s "$ALATIS_MOL" ] || ! grep -q 'alatis_output_mol' "$ALATIS_MOL"; then
    echo "Error: ALATIS did not write an ordered mol file."
    if [ -f "$WORK/display_info" ]; then
        echo "---- ALATIS messages ----"
        cat "$WORK/display_info"
    fi
    exit 1
fi

ALATIS_ERRORS="$(awk '
    /<ALATIS_Errors>/ { grab = 1; next }
    /<ALATIS_/ { grab = 0 }
    grab && NF { print }
' "$WORK/display_info")"
if [ -n "$ALATIS_ERRORS" ]; then
    echo "Error: ALATIS reported:"
    echo "$ALATIS_ERRORS"
    exit 1
fi

cp "$ALATIS_MOL" "$OUTPUT_FILE"
echo "Success! ALATIS canonical mol saved to $OUTPUT_FILE"
