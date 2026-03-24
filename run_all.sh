#!/usr/bin/env bash
# Run the full processing pipeline across all datasets, modes, and save modes.
#
# Flags used:
#   --sync         Download / update raw data before processing
#   --no-compress  Save uncompressed NPZ files
#   --cpu          Force CPU-only execution
#
# Edit CPU_FLAG below to switch to GPU workers (e.g. remove it or set --workers N).

set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT="process.py"
CPU_FLAG="--cpu"  # remove or replace with "--workers N" for GPU machines

run() {
    local dataset="$1"
    local mode="$2"
    local save_mode="$3"
    echo ""
    echo "========================================================"
    echo "  dataset=${dataset}  mode=${mode}  save-mode=${save_mode}"
    echo "========================================================"
    $PYTHON "$SCRIPT" \
        --dataset    "$dataset"   \
        --mode       "$mode"      \
        --save-mode  "$save_mode" \
        --sync                    \
        --no-compress             \
        $CPU_FLAG
}

# lidc_idri — nodule only
run lidc_idri       nodule 3d
run lidc_idri       nodule 2d

# nsclc_radiomics — nodule + roi
run nsclc_radiomics nodule 3d
run nsclc_radiomics nodule 2d
run nsclc_radiomics roi    3d
run nsclc_radiomics roi    2d

# nlst_labeled — nodule + roi
run nlst_labeled    nodule 3d
run nlst_labeled    nodule 2d
run nlst_labeled    roi    3d
run nlst_labeled    roi    2d

echo ""
echo "All runs complete."
