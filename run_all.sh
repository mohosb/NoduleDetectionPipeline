#!/usr/bin/env bash
# Run the full processing pipeline across all datasets, modes, and save modes.
#
# Usage:
#   ./run_all.sh [--sync] [--cpu]
#
#   --sync   Download / update raw data before each run (default: off)
#   --cpu    Force CPU-only single-process execution (default: off, uses available GPUs)
#
# Examples:
#   ./run_all.sh                 # GPU, no re-download
#   ./run_all.sh --cpu           # CPU, no re-download
#   ./run_all.sh --sync          # GPU, sync data first
#   ./run_all.sh --sync --cpu    # CPU, sync data first

set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT="process.py"

# --- Parse flags ---
SYNC_FLAG=""
CPU_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --sync) SYNC_FLAG="--sync" ;;
        --cpu)  CPU_FLAG="--cpu"   ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

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
        --no-compress             \
        $SYNC_FLAG                \
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
