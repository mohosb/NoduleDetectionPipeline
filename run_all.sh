#!/usr/bin/env bash
# Run the full processing pipeline across all datasets and modes.
#
# By default runs all save-mode, save-format, and label-type combinations.
# Any of these can be narrowed with flags (see Usage below).
#
# Usage:
#   ./run_all.sh [--sync] [--cpu]
#                [--save-mode  3d|2d|3d,2d]
#                [--save-format numpy|nifti|numpy,nifti]
#                [--label-type semantic|instance|semantic,instance]
#
#   --sync         Download / update raw data before each run (default: off)
#   --cpu          Force CPU-only single-process execution (default: off, uses available GPUs)
#   --save-mode    Comma-separated list of save modes   (default: 3d,2d)
#   --save-format  Comma-separated list of save formats (default: numpy,nifti)
#   --label-type   Comma-separated list of label types  (default: semantic,instance)
#                  Note: 'instance' is silently dropped for roi mode runs.
#
# Examples:
#   ./run_all.sh                                          # all combinations, GPU
#   ./run_all.sh --cpu                                    # all combinations, CPU
#   ./run_all.sh --save-mode 3d --save-format nifti       # 3D NIfTI only
#   ./run_all.sh --label-type semantic                    # semantic masks only
#   ./run_all.sh --sync --save-format numpy               # sync data, NumPy only

set -euo pipefail

PYTHON="${PYTHON:-python}"
SCRIPT="process.py"

# --- Defaults ---
SYNC_FLAG=""
CPU_FLAG=""
SAVE_MODE="3d,2d"
SAVE_FORMAT="numpy,nifti"
LABEL_TYPE="semantic,instance"

# --- Parse flags ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --sync)         SYNC_FLAG="--sync" ; shift ;;
        --cpu)          CPU_FLAG="--cpu"   ; shift ;;
        --save-mode)    SAVE_MODE="$2"     ; shift 2 ;;
        --save-format)  SAVE_FORMAT="$2"   ; shift 2 ;;
        --label-type)   LABEL_TYPE="$2"    ; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Strip 'instance' from label-type for roi mode (not supported).
roi_label_type() {
    echo "$LABEL_TYPE" | tr ',' '\n' | grep -v '^instance$' | paste -sd ',' -
}

run() {
    local dataset="$1"
    local mode="$2"
    local label_type="$3"
    echo ""
    echo "========================================================"
    echo "  dataset=${dataset}  mode=${mode}"
    echo "  save-mode=${SAVE_MODE}  save-format=${SAVE_FORMAT}  label-type=${label_type}"
    echo "========================================================"
    $PYTHON "$SCRIPT" \
        --dataset     "$dataset"    \
        --mode        "$mode"       \
        --save-mode   "$SAVE_MODE"  \
        --save-format "$SAVE_FORMAT"\
        --label-type  "$label_type" \
        --no-compress               \
        $SYNC_FLAG                  \
        $CPU_FLAG
}

# lidc_idri — nodule only
run lidc_idri       nodule "$LABEL_TYPE"

# nsclc_radiomics — nodule + roi
run nsclc_radiomics nodule "$LABEL_TYPE"
run nsclc_radiomics roi    "$(roi_label_type)"

# nlst_labeled — nodule + roi
run nlst_labeled    nodule "$LABEL_TYPE"
run nlst_labeled    roi    "$(roi_label_type)"

echo ""
echo "All runs complete."
