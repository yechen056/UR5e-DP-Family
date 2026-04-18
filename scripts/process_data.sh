#!/bin/bash
set -euo pipefail

RAW_DIR=${1:?Usage: bash scripts/process_data.sh <raw_dir> [processed_zarr] [dp3_zarr]}
PROCESSED_ZARR=${2:-${RAW_DIR}.zarr}
DP3_ZARR=${3:-data/state.zarr}

python scripts/convert_raw_to_zarr.py "$RAW_DIR" "$PROCESSED_ZARR"
python scripts/preprocess_pointcloud.py "$PROCESSED_ZARR"
python scripts/prepare_data.py "$PROCESSED_ZARR" "$DP3_ZARR"

echo "Processed zarr: $PROCESSED_ZARR"
echo "DP3 zarr: $DP3_ZARR"
