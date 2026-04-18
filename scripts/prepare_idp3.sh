#!/bin/bash
set -euo pipefail

RAW_DIR=${1:?Usage: bash scripts/prepare_idp3.sh <raw_dir> [processed_zarr] [idp3_zarr]}
PROCESSED_ZARR=${2:-${RAW_DIR}.zarr}
IDP3_ZARR=${3:-data/state_idp3.zarr}

python scripts/convert_raw_to_zarr.py "$RAW_DIR" "$PROCESSED_ZARR"
python scripts/preprocess_pointcloud.py \
    "$PROCESSED_ZARR" \
    --point-count 4096 \
    --sampling-method voxel_uniform \
    --voxel-size 0.005 \
    --output-zarr "$IDP3_ZARR"

echo "Processed zarr: $PROCESSED_ZARR"
echo "iDP3 zarr: $IDP3_ZARR"
