#!/usr/bin/env bash
# Full M-4 pipeline: download the entire ISIC archive, build the manifest, train the
# gated EfficientNetV2-S triage model on GPU, package + evaluate it. Long-running.
# Launch inside the GPU nix-shell:
#   nix-shell -p python311 stdenv.cc.cc.lib zlib --run 'bash scripts/train_full_m4.sh'
set -uo pipefail
cd /home/thomas/LesNet

PY=/tmp/lesnet-gpu/bin/python
export LD_LIBRARY_PATH="$(nix eval --raw nixpkgs#zlib.out)/lib:$(nix eval --raw nixpkgs#stdenv.cc.cc.lib)/lib:${LD_LIBRARY_PATH:-}"
export TF_CPP_MIN_LOG_LEVEL=1

echo "[$(date)] STEP 1/5: download full ISIC archive (resumable, multithreaded)"
$PY commands/download_isic_full.py --out data/isic_full --workers 64
echo "[$(date)] download done (exit $?)"

echo "[$(date)] STEP 2/5: build full grouped manifest"
$PY commands/run_build_dataset.py --mode full --datasets isic \
    --isic-root data/isic_full --output data/manifest_full.csv

echo "[$(date)] STEP 3/5: train M-4 (gated, GPU, hair-removal off for throughput)"
$PY commands/run_train_triage.py --manifest data/manifest_full.csv --artifacts models/triage_m4 \
    --backbone efficientnetv2s --image-size 224 --batch-size 16 \
    --until-target --max-epochs 200 --no-hair-removal
echo "[$(date)] training done (exit $?)"

echo "[$(date)] STEP 4/5: package M-4 release assets"
$PY commands/package_model.py --artifacts models/triage_m4 --model-id M-4

echo "[$(date)] STEP 5/5: evaluate M-4 on held-out test split"
$PY commands/run_evaluate.py --manifest data/manifest_full.csv --artifacts models/triage_m4 --split test

echo "[$(date)] FULL M-4 PIPELINE COMPLETE"
