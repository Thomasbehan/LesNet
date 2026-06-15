#!/usr/bin/env bash
# Full M-4 pipeline: source + build the balanced dataset, train the gated EfficientNetV2-S
# triage model on GPU, package + evaluate it. Long-running.
# Launch inside the GPU nix-shell:
#   nix-shell -p python311 stdenv.cc.cc.lib zlib --run 'bash scripts/train_full_m4.sh'
set -uo pipefail
cd /home/thomas/LesNet

PY=/tmp/lesnet-gpu/bin/python
export LD_LIBRARY_PATH="$(nix eval --raw nixpkgs#zlib.out)/lib:$(nix eval --raw nixpkgs#stdenv.cc.cc.lib)/lib:${LD_LIBRARY_PATH:-}"
export TF_CPP_MIN_LOG_LEVEL=1

echo "[$(date)] STEP 1/4: source + build the balanced, sorted, leakage-safe dataset"
$PY commands/build_dataset.py --sources isic pad_ufes_20 fitzpatrick17k ddi --dest data/dataset
echo "[$(date)] dataset build done (exit $?)"

echo "[$(date)] STEP 2/4: train M-4 (gated, GPU, hair-removal off for throughput)"
$PY commands/run_train_triage.py --manifest data/dataset/manifest.csv --artifacts models/triage_m4 \
    --backbone efficientnetv2s --image-size 224 --batch-size 16 \
    --until-target --max-epochs 200 --no-hair-removal
echo "[$(date)] training done (exit $?)"

echo "[$(date)] STEP 3/4: package M-4 release assets"
$PY commands/package_model.py --artifacts models/triage_m4 --model-id M-4

echo "[$(date)] STEP 4/4: evaluate M-4 on held-out test split"
$PY commands/run_evaluate.py --manifest data/dataset/manifest.csv --artifacts models/triage_m4 --split test

echo "[$(date)] FULL M-4 PIPELINE COMPLETE"
