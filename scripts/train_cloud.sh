#!/usr/bin/env bash
# Provision a rented GPU box and pretrain the I-JEPA world model on the FULL ISIC archive.
# Single GPU or multi-GPU (torchrun/DDP) — the same script, decided by how many GPUs it finds.
#
#   curl -sSL <this file> | bash                 # or: bash scripts/train_cloud.sh
#   VERIFY_ONLY=1 bash scripts/train_cloud.sh    # provision + self-check, then stop
#
# It ALWAYS runs a self-check before the real run: a two-rank DDP smoke plus a short live-data
# training burst on the actual hardware. A broken run then fails in minutes instead of after
# hours of paid time. Set SKIP_VERIFY=1 only if you have already verified on this exact box.
#
# Tunables: ENCODER EPOCHS BATCH GRAD_ACCUM IMAGE_SIZE DATA_DIR ARTIFACTS NUM_WORKERS
#           PROBE_EVERY LR REPO BRANCH VERIFY_ONLY SKIP_VERIFY SKIP_DOWNLOAD
set -euo pipefail

REPO="${REPO:-https://github.com/Thomasbehan/LesNet.git}"
BRANCH="${BRANCH:-feat/jepa-world-model}"
WORKDIR="${WORKDIR:-$HOME/lesnet}"
DATA_DIR="${DATA_DIR:-$WORKDIR/data/isic_384}"
ARTIFACTS="${ARTIFACTS:-$WORKDIR/artifacts/jepa_cloud}"
ENCODER="${ENCODER:-vit_large}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
EPOCHS="${EPOCHS:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PROBE_EVERY="${PROBE_EVERY:-2}"

echo "=== [$(date -u)] LesNet I-JEPA cloud training ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "no GPU"; exit 1; }
NGPU="$(nvidia-smi --list-gpus | wc -l)"
VRAM_GB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{print int($1/1024)}')"
echo "GPUs: $NGPU x ${VRAM_GB}GB"

# Batch sized to VRAM. The 8 GB laptop that this was developed on had to accumulate heavily and
# still paged; on an 80 GB card ViT-L runs the reference batch outright, which is both faster and
# a better optimisation regime (no accumulation, one mask shape per step).
if [ -z "${BATCH:-}" ]; then
  if   [ "$VRAM_GB" -ge 70 ]; then BATCH=64
  elif [ "$VRAM_GB" -ge 40 ]; then BATCH=32
  elif [ "$VRAM_GB" -ge 20 ]; then BATCH=16
  else BATCH=8; fi
fi
# Global effective batch 2048 across all ranks (I-JEPA's reference regime), via accumulation.
if [ -z "${GRAD_ACCUM:-}" ]; then
  GRAD_ACCUM=$(( 2048 / (BATCH * NGPU) )); [ "$GRAD_ACCUM" -lt 1 ] && GRAD_ACCUM=1
fi
echo "plan: $ENCODER ${IMAGE_SIZE}px batch=$BATCH x accum=$GRAD_ACCUM x ${NGPU} gpu "
echo "      = effective $(( BATCH * GRAD_ACCUM * NGPU )) images/step, $EPOCHS epochs"

echo "=== [$(date -u)] STEP 1: code + deps ==="
if [ ! -d "$WORKDIR/.git" ]; then git clone --branch "$BRANCH" --depth 1 "$REPO" "$WORKDIR"; fi
cd "$WORKDIR"
python -m pip install -q --upgrade pip setuptools wheel
python -m pip install -q -e ".[jepa]"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "=== [$(date -u)] STEP 2: dataset (full ISIC archive at 384px, ~6 GB, resumable) ==="
if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
  python commands/download_isic_archive.py --dest "$DATA_DIR" --size 384 --workers 64 \
    --progress-every 5000
fi
COUNT="$(find "$DATA_DIR/images" -name '*.jpg' | wc -l)"
echo "dataset: $COUNT images"
[ "$COUNT" -lt 400000 ] && { echo "FATAL: expected ~550k images, found $COUNT"; exit 1; }

# The labelled manifest supplies the in-training probe AND the held-out ids kept out of SSL.
PROBE_MANIFEST="${PROBE_MANIFEST:-$WORKDIR/data/isic_dx/manifest.csv}"
PROBE_ARGS=()
if [ -f "$PROBE_MANIFEST" ]; then
  PROBE_ARGS=(--exclude-manifest "$PROBE_MANIFEST" --probe-manifest "$PROBE_MANIFEST"
              --probe-every "$PROBE_EVERY" --probe-subset 3000 --select-on probe)
else
  echo "WARNING: no labelled manifest at $PROBE_MANIFEST — selecting on SSL loss and NOT"
  echo "         excluding held-out images. Copy data/isic_dx/ over for an honest evaluation."
fi

if [ "${SKIP_VERIFY:-0}" != "1" ]; then
  echo "=== [$(date -u)] STEP 3: self-check BEFORE spending GPU hours ==="
  echo "--- 3a: two-rank DDP smoke (gloo/CPU): sync, lockstep, single writer ---"
  python tests/ddp_smoke.py /tmp/ddp_smoke.json
  echo "--- 3b: single-GPU end-to-end on synthetic data ---"
  python commands/run_pretrain_jepa.py --smoke --artifacts /tmp/lesnet_smoke
  echo "--- 3c: 3-minute burst on the REAL data at the REAL settings ---"
  timeout 600 python commands/run_pretrain_jepa.py \
    --data-dir "$DATA_DIR/images" "${PROBE_ARGS[@]}" \
    --encoder "$ENCODER" --image-size "$IMAGE_SIZE" --batch-size "$BATCH" \
    --grad-accum "$GRAD_ACCUM" --predictor-dim 384 --predictor-depth 12 \
    --drop-path 0.1 --layerscale 1e-4 --epochs 1 --warmup-epochs 0 --probe-every 0 \
    --select-on loss --num-workers "$NUM_WORKERS" --mixed-precision fp16 --no-hair-removal \
    --max-train-seconds 180 --artifacts /tmp/lesnet_burst || true
  test -f /tmp/lesnet_burst/context_encoder.pt || { echo "FATAL: burst produced no encoder"; exit 1; }

  if [ "$NGPU" -gt 1 ]; then
    # The gloo smoke proves the control flow; this proves NCCL on THIS box. It cannot be tested
    # on the dev machine (Windows has no NCCL, and one GPU), so it has to happen here — before
    # the long run, not four hours into it.
    echo "--- 3d: ${NGPU}-rank torchrun burst on real GPUs (NCCL) ---"
    timeout 900 torchrun --standalone --nproc_per_node="$NGPU" \
      commands/run_pretrain_jepa.py \
      --data-dir "$DATA_DIR/images" "${PROBE_ARGS[@]}" \
      --encoder "$ENCODER" --image-size "$IMAGE_SIZE" --batch-size "$BATCH" \
      --grad-accum "$GRAD_ACCUM" --predictor-dim 384 --predictor-depth 12 \
      --drop-path 0.1 --layerscale 1e-4 --epochs 1 --warmup-epochs 0 --probe-every 0 \
      --select-on loss --num-workers "$NUM_WORKERS" --mixed-precision fp16 --no-hair-removal \
      --max-train-seconds 240 --artifacts /tmp/lesnet_ddp_burst || true
    test -f /tmp/lesnet_ddp_burst/context_encoder.pt || {
      echo "FATAL: multi-GPU burst produced no encoder — do NOT start the long run"; exit 1; }
    echo "multi-GPU burst OK"
  fi
  echo "self-check PASSED"
fi
[ "${VERIFY_ONLY:-0}" = "1" ] && { echo "VERIFY_ONLY set — stopping before the real run."; exit 0; }

echo "=== [$(date -u)] STEP 4: pretrain ==="
mkdir -p "$ARTIFACTS"
TRAIN_ARGS=(commands/run_pretrain_jepa.py
  --data-dir "$DATA_DIR/images" "${PROBE_ARGS[@]}"
  --encoder "$ENCODER" --image-size "$IMAGE_SIZE"
  --batch-size "$BATCH" --grad-accum "$GRAD_ACCUM"
  --predictor-dim 384 --predictor-depth 12 --drop-path 0.1 --layerscale 1e-4
  --epochs "$EPOCHS" --patience 6 --num-workers "$NUM_WORKERS"
  --mixed-precision fp16 --no-hair-removal --artifacts "$ARTIFACTS")
[ -f "$ARTIFACTS/training_state.pt" ] && TRAIN_ARGS+=(--resume "$ARTIFACTS/training_state.pt")

if [ "$NGPU" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$NGPU" "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$ARTIFACTS/train.log"
else
  python "${TRAIN_ARGS[@]}" 2>&1 | tee -a "$ARTIFACTS/train.log"
fi

echo "=== [$(date -u)] STEP 5: export int8/fp16 tiers + measured RSS gate ==="
python commands/run_pretrain_jepa.py --data-dir "$DATA_DIR/images" --encoder "$ENCODER" \
  --artifacts "$ARTIFACTS" --epochs 0 --quant-tiers --no-early-stopping 2>/dev/null || \
  python - "$ARTIFACTS" <<'PY'
import json, sys
from pathlib import Path
from lesnet.jepa.export import export_tiers
out = Path(sys.argv[1]) / 'export'
print(json.dumps(export_tiers(Path(sys.argv[1]) / 'context_encoder.pt', out), indent=2))
PY

echo "=== [$(date -u)] DONE. Artifacts in $ARTIFACTS ==="
echo "Pull them back with:  rsync -avz <user>@<host>:$ARTIFACTS ./"
