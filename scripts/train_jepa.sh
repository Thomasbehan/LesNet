#!/usr/bin/env bash
# Fully-automatic I-JEPA world-model pipeline for the WHOLE model-size family. Downloads + curates
# the ISIC archive once, then for each variant (Tiny/Small/Medium/Large/XLarge) pretrains the
# self-supervised encoder (TensorBoard logs live), runs a leakage-free clinical probe, and exports
# an ONNX/int8 encoder gated on a MEASURED 512 MB inference-RSS budget. Continues through the whole
# family even if the big variants exceed 512 MB (they target accuracy, not the edge); exits non-zero
# only if NO tiny/small edge variant fits the budget.
#
# Launch inside the GPU nix-shell (mirrors scripts/train_full_m4.sh):
#   nix-shell -p python311 stdenv.cc.cc.lib zlib --run 'bash scripts/train_jepa.sh'
#
# Tunables (env overrides): ENCODERS EPOCHS BATCH_SIZE IMAGE_SIZE DATA_DIR ARTIFACTS SOURCES
#   NUM_WORKERS SAMPLE_LIMIT PY   SKIP_INSTALL=1   LAUNCH_TB=1 (+TENSORBOARD_PORT)
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
ENCODERS="${ENCODERS:-vit_tiny vit_small vit_base vit_large vit_huge}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-128}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
DATA_DIR="${DATA_DIR:-data/dataset}"
ARTIFACTS="${ARTIFACTS:-artifacts/jepa}"
SOURCES="${SOURCES:-isic}"                 # add pad_ufes_20 fitzpatrick17k ddi for more data
MANIFEST="$DATA_DIR/manifest.csv"

echo "[$(date)] STEP 0: install the optional [jepa] extra"
if [ "${SKIP_INSTALL:-0}" != "1" ]; then $PY -m pip install -e ".[jepa]"; fi

echo "[$(date)] STEP 1: download + curate the ISIC archive (leakage-free grouped manifest)"
BUILD_ARGS=(--sources $SOURCES --dest "$DATA_DIR")
[ -n "${SAMPLE_LIMIT:-}" ] && BUILD_ARGS+=(--sample-limit "$SAMPLE_LIMIT")
$PY commands/build_dataset.py "${BUILD_ARGS[@]}"

if [ "${LAUNCH_TB:-0}" = "1" ]; then
  $PY -m tensorboard.main --logdir "$ARTIFACTS" --port "${TENSORBOARD_PORT:-6006}" &
fi

echo "[$(date)] STEP 2: train the model-size family: $ENCODERS"
SUMMARY="$ARTIFACTS/family_summary.tsv"
mkdir -p "$ARTIFACTS"
printf 'variant\tint8_onnx_mb\tpeak_rss_mb\tfits_512mb\tprobe_sensitivity\tprobe_auc\n' > "$SUMMARY"

for ENC in $ENCODERS; do
  OUT="$ARTIFACTS/$ENC"
  echo "[$(date)]   --- $ENC ---  (TensorBoard: tensorboard --logdir $OUT/tb)"
  if $PY commands/run_pretrain_jepa.py \
        --manifest "$MANIFEST" --encoder "$ENC" --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" --image-size "$IMAGE_SIZE" \
        --num-workers "${NUM_WORKERS:-8}" --artifacts "$OUT" --export; then
    $PY - "$ENC" "$OUT" "$SUMMARY" <<'PY'
import json, sys
enc, out, summary = sys.argv[1], sys.argv[2], sys.argv[3]
rep = json.load(open(f"{out}/export/report.json"))
try:
    probe = json.load(open(f"{out}/probe_metrics.json"))
except FileNotFoundError:
    probe = {}
row = [enc, rep.get("int8_onnx_mb"), rep.get("peak_rss_mb"), rep.get("fits_budget"),
       probe.get("sensitivity"), probe.get("roc_auc")]
open(summary, "a").write("\t".join(str(x) for x in row) + "\n")
print(f"   {enc}: int8={rep.get('int8_onnx_mb')}MB rss={rep.get('peak_rss_mb')}MB "
      f"fits512={rep.get('fits_budget')} sens={probe.get('sensitivity')} auc={probe.get('roc_auc')}")
PY
  else
    echo "   WARNING: $ENC failed (see log above); continuing with the rest of the family."
    printf '%s\tFAILED\t-\t-\t-\t-\n' "$ENC" >> "$SUMMARY"
  fi
done

echo "[$(date)] STEP 3: family summary"
cat "$SUMMARY"

echo "[$(date)] STEP 4: verify at least one edge variant fits the 512 MB budget"
$PY - "$SUMMARY" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
edge = [r for r in rows if r["variant"] in ("vit_tiny", "vit_small") and r["fits_512mb"] == "True"]
if not edge:
    raise SystemExit("No tiny/small variant fits the 512 MB budget — investigate the RSS report.")
print("  edge-deployable (<512 MB):", ", ".join(r["variant"] for r in edge))
PY

echo "[$(date)] JEPA FAMILY PIPELINE COMPLETE — artifacts in $ARTIFACTS/<variant>/"
