#!/bin/bash
# Write profiles/: the FairInference experiments' step-latency profiles (format 2), the p95
# pooled models plus the isolated models for each experiment's number of users, with a
# STATIC_GAP_MS (10) static gap on every isolated decode step.
#
# Only the -final models are exported: the ones the experiments use. When a model is refitted
# on new data, the model it replaces gets another name in export_profile.py and the new one
# becomes -final, so there is one -final model per setup.
#
# Usage: bash src/make_profiles.sh   (PYTHON=python with scikit-learn, if python3 lacks it)
set -euo pipefail
cd "$(dirname "$0")"
OUT=$(cd .. && pwd)/profiles
PERCENTILE=${PERCENTILE:-95}
STATIC_GAP_MS=${STATIC_GAP_MS:-10}
mkdir -p "$OUT"
for spec in gpt-oss-20b-a100-final:4 gpt-oss-20b-a100-final:8 gpt-oss-120b-tp4-final:4; do
    model=${spec%:*} n=${spec#*:}
    out=$OUT/$model-p$PERCENTILE-n$n.json
    ${PYTHON:-python3} -m estimators.export_profile "$model" --percentile "$PERCENTILE" --n "$n" \
        --static-gap-ms "$STATIC_GAP_MS" --out "$out"
    echo "wrote profiles/${out##*/}"
done
# Hybrid batching (Micro 2): a MIXED_GAP_MS (50) extra term instead of 10 ms, the time a step may
# grow by prefill beyond the largest pooled decode step (D^MT, over at most 256 running requests).
MIXED_GAP_MS=${MIXED_GAP_MS:-50}
out=$OUT/gpt-oss-20b-a100-final-p$PERCENTILE-n4-mixed.json
${PYTHON:-python3} -m estimators.export_profile gpt-oss-20b-a100-final --percentile "$PERCENTILE" --n 4 \
    --static-gap-ms "$MIXED_GAP_MS" --out "$out"
echo "wrote profiles/${out##*/}"
