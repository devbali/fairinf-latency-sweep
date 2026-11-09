#!/bin/bash
# The PD KV transfer model of results/pd-transfer-model, from a pd_load.py run with millisecond logs.
# Usage: src/pd_transfer/run.sh RUN_DIR WORK_DIR [FIGURE.png]
# Needs numpy, scikit-learn (the two tree models) and matplotlib (the figure).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
RUN=$(cd "$1" && pwd); mkdir -p "$2"; cd "$2"
python3 "$HERE/stepmodels.py" > stepmodels.txt          # prefill step model, for the step running behind
python3 "$HERE/steps.py" "$RUN" steps.pkl               # prefill steps, cut prompts at their last chunk
python3 "$HERE/posting.py" steps.pkl posting.pkl | tee posting.txt
for m in "component model" "linear" "degree 2" "degree 2 + sum l^2" "selected" "random forest" "gradient boosting"; do
  f="scored_$(echo "$m" | sed -e 's/ /_/g' -e 's/+/p/g' -e 's/\^//g').pkl"
  REG=posting.pkl MODEL="$m" OUT="$f" python3 "$HERE/score.py" steps.pkl > "${f%.pkl}.txt" &
done
wait
python3 "$HERE/table.py" . | tee barometer.md
[ $# -ge 3 ] && python3 "$HERE/figure.py" scored_selected.pkl "$3"
true
