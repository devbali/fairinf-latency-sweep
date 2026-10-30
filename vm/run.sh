#!/bin/bash
# Time gpt-oss-20b (BF16) on one GPU with SGLang pinned to fairinf-sglang
# 4e84427ab, in hooks-venv plus the side directory from vm/setup.sh.
#
# Usage (on the VM): bash vm/run.sh BATCHES_FILE OUT_FILE [extra SGLang flags]
set -euo pipefail
LT=${LT:-$HOME/latency-test}
SIDE=${SIDE:-$LT/pydeps}
MODEL=${MODEL:-$(ls -d "$HOME"/.cache/huggingface/hub/models--lmsys--gpt-oss-20b-bf16/snapshots/*)}
cd "$(dirname "$0")/.."
export PATH=/home/devbali/fairinf/hooks-venv/bin:/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=$SIDE:$LT/fairinf-sglang-4e84427ab/python
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Leave --attention-backend unset: SGLang picks triton for gpt-oss on A100.
python src/runner.py --model-path "$MODEL" --dtype bfloat16 --mem-fraction-static 0.7 \
    --sampling-backend pytorch --disable-piecewise-cuda-graph \
    --batches "$1" --out "$2" "${@:3}"
