#!/bin/bash
# Time MODEL on GPU 0 with SGLang pinned to fairinf-sglang 4e84427ab, in the
# fresh venv from fresh_venv.sh: the fork's own package pins and the real
# sgl_kernel, with no side directory. MODEL defaults to gpt-oss-20b (BF16), and
# with no extra flags this is the setup of results/gptoss-clean-20261028.
#
# Usage (on the VM, from the repo root):
#   [PIECEWISE=off|inductor] bash vm/run_clean.sh BATCHES_FILE OUT_FILE [extra SGLang flags]
#
# PIECEWISE picks how prefill and hybrid steps run (decode always runs as CUDA
# graphs):
# - off, the default: eagerly, kernel by kernel.
# - inductor: as piecewise CUDA graphs, split at attention and compiled by torch
#   inductor. That compiler runs SGLang's norms, RoPE and activations as torch
#   code, which matters here: torch.compile can't trace this fork's RMSNorm
#   (FlashInfer's, see layernorm.py), so the default eager compiler fails.
#   Inductor's caches are off too: SGLang's compile hook hands torch 2.11's
#   graph cache a stand-in shape env without var_to_hint_override, and with the
#   caches off torch never asks for it. Compiling then takes about 90 s and
#   capturing about 70 s more, on every start.
set -euo pipefail
LT=${LT:-$HOME/latency-test}
MODEL=${MODEL:-$(ls -d "$HOME"/.cache/huggingface/hub/models--lmsys--gpt-oss-20b-bf16/snapshots/*)}
case ${PIECEWISE:-off} in
    off) GRAPHS=(--disable-piecewise-cuda-graph) ;;
    inductor)
        GRAPHS=(--piecewise-cuda-graph-compiler inductor)
        export TORCHINDUCTOR_FORCE_DISABLE_CACHES=1
        ;;
    *) echo "PIECEWISE must be off or inductor" >&2; exit 2 ;;
esac
cd "$(dirname "$0")/.."
export PATH=$LT/venv/bin:$PATH
export PYTHONPATH=$LT/fairinf-sglang-4e84427ab/python
export CUDA_VISIBLE_DEVICES=0

python src/versions.py  # the setup lines src/compare_runs.py compares

# Kernels are chosen by flags passed after OUT_FILE, e.g. --moe-runner-backend
# triton_kernel|triton or --attention-backend. Later value flags win, so they can
# also override the defaults here.
python src/runner.py --model-path "$MODEL" --dtype bfloat16 --mem-fraction-static 0.7 \
    --sampling-backend pytorch "${GRAPHS[@]}" \
    --batches "$1" --out "$2" "${@:3}"
