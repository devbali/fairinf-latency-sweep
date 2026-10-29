#!/bin/bash
# Time MODEL on GPU 0 with SGLang pinned to fairinf-sglang 4e84427ab, in the
# fresh venv from fresh_venv.sh: the fork's own package pins and the real
# sgl_kernel, with no side directory. MODEL defaults to gpt-oss-20b (BF16), and
# with no extra flags this is the setup of results/gptoss-clean-20261028.
#
# Usage (on the VM): bash vm/run_clean.sh BATCHES_FILE OUT_FILE [extra SGLang flags]
set -euo pipefail
LT=${LT:-$HOME/latency-test}
MODEL=${MODEL:-$(ls -d "$HOME"/.cache/huggingface/hub/models--lmsys--gpt-oss-20b-bf16/snapshots/*)}
cd "$(dirname "$0")/.."
export PATH=$LT/venv/bin:$PATH
export PYTHONPATH=$LT/fairinf-sglang-4e84427ab/python
export CUDA_VISIBLE_DEVICES=0

python - <<'EOF'
import flashinfer, sglang, sgl_kernel, torch, transformers
print("sglang      ", sglang.__file__)
print("sgl_kernel  ", sgl_kernel.__file__)
print("flashinfer  ", flashinfer.__version__)
print("transformers", transformers.__version__)
print("torch       ", torch.__version__, "cuda", torch.version.cuda)
print("gpu         ", torch.cuda.get_device_name(0))
EOF

# Kernels are chosen by flags passed after OUT_FILE, e.g. --moe-runner-backend
# triton_kernel|triton or --attention-backend. Later value flags win, so they can
# also override the defaults here. Piecewise CUDA graphs stay off: in this fork
# torch.compile can't trace its RMSNorm (FlashInfer's, see layernorm.py).
python runner.py --model-path "$MODEL" --dtype bfloat16 --mem-fraction-static 0.7 \
    --sampling-backend pytorch --disable-piecewise-cuda-graph \
    --batches "$1" --out "$2" "${@:3}"
