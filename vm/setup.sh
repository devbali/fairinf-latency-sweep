#!/bin/bash
# One-time setup on an A100 VM cloned from fairinf-vm-matteo-1, for running
# gpt-oss-20b (BF16) on SGLang pinned to fairinf-sglang 4e84427ab. Leaves
# hooks-venv and the VM's fairinf-sglang checkout untouched.
#
# Usage (on the VM, from this repo): bash vm/setup.sh
set -euo pipefail
LT=${LT:-$HOME/latency-test}
SIDE=${SIDE:-$LT/pydeps}
VENV=/home/devbali/fairinf/hooks-venv
PY=$VENV/bin/python
SITE=$VENV/lib/python3.10/site-packages
REPO=/home/devbali/fairinf/fairinf-sglang
mkdir -p "$LT"

# 1. SGLang exactly at 4e84427ab (matteo/fairbatching), from the VM's own repo.
if [ ! -d "$LT/fairinf-sglang-4e84427ab" ]; then
    git -C "$REPO" archive --prefix=fairinf-sglang-4e84427ab/ 4e84427ab python | tar x -C "$LT"
fi

# 2. Side directory, first on PYTHONPATH: transformers 5.6 and triton 3.6 (the
#    fork's pins), plus patched copies of the venv's sgl_kernel stub and
#    triton_kernels. See vm/patch_side_dir.py.
NUMPY=$($PY -c 'import numpy; print(numpy.__version__)')
$PY -m pip install --quiet --disable-pip-version-check --upgrade --target "$SIDE" \
    "transformers==5.6.0" "numpy==$NUMPY"
$PY -m pip install --quiet --disable-pip-version-check --no-deps --upgrade --target "$SIDE" \
    "triton==3.6.0"
rm -rf "$SIDE/sgl_kernel" "$SIDE/triton_kernels"
cp -r "$SITE/sgl_kernel" "$SITE/triton_kernels" "$SIDE/"
find "$SIDE/sgl_kernel" "$SIDE/triton_kernels" -name __pycache__ -prune -exec rm -rf {} +
$PY "$(dirname "$0")/patch_side_dir.py" "$SIDE"

# 3. The BF16 checkpoint: the MXFP4 original runs its MoE kernels ~44x slower on A100.
PYTHONPATH=$SIDE $PY -c "
from huggingface_hub import snapshot_download
print(snapshot_download('lmsys/gpt-oss-20b-bf16',
                        allow_patterns=['*.json', '*.jinja', '*.safetensors', 'tokenizer*']))"
