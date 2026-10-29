#!/bin/bash
# A fresh venv for the latency sweep with fairinf-sglang 4e84427ab's own pins:
# torch 2.11.0 (CUDA 13.0), the real sglang-kernel 0.4.2.post1, flashinfer
# 0.6.8.post1 and transformers 5.6.0. hooks-venv and the pydeps side directory
# are neither used nor changed. SGLang itself still comes from PYTHONPATH.
# requirements-4e84427ab.txt is the fork's python/pyproject.toml dependency list.
#
# Usage (on the VM): bash vm/fresh_venv.sh
set -euxo pipefail
REQS=$(cd "$(dirname "$0")" && pwd)/requirements-4e84427ab.txt
LT=${LT:-$HOME/latency-test}
VENV=$LT/venv
export UV_CACHE_DIR=$LT/.uv-cache  # same disk as the venv, so uv hardlinks instead of copying
python3 -m venv "$VENV"
"$VENV/bin/pip" install --no-cache-dir -q uv
install() { "$VENV/bin/uv" pip install --python "$VENV/bin/python" "$@"; }
install torch==2.11.0 torchvision torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
install -r "$REQS"
# The fork leaves kernels unpinned, and releases from 0.13 on break importing
# transformers 5.6.0, which itself declares kernels>=0.12,<0.13.
install "kernels>=0.12.0,<0.13"
# Prebuilt FlashInfer modules, as SGLang's own Docker image installs, so nothing
# needs compiling with the VM's CUDA 12.4 nvcc.
install flashinfer-jit-cache==0.6.8.post1 --index-url https://flashinfer.ai/whl/cu130
# No separate triton_kernels install: sglang-kernel ships the copy of OpenAI's
# triton_kernels that this SGLang calls, and Triton's own releases have a newer,
# incompatible routing API. SGLang uses it for gpt-oss's experts by default;
# pass --moe-runner-backend triton_kernel or triton to choose.
"$VENV/bin/uv" cache clean
"$VENV/bin/uv" pip list --python "$VENV/bin/python" 2>/dev/null |
    grep -iE "^(torch|torchvision|torchaudio|triton|sglang-kernel|flashinfer|transformers|cuda-python|nvidia-cutlass-dsl) "

# The real sgl_kernel must load and compute on the A100. These are the ops the
# venv's stub faked: INT8 GEMM, MoE routing and token grouping, RMSNorm.
"$VENV/bin/python" - <<'EOF'
import torch
import sgl_kernel

print("torch", torch.__version__, "cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0))
print("sgl_kernel", sgl_kernel.__file__)
a = torch.randint(-128, 127, (32, 2816), dtype=torch.int8, device="cuda")
b = torch.randint(-128, 127, (8192, 2816), dtype=torch.int8, device="cuda").t()
one_a, one_b = torch.ones(32, 1, device="cuda"), torch.ones(8192, 1, device="cuda")
out = sgl_kernel.int8_scaled_mm(a, b, one_a, one_b, torch.bfloat16).float()
ref = torch._int_mm(a, b).float()
print("int8_scaled_mm", tuple(out.shape), "max rel err", ((out - ref).abs() / ref.abs().clamp(min=1e4)).max().item())

x = torch.randn(64, 2816, dtype=torch.bfloat16, device="cuda")
w = torch.randn(2816, dtype=torch.bfloat16, device="cuda")
ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()).bfloat16()
print("rmsnorm max abs err", (sgl_kernel.rmsnorm(x, w, 1e-6) - ref).abs().max().item())

logits = torch.randn(64, 128, dtype=torch.float32, device="cuda")
weights = torch.empty(64, 8, dtype=torch.float32, device="cuda")
ids = torch.empty(64, 8, dtype=torch.int32, device="cuda")
sgl_kernel.topk_softmax(weights, ids, logits, False)
ref_w, ref_ids = torch.softmax(logits, -1).topk(8)
print("topk_softmax ids match", torch.equal(ids.sort(-1).values, ref_ids.int().sort(-1).values),
      "| weights max abs err", (weights.sort(-1).values - ref_w.sort(-1).values).abs().max().item())
EOF
echo VENV_DONE
