"""Patch the side directory's copies of sgl_kernel and triton_kernels.

1. sgl_kernel is the hooks venv's hand-written A100 stub. Its generic submodule
   stub makes sgl_kernel.utils.is_arch_support_pdl() return an empty CUDA
   tensor, which SGLang's Triton attention passes on as a kernel constexpr. On
   A100 (no PDL) the real function returns False.
2. triton_kernels reads torch's shared_memory_per_block_optin, which torch 2.5.1
   lacks; read the same per-block limit from the Triton driver instead. It also
   re-queries device properties and the current target on every matmul (about
   1 ms each); they are constant per process, so cache them.

Usage: patch_side_dir.py SIDE_DIR
"""

import sys

side = sys.argv[1]


def replace(rel, old, new, count):
    path = f"{side}/{rel}"
    src = open(path).read()
    assert src.count(old) == count, f"{rel}: expected {count} x {old!r}, found {src.count(old)}"
    open(path, "w").write(src.replace(old, new))


def append(rel, text):
    with open(f"{side}/{rel}", "a") as f:
        f.write(text)


append(
    "sgl_kernel/__init__.py",
    """
# latency-test side directory: the submodule stub above makes
# sgl_kernel.utils.is_arch_support_pdl() return an empty CUDA tensor; on A100
# (no PDL) the real function returns False.
_sys.modules["sgl_kernel.utils"].is_arch_support_pdl = lambda: False
""",
)

PROPS = """

# latency-test side directory: device properties are constant per process.
import functools as _lt_functools
import torch as _lt_torch

_lt_props = _lt_functools.cache(lambda: _lt_torch.cuda.get_device_properties(0))
"""
for rel, count in (
    ("target_info.py", 1),
    ("matmul_ogs_details/opt_flags.py", 2),
    ("matmul_ogs_details/opt_flags_details/opt_flags_nvidia.py", 2),
):
    replace(f"triton_kernels/{rel}", "torch.cuda.get_device_properties(0)", "_lt_props()", count)
    append(f"triton_kernels/{rel}", PROPS)

replace(
    "triton_kernels/matmul_ogs_details/opt_flags_details/opt_flags_nvidia.py",
    "    smem_capacity = device_props.shared_memory_per_block_optin\n",
    "    smem_capacity = _lt_smem_capacity()\n",
    1,
)
append(
    "triton_kernels/matmul_ogs_details/opt_flags_details/opt_flags_nvidia.py",
    """

# latency-test side directory: torch 2.5.1 has no shared_memory_per_block_optin;
# read the same opt-in per-block limit from the Triton driver, once.
@_lt_functools.cache
def _lt_smem_capacity():
    return getattr(_lt_props(), "shared_memory_per_block_optin", None) or (
        triton.runtime.driver.active.utils.get_device_properties(0)["max_shared_mem"]
    )
""",
)

append(
    "triton_kernels/__init__.py",
    """

# latency-test side directory: target checks ask the Triton driver for the
# current target on every call; it is constant per process.
import functools as _lt_functools
import triton as _lt_triton

_lt_driver = _lt_triton.runtime.driver.active
_lt_driver.get_current_target = _lt_functools.cache(_lt_driver.get_current_target)
""",
)
print("patched", side)
