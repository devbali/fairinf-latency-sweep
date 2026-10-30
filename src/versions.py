"""Print the setup lines a run's log starts with: package versions and the GPU.

vm/run_clean.sh and sanity_latency.py run this with the run's own Python, and
compare_runs.py compares these lines between two runs. They hold nothing
specific to one machine: SGLang shows as its version and a hash of its Python
source, so the same code matches wherever it is installed.
"""

import hashlib
import importlib
import os


def source_hash(package):
    """A short hash of every .py file under the package, with its path inside the package."""
    root = os.path.dirname(package.__file__)
    digest = hashlib.sha256()
    for directory, subdirectories, files in os.walk(root):
        subdirectories.sort()
        for name in sorted(files):
            if name.endswith(".py"):
                path = os.path.join(directory, name)
                digest.update(os.path.relpath(path, root).encode())
                with open(path, "rb") as f:
                    digest.update(f.read())
    return digest.hexdigest()[:12]


for name, module, get in [
    ("sglang", "sglang", lambda m: f"{m.__version__}, source {source_hash(m)}"),
    ("sgl_kernel", "sgl_kernel", lambda m: m.__version__),
    ("flashinfer", "flashinfer", lambda m: m.__version__),
    ("transformers", "transformers", lambda m: m.__version__),
    ("torch", "torch", lambda m: f"{m.__version__} cuda {m.version.cuda}"),
    ("gpu", "torch", lambda m: m.cuda.get_device_name(0)),
]:
    try:
        value = get(importlib.import_module(module))
    except Exception as error:
        value = f"unavailable: {error}"
    print(f"{name:<12} {value}", flush=True)
