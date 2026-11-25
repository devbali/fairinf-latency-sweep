# gpt-oss-120b with --tp 4 on four A100 80GB, on the FairInference experiments setup (2026-11-24)

The full plan (`src/batches.json`) again, timed by `src/runner.py` with the setup the FairInference experiments
serve with: the fork after its upstream sync (sglang source 66697235d3bb),
torch 2.14.1+cu130, sgl_kernel 0.4.10, flashinfer 0.7.2, transformers 5.19.0, and serve.sh's model flags
(`--dtype bfloat16 --sampling-backend pytorch`, `--tp 4`; the fork's default
CUDA graphs, decode full up to 512 and prefill "breakable"). runner.py takes this SGLang through its NEW_API path.
Four decode batches at the 120b Micro 1 operating point (8 and 32 requests of 1,300 tokens, 98 of 4,300, 150 of 2,800) and lone prefills of 1,000 and 5,000 tokens are added.
The run was resumed once after a spot preemption (runner.py skips batches already recorded).

Against `gptoss-120b-tp4-20261103`, the same batches on the older setup (fork 99f39f2b5, torch 2.11): decode
+55% on average, prefill -16%, hybrid -31% (`src/compare_runs.py`). The models fitted here are
`export_profile.py`'s `gpt-oss-120b-tp4-final`.
