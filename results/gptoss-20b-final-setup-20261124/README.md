# gpt-oss-20b on one A100 80GB, on the FairInference experiments setup (2026-11-24)

The full plan (`src/batches.json`) again, timed by `src/runner.py` with the setup the FairInference experiments
serve with: the fork after its upstream sync (sglang source 66697235d3bb),
torch 2.14.1+cu130, sgl_kernel 0.4.10, flashinfer 0.7.2, transformers 5.19.0, and serve.sh's model flags
(`--dtype bfloat16 --sampling-backend pytorch`, `--mem-fraction-static 0.7`; the fork's default
CUDA graphs, decode full up to 512 and prefill "breakable"). runner.py takes this SGLang through its NEW_API path.
Two decode batches of 8 and 32 requests of 1,300 tokens and lone prefills of 1,000 and 5,000 tokens are added.
The run was resumed once after a spot preemption (runner.py skips batches already recorded).

Against `gptoss-cuda-graphs-20261029`, the same batches on the older setup (fork 4e84427ab, torch 2.11): decode
+29% on average, prefill -19%, hybrid -38% (`src/compare_runs.py`). The models fitted here are
`export_profile.py`'s `gpt-oss-20b-a100-final`.
