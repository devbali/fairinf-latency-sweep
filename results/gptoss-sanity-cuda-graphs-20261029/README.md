# gpt-oss-20b on one A100 with CUDA graphs: the sanity reference

The run `sanity_latency.py` compares with: the 88 batches of
`src/sanity_batches.json`, timed on eu1 in the setup of
`results/gptoss-cuda-graphs-20261029`, with piecewise CUDA graphs.

- `results.jsonl`: 264 samples (88 batches, each timed 3 times).
- `run.log`: the runner's output. It starts with the setup lines
  `src/compare_runs.py` compares, from `src/versions.py`.

## The run

| | |
|---|---|
| date | 2026-10-29, 18:54 UTC; 6 minutes in all, most of it loading the model and capturing graphs |
| VM | eu1 (`fairinf-matteo-fork-1xa100-eu1`), one A100-SXM4-80GB |
| SGLang | fairinf-sglang `4e84427ab`, source hash `062248804ecc` |
| packages | sgl_kernel 0.4.2.post1, flashinfer 0.6.8.post1, transformers 5.6.0, torch 2.11.0+cu130, in the fresh venv |
| command | `PIECEWISE=inductor bash vm/run_clean.sh src/sanity_batches.json OUT` |
| flags | `--dtype bfloat16 --mem-fraction-static 0.7 --sampling-backend pytorch --piecewise-cuda-graph-compiler inductor`, with `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` |

Steps take 19.6-552 ms (prefill), 10.0-32 ms (decode) and 35-3,614 ms
(hybrid). They agree with the same batches in the full sweep an hour
earlier: the mean differences are -0.5% (prefill), -0.9% (decode) and
+0.1% (hybrid).

## The check against it

`results/gptoss-sanity-latency-eu1-20261029` is `sanity_latency.py` run
right after, on the same VM, with the server command of this setup. It
MATCHES: all 13 setup lines are the same, and the phases' mean differences
are +0.1%, +1.2% and +0.1%.
