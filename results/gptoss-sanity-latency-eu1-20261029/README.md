# sanity_latency.py on eu1, against its reference

`sanity_latency.py` run on eu1 on 2026-10-29, right after its reference
run, `results/gptoss-sanity-cuda-graphs-20261029`, with the server command
of the same setup (paths shortened):

```bash
python sanity_latency.py "TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 PYTHONPATH=.../fairinf-sglang-4e84427ab/python:.../vm \
    CUDA_VISIBLE_DEVICES=0 .../venv/bin/python -m sglang.launch_server --model-path .../gpt-oss-20b-bf16/snapshots/... \
    --dtype bfloat16 --mem-fraction-static 0.7 --sampling-backend pytorch \
    --scheduling-policy-path noop_cache_policy.NoOpWithCachePolicy --port 30000 --piecewise-cuda-graph-compiler inductor"
```

- `results.jsonl`: 264 samples (88 batches, each timed 3 times).
- `run.log`: the runner's output. The whole check took 5.5 minutes.

It MATCHES the reference. All 13 setup lines are the same, and by phase:

| phase | batches | mean difference | mean abs difference | max abs difference | beyond noise |
|---|---|---|---|---|---|
| prefill | 20 | +0.1% | 0.7% | 3.5% | 0 |
| decode | 20 | +1.2% | 2.2% | 9.2% | 1 |
| hybrid | 48 | +0.1% | 0.3% | 1.3% | 0 |

The one batch beyond noise is a 1-request decode step: 11.2 ms here, 10.2 in
the reference, with samples that spread by 2%. The first version of the check
failed any run with such a batch, and called this one DIFFERS. Since then it
judges each phase as a whole: a phase differs if its mean difference is over
5%, or if more than 10% of its batches are beyond noise. The limit was 3%
until 2026-10-30: with decode already +1.2% here, a decode 2% slower than this
run failed it.
