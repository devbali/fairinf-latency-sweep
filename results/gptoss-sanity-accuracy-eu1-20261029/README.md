# Accuracy check on eu1, with CUDA graphs

`sanity_accuracy.py` with its defaults, run on eu1 on 2026-10-29 with the
server command of `results/gptoss-cuda-graphs-20261029`'s setup (paths
shortened):

```bash
gpt-oss-evals-venv/bin/python sanity_accuracy.py "TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
    PYTHONPATH=.../fairinf-sglang-4e84427ab/python:.../vm CUDA_VISIBLE_DEVICES=0 .../venv/bin/python \
    -m sglang.launch_server --model-path .../gpt-oss-20b-bf16/snapshots/... --dtype bfloat16 \
    --mem-fraction-static 0.7 --sampling-backend pytorch \
    --scheduling-policy-path noop_cache_policy.NoOpWithCachePolicy --port 30000 \
    --piecewise-cuda-graph-compiler inductor"
```

It added `--reasoning-parser gpt-oss --allow-auto-truncate` to the server, then
asked the 198 GPQA Diamond questions once each at low reasoning effort, with
OpenAI's eval code at `2770201`. `gpt-oss-evals-venv` is the venv of
`results/gptoss-openai-evals-cuda-graphs-20261029`.

| | |
|---|---|
| GPQA Diamond, low effort | **54.5%** (198 questions x 1) |
| OpenAI's (model card, Table 3) | 56.8% |
| lowest that passes | 46.2% |
| verdict | MATCHES (exit code 0) |
| rejected requests, empty answers | 0, 0 |
| time | 291 s to start the server, 0.7 minutes for the questions, 5m33s in all |

- **54.5% is on the low side of what one attempt gives,** about the 3rd
  percentile of the spread drawn from the 8-attempt run (59.0%), but 8 points
  above the pass line. One answer per question carries about 2.5 points of
  sampling noise.
- **No process was left on the GPU** after the check; it now waits for every
  server process to exit before it reports.

Files:
- `server.log`: the server's output.
- `gpqa-low.log`: the eval's output.
- `gpqa-low.json`: its metrics.
- `gpqa-low-answers.json.gz`: every question as asked, the final answer, and
  its grade.
- `scores.json`: the score against OpenAI's.
