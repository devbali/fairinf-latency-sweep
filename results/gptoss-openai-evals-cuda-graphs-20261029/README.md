# gpt-oss-20b on GPQA Diamond with OpenAI's eval script, 8 attempts

OpenAI's own eval script for gpt-oss, `python -m gpt_oss.evals` from
github.com/openai/gpt-oss at `2770201`, run unchanged against the SGLang
server of `results/gptoss-cuda-graphs-20261029`'s setup, on eu1, 2026-10-29.
An earlier version of `sanity_accuracy.py` ran it as

```bash
gpt-oss-evals-venv/bin/python sanity_accuracy.py "TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
    PYTHONPATH=.../fairinf-sglang-4e84427ab/python:.../vm CUDA_VISIBLE_DEVICES=0 .../venv/bin/python \
    -m sglang.launch_server --model-path .../gpt-oss-20b-bf16/snapshots/... --dtype bfloat16 \
    --mem-fraction-static 0.7 --sampling-backend pytorch \
    --scheduling-policy-path noop_cache_policy.NoOpWithCachePolicy --port 30000 \
    --piecewise-cuda-graph-compiler inductor" --evals gpqa:low,aime25:high
```

which started that server with `--reasoning-parser gpt-oss
--allow-auto-truncate` added, then ran

```bash
python -m gpt_oss.evals --base-url http://127.0.0.1:30000/v1 --eval gpqa \
    --sampler chat_completions --model .../gpt-oss-20b-bf16/snapshots/... --reasoning-effort low
```

`gpt-oss-evals-venv` held gpt-oss's eval packages as released: openai 3.41.0,
pandas 2.3.3, numpy 2.2.6, Jinja2 3.1.6, tqdm 4.70.1 and blobfile 3.3.0, on
Python 3.10.12.

| | |
|---|---|
| GPQA Diamond, low effort | **59.0%** (198 questions x 8 attempts) |
| OpenAI's (model card, Table 3) | 56.8% |
| lowest that passes | 46.2% |
| rejected requests | 0 |
| empty answers, asked again | 4 |
| time | 291 s to start the server, 3.3 minutes for the 1,584 answers |

- **The setup runs the model right.** It scores 2.2 points above OpenAI's
  number. 51 questions were answered right all 8 times, 13 never, and 134 in
  between.
- **A one-attempt check would pass too.** Drawing one of the 8 answers per
  question, 20,000 times over, gives 53.5% to 63.1% (1st to 95th percentile),
  never below 46.2%. So `sanity_accuracy.py` now asks once by default.
- **AIME 2025 at high effort,** the example in OpenAI's guide, was also
  queued. It was stopped as it started, at the user's request: the check
  only needs to show the model runs right, and AIME at high effort writes
  over 20k tokens per answer.

Files:
- `server.log`: the server's output.
- `gpqa-low.log`: the eval script's output.
- `gpqa-low.json`: its metrics.
- `gpqa-low-answers.json.gz`: every question as asked, the final answer, and
  its grade (the script's "all results" file).
- `scores.json`: the score against OpenAI's. Its format is that earlier
  version's.
