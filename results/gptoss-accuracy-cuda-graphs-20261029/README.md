# gpt-oss-20b accuracy on eu1, with CUDA graphs

`sanity_accuracy.py` run on eu1 on 2026-10-29, with the server command of
`results/gptoss-cuda-graphs-20261029`'s setup (paths shortened). This was its
first version, which ran SGLang's eval script, `sglang.test.run_eval`; it now
runs OpenAI's eval code instead.

```bash
python sanity_accuracy.py "TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 PYTHONPATH=.../fairinf-sglang-4e84427ab/python:.../vm \
    CUDA_VISIBLE_DEVICES=0 .../venv/bin/python -m sglang.launch_server --model-path .../gpt-oss-20b-bf16/snapshots/... \
    --dtype bfloat16 --mem-fraction-static 0.7 --sampling-backend pytorch \
    --scheduling-policy-path noop_cache_policy.NoOpWithCachePolicy --port 30000 --piecewise-cuda-graph-compiler inductor"
```

- `server.log`: the server's output. The check added `--reasoning-parser gpt-oss`.
- `mmlu.log`, `gpqa.log`: the output of `sglang.test.run_eval` for each eval.
- `scores.json`: the scores, OpenAI's, and the lowest that passes.

`results/gptoss-accuracy-clean-20261029` is the same check without CUDA
graphs. Both at low reasoning effort, temperature 1:

| benchmark | questions | CUDA graphs | no CUDA graphs | OpenAI (model card) | lowest that passes |
|---|---|---|---|---|---|
| MMLU | 2,000 | 68.4% | 68.0% | 80.4% | 75.4% |
| GPQA Diamond | 198 | 36.4% | 33.3% | 56.8% | 46.2% |

- **CUDA graphs don't change the scores.** The two setups differ by 0.4 and
  3.1 points, within sampling noise (1.0 and 3.4 points, one standard error).
- **Both are far above chance (25%), but well below OpenAI's,** so both
  checks reported BELOW.
- **SGLang's own eval script doesn't reach OpenAI's numbers either.** SGLang
  tests gpt-oss-20b with this same script and GPQA
  (`test/registered/core/test_gpt_oss_1gpu.py` at `4e84427ab`). It expects
  at least 34% at low effort, for the BF16 and MXFP4 checkpoints alike,
  against OpenAI's 56.8%. Its settings differ a little: temperature 0.1, at
  most 4,096 tokens, and no reasoning parser.
- **OpenAI's own eval script puts the gap on SGLang's script.** On the same
  server command, `python -m gpt_oss.evals` from github.com/openai/gpt-oss
  scored GPQA Diamond 59.0% at low effort, 8 attempts per question, against
  OpenAI's 56.8% (`results/gptoss-openai-evals-cuda-graphs-20261029`). So the
  setup runs the model right. SGLang's script asks for a last line `Answer:
  $LETTER` and counts only the first `Answer: X` in the response; OpenAI's
  asks for the option's letter and accepts it in many forms. Which of the two
  differences caused the gap was not checked.
- **Graphs made serving faster.** The 2,000 MMLU questions took 100 s with
  graphs and 161 s without (3,571 against 2,227 output tokens/s); GPQA took
  48 s against 60. The server took 256 s to start with graphs, 126 without.
