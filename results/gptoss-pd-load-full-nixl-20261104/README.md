# The PD transfer model under each load

`src/pd_load.py` with the PD plan's batches as its probes, on
`fairinf-matteo-fork-4xa100` on 2026-11-04, with the setup of
`results/gptoss-pd-nixl-20261103`: gpt-oss-20b on fairinf-sglang main
(`99f39f2b5`), prefill on GPU 0, decode on GPU 1, NIXL over NVLink.
`vm/README.md` has the exact commands. The question: what the four transfer
models look like, and how well they predict, when each is fitted on one load
alone.

- `results.jsonl.gz`: 78,458 requests, one line each (see `pd_runner.py`):
  15,834 probe requests per scenario, 15,048 load requests and the warmup.
  Every probe got both answers. The 32 decode-load requests still running at
  the end of a scenario were aborted, as planned.
- `prefill.log.gz`, `decode.log.gz`: the two servers' logs.
- `run.log`: the setup, and the NVLink counters before and after the run.

`pd_load.py` also wrote `requests.jsonl`, the same lines without the log
stages. It was left out: `results.jsonl` holds all of it.

## The plan

The 334 batches of `src/pd_batches.json`, without its bursts, went out 3 times
each, round-robin, 0.5 s apart, with 2 output tokens per request, as
`pd_runner.py` sends them. They went out under each load of
`src/pd_load.json` in turn:

| scenario | load |
|---|---|
| idle | none |
| prefill | 2 clients, 2,048-token prompts, 1 output token |
| decode | 32 clients, 256-token prompts, 6,000 output tokens |
| both | both loads |

A decode-load request ends after 6,000 tokens and its client sends the next,
so the decode server ran 32 requests (median) throughout, with a short
prefill now and then. Its cache stayed under 54% full, and a decode step took
17 ms. It ran from 00:29 to 01:35 UTC: 3.6 minutes for the servers to
start, then 14.3, 16.4, 14.7 and 16.7 minutes for the four scenarios.

## What it shows

From `python src/compare_fits.py results/gptoss-pd-load-full-nixl-20261104/results.jsonl.gz --pd-phase SCENARIO`,
the four regressions fitted on each scenario alone, with errors on held-out
sets of prompt lengths (5-fold), as for `results/gptoss-pd-nixl-20261103`:

| scenario | steps | sets of prompt lengths | linear | degree 2 | degree 2 + sum l^2 | selected | held-out R^2, selected | RMSE, selected |
|---|---|---|---|---|---|---|---|---|
| idle | 1,690 | 926 | 27.1% | 27.4% | 27.5% | 27.1% | 0.989 | 5.1 ms |
| prefill | 1,134 | 425 | 43.0% | 50.9% | 50.9% | 34.5% | 0.865 | 30.0 ms |
| decode | 1,816 | 661 | 19.3% | 19.7% | 19.7% | 19.3% | 0.983 | 6.4 ms |
| both | 1,167 | 537 | 31.7% | 35.8% | 35.7% | 31.3% | 0.759 | 34.7 ms |

The selected models, in ms, with S and M in tokens:

| scenario | selected model |
|---|---|
| idle | 6.74 + 2.025 N + 0.00069 S - 0.00095 M |
| prefill | 1.08 + 2.576 N + 1.06e-6 S^2 - 1.03e-6 sum l^2 |
| decode | 12.21 + 2.132 N + 0.00201 S - 0.00153 M |
| both | 17.74 + 2.435 N + 0.83e-6 S^2 - 0.70e-6 sum l^2 |

Each scenario's selected model, fitted on all its sets, on the others' sets
(mean error; the diagonal is in-sample):

| model from | on idle | on prefill | on decode | on both |
|---|---|---|---|---|
| idle | 27.0% | 29.5% | 31.6% | 37.8% |
| prefill | 41.3% | 28.7% | 45.7% | 35.6% |
| decode | 75.2% | 59.3% | 19.1% | 21.9% |
| both | 137.5% | 107.1% | 46.1% | 30.4% |

- **Idle reproduces the plan's model:** 6.74 + 2.03 N ms against 7.92 + 2.11
  N, and 27.1% held-out error against 28.6%, again mostly from steps of one
  prompt (66.7%, against 13.2% for equal prompts).
- **A busy decode server adds 5.5 ms to the constant and leaves the cost per
  request alone** (2.13 ms). Its relative error is the lowest, 19.3%, because
  the constant makes small steps less small, but the same batch's samples
  spread twice as far: 38% (median) against 18% idle.
- **The prefill load is the hardest to model.** Each request adds 2.6 ms
  instead of 2.0, and only 13% of the batches split into two steps, against
  68% idle. Transfers can wait for the load's running prefill step: one
  8,192-token prompt took 97 ms (median) against 8 ms idle and 14 ms under the
  decode load. The largest misses are steps of about 6,000 tokens split
  among 2 to 25 prompts, which waited like that (112 to 217 ms, against 20 to
  102 predicted; the one of 2 prompts took 13 ms idle and 21 ms under the
  decode load), and steps of 145 to 193 prompts of 32 tokens, which took 625
  to 700 ms against 370 to 530. The selected models' S^2 - sum l^2 term, near 0 for one prompt,
  follows them, and overshoots one 32,768-token prompt: 201 and 486 ms
  predicted, 15 and 114 measured.
- **The degree-2 models never help**, and are worse under the prefill load.
- **One model per scenario matters most between quiet and busy decoding.**
  The decode models are off by 75% and 138% on idle; the idle model is off by
  30% to 38% on the loaded scenarios.

`python src/pd_load_report.py results/gptoss-pd-load-full-nixl-20261104`
splits the steps by state, as in `results/gptoss-pd-load-nixl-20261103`, all
four scenarios pooled:

| next_prefill | decode_busy | posting per extra request (ms) | tail D (ms), median [p10-p90] |
|---|---|---|---|
| no | no | 2.19 | 7.3 [3.5-12.3] |
| no | yes | 2.41 | 13.3 [6.4-21.6] |
| yes | no | 3.43 | 5.8 [3.6-11.3] |
| yes | yes | 3.60 | 14.9 [7.7-23.1] |
