# The PD check's reference: the short PD plan over NVLink

`src/pd_runner.py` on `src/pd_sanity_batches.json`, on `fairinf-matteo-fork-4xa100`
on 2026-11-03, with the setup of `results/gptoss-pd-nixl-20261103`: gpt-oss-20b
on fairinf-sglang main (`99f39f2b5`), the prefill server on GPU 0 and the
decode server on GPU 1, and NIXL with UCX copying the KV cache between them
over NVLink. `vm/README.md` has the exact commands. `sanity_pd.py` compares
every run with this one.

- `results.jsonl.gz`: 2,003 requests, one line each (see `pd_runner.py`),
  every one with both answers.
- `prefill.log.gz`, `decode.log.gz`: the two servers' logs.
- `run.log`: the setup, and the NVLink counters before and after the run.

`pd_runner.py` also wrote `requests.jsonl`, the same lines without the log
stages. It was left out: `results.jsonl` holds all of it.

The plan is 20 batches of `src/pd_batches.json` and one of its bursts, each
sent 3 times after the 4-batch warmup:
- one prompt alone, from 16 to 8,192 tokens: what moving the bytes costs;
- 2 to 64 prompts of 128 tokens and 128 or 256 of 16, and the 8,192-token
  budget split into 2, 4 or 8 prompts: the cost per request of a step;
- a long prompt with short ones: 6,144 + 31 x 16 and 2,048 + 15 x 256;
- 64 prompts of 1,024 tokens at once, where the transfer has to keep up with
  prefill.

It ran on 2026-11-03 from 06:01 to 06:06 UTC, right after
`results/gptoss-120b-tp4-20261103` on the same VM: 3.3 minutes for both
servers to start, then 1.1 minutes of sending.

## What it shows

From `python src/pd_fits.py results/gptoss-pd-sanity-nixl-20261103/results.jsonl.gz`:

- **One request alone** takes 3.5 ms at 16 tokens and 5.5 ms at 8,192, as in
  the full run (3.6 and 5.6).
- **All of the KV cache went over NVLink:** 12.40 GiB, all that the prefill
  server reports sending.
- **The burst kept up:** prefill produced 26,577 tokens/s and the transfer
  drained 28,456, against 26,501 and 30,458 in the full run.
- **Bootstrap** split 58% of the sent batches into more than one prefill step,
  the second starting 80 ms after the first (median), as in the full run.

## How well it represents the setup

Three runs of this setup on the same VM sent these 20 batches: this one, the
check right after it (`results/gptoss-sanity-pd-4xa100-20261103`) and the full
plan earlier the same day (`results/gptoss-pd-nixl-20261103`). Compared with
`sanity_pd.py`, each pair MATCHES: the same 36 setup lines, the same KV cache
per prompt, the burst kept up. Their transfer delays, mean difference over the
batches:

| this run | against | mean difference | batches beyond noise |
|---|---|---|---|
| the check | this one | -7.6% | 1 of 20 |
| the full plan | this one | -6.8% | 0 of 20 |
| the full plan | the check | +1.3% | 0 of 20 |

**This run came out about 7% slower than the other two** on batches of
several prompts. It was the first to start after the 120b sweep: its servers
took 196-202 s to start, against 111-112 s for the check's, and the decode
server picked requests up 2.6 ms after their KV arrived (median), against 0.9
ms in the full run. `sanity_pd.py` therefore lets the mean move by 15%, about
twice the largest of these shifts.
