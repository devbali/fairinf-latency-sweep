# The PD check against its reference: a match

`python sanity_pd.py "$PREFILL" "$DECODE"` on `fairinf-matteo-fork-4xa100` on
2026-11-03, from 06:06 to 06:09 UTC, with the NIXL commands of `vm/README.md`.
It ran right after its reference, `results/gptoss-pd-sanity-nixl-20261103`,
with the same commands on the same VM.

- `output.txt`: what `sanity_pd.py` printed, the comparison included.
- `results.jsonl.gz`: 2,003 requests, one line each (see `pd_runner.py`),
  every one with both answers.
- `prefill.log.gz`, `decode.log.gz`: the two servers' logs.
- `run.log`: the setup, and the NVLink counters before and after the run.

`pd_runner.py` also wrote `requests.jsonl`, the same lines without the log
stages. It was left out: `results.jsonl` holds all of it.

## The verdict: MATCHES

- **Setup:** all 36 setup lines are the same.
- **KV cache:** every prompt length carries the same MB, from 0.750 for 16
  tokens to 194.980 for 8,192.
- **Transfer delays:** -7.6% on average over the 20 batches (mean absolute
  8.8%). One is beyond the noise: two prompts of 4,096 tokens, 4.4 ms
  against 6.0. Single prompts are within 7%.
- **The burst kept up:** prefill produced 26,725 tokens/s and the transfer
  drained 29,259, against 26,577 and 28,456 in the reference.
- **Not part of the verdict:** all 12.40 GiB of KV went over NVLink. The
  bootstrap took 98 ms before prefill (median, against 100) and the decode
  server picked requests up 2.4 ms after their KV arrived (against 2.6).

The check ran with the mean limit at 10%. It is 15% now, because this run and
the full plan both came out about 7% faster than the reference. The
reference's README has that comparison.
