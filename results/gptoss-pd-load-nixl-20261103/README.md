# PD transfers under load: idle, prefill, decode, both

`src/pd_load.py` on `src/pd_load.json`, on `fairinf-matteo-fork-4xa100` on
2026-11-03, with the setup of `results/gptoss-pd-nixl-20261103`: gpt-oss-20b on
fairinf-sglang main (`99f39f2b5`), the prefill server on GPU 0 and the decode
server on GPU 1, and NIXL with UCX copying the KV cache between them over
NVLink. `vm/README.md` has the exact commands. The question: what a KV
transfer's time depends on besides its batch, and whether it is bimodal.

- `results.jsonl.gz`: 3,718 requests, one line each (see `pd_runner.py`), with
  `kind` "probe", "load" or "warmup". Every probe got both answers.
- `prefill.log.gz`, `decode.log.gz`: the two servers' logs.
- `run.log`: the setup, and the NVLink counters before and after the run.
- `profiles/`: py-spy's raw profiles of both schedulers, one per scenario run
  with `--profile`, and py-spy's own output (`.txt.log`: samples, errors).

`pd_load.py` also wrote `requests.jsonl`, the same lines without the log
stages. It was left out: `results.jsonl` holds all of it.

## The plan

Six probe batches go out one at a time, round-robin, 0.5 s after the last
one's answers, each 8 times, with 2 output tokens per request: one prompt of
128, 1,024 or 4,096 tokens, 4 or 16 of 128, and 8 of 1,024. Four scenarios
send them, each keeping a different server busy with a load of clients that
each send one request at a time until the scenario ends:

| scenario | load | what it keeps busy |
|---|---|---|
| idle | none | nothing |
| prefill | 2 clients, 2,048-token prompts, 1 output token | a prefill step is nearly always queued; the decode server only receives KV |
| decode | 32 clients, 256-token prompts, 6,000 output tokens, aborted at the end | the decode server runs decode steps of 32 requests, with its memory 24-30% full |
| both | both loads | both |

Then each scenario ran again with py-spy sampling both scheduler processes
(`--rate 250 --idle --threads --nonblocking`). It ran from 22:32 to 22:43 UTC:
3.7 minutes for the servers to start, then 6.3 minutes of scenarios.

Each prefill step of a probe batch gets its transfer time T, from its first
end to the decode server finding its last KV, in two parts: E, the posting span
(first end to last end), and the tail D (last end to last KV). Two states are
marked: **next_prefill**, another prefill step launched before the step's last
end, so its results were processed while that step ran; and **decode_busy**,
another request in the decode server's running batch while the KV landed.

## What it shows

From `python src/pd_load_report.py results/gptoss-pd-load-nixl-20261103`,
passes without the profiler:

| scenario | batches split into 2+ steps | T median (ms) | T / selected model | posting per extra request (ms) | tail D (ms) |
|---|---|---|---|---|---|
| idle | 50% | 4.1 | 0.44 | 1.97 | 3.9 |
| prefill | 0% | 9.9 | 0.76 | 3.29 | 5.0 |
| decode | 50% | 15.1 | 1.13 | 1.96 | 13.2 |
| both | 6% | 17.6 | 1.06 | 3.14 | 9.7 |

By state, all four scenarios pooled:

| next_prefill | decode_busy | posting per extra request (ms) | tail D (ms), median [p10-p90] | D under 6 ms |
|---|---|---|---|---|
| no | no | 1.92 | 3.8 [3.0-9.6] | 65% |
| no | yes | 1.95 | 10.7 [6.1-17.2] | 10% |
| yes | no | 3.38 | 4.9 [3.3-9.7] | 67% |
| yes | yes | 3.41 | 11.7 [6.2-18.3] | 8% |

- **Each state moves one part of T, and the two add up.** A next prefill step
  running slows posting from 1.9 to 3.4 ms per request; a busy decode server
  adds about 7 ms to the tail.
- **One request alone, with nothing else running, takes 3.3 to 4.0 ms** at
  128 to 4,096 tokens, p10 to p90 within 0.3 ms. The selected model says 9.2
  to 10.0 ms: it was fitted on the plan's batches, where these states mixed.
- **A busy decode server spreads the tail evenly over one decode step, not in
  two humps.** One-request steps under the decode load: D 12.4 ms median,
  5.8 to 18.4 (p10 to p90), against 3.5 when idle; a decode step of 32
  requests took 15.6 ms. In the idle scenario, the same steps fall in groups:
  40 of 48 at 3.0 to 5.1 ms; 7 at 7.1 to 9.7 ms, each the first step of a
  split 8 x 1024 batch, posted while the step with the batch's other 7
  requests ran; and one at 18.4 ms that landed while the decode server ran
  its batch's other step. Mixes like these make the full plan's small steps
  bimodal.
- **Under the prefill load, no probe batch was split.** It waited behind the
  load's step, so all its requests ran in one step and posted one after
  another at 3.4 ms each: 16 x 128 took 38.3 ms against 19.0 when idle, where
  half the batches split and the halves overlapped.
- **A step's KVs land together, after its last post.** In steps of 4 or more
  requests, the KVs arrived within 0.4 to 5.4 ms of each other (median by
  scenario), against posting spans of 20 to 24 ms. The decode server found a
  KV 2.7 to 3.3 ms after the prefill scheduler's poll first saw its transfer
  done (the latency the prefill server logs, `transfer_total` /
  `transfer_speed`), 6.5 ms under both loads and 11.8 ms under the decode
  load.

## Why, from the profiles

- **The decode server checks for arrived KV once per decode step.** Under the
  decode load, its scheduler thread spent 91% of its samples in `run_batch`,
  84% on one line: `triton_backend.py:1519` in
  `update_sliding_window_buffer_cuda_graph`, which slices a tensor by a length
  still on the GPU. Reading that length waits for the GPU, including the
  decode step running, so the loop reaches `pop_transferred` only between
  steps.
- **Posting takes the prefill scheduler 2.3 to 3.7 ms per request.** Per
  request posted (ms): NIXL building and posting its two transfers
  (`add_transfer_request`: the full-attention KV and the sliding-window state)
  1.3 to 1.9, copying the KV indices to the CPU 0.3 to 0.7, the radix cache
  insert 0.3 to 0.6, stats 0.2 to 0.4. Each part takes longer in the
  scenarios with the prefill load, where the next step is usually running;
  the index copies grow the most, from 0.29-0.34 to 0.55-0.72.
- **Most of processing a step's results is waiting for the step itself**
  (`copy_done.synchronize`, 8.6 to 11.2 ms of samples per request posted).
  That comes before the requests' ends, so it is not part of T.
- **An idle prefill server spends 74% of its loop in a radix cache check**
  (`on_idle`, `swa_radix_cache.sanity_check`). Under the prefill load, 11.5%
  sits on `triton_backend.py:1464`, the same kind of GPU read in the prefill
  forward's sliding-window setup: it sizes a tensor by a length on the GPU.

## How much the profiler changed

T median by probe without / with py-spy moved by 0 to 1 ms for most
one-request probes, 1.4 to 3.7 ms for four of the twelve, and by up to 7 ms either
way for the batches of several requests, with 8 samples each. The profiled
passes are in `results.jsonl.gz` as `-profile` tags; the tables above leave
them out.
