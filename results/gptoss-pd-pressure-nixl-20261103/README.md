# PD transfers under memory pressure

`src/pd_load.py`'s pressure scenario, on `fairinf-matteo-fork-4xa100` on
2026-11-03, right after `results/gptoss-pd-load-nixl-20261103` and with its
setup: gpt-oss-20b on fairinf-sglang main (`99f39f2b5`), prefill on GPU 0,
decode on GPU 1, NIXL over NVLink. The one change is that the decode server's
KV cache is capped with `--max-total-tokens 40000`: it holds 40,000
full-attention tokens and 32,000 sliding-window tokens (its log). `vm/README.md`
has the exact commands.

- `results.jsonl.gz`: 516 requests, one line each (see `pd_runner.py`): the
  warmup, then for each pass 188 load requests and 8 probe batches (33
  requests). Every probe got both answers; the 63 load requests still running
  at the end of a pass were aborted, as planned.
- `prefill.log.gz`, `decode.log.gz`: the two servers' logs.
- `run.log`: the setup, and the NVLink counters before and after the run.
- `profiles/`: py-spy's raw profiles of both schedulers in the second pass,
  and py-spy's own output (`.txt.log`).

## The plan

The load is 32 clients, each sending one request at a time of 1,024 prompt
tokens and 1,500 output tokens: at their longest, 32 requests hold 80,768
tokens, twice the cache. The six probe batches of the load run go out under it,
round-robin, 4 times each or until 180 s are up: they got 8 batches out each
pass. The scenario ran once plain and once with py-spy on both schedulers, from
22:42 to 22:52 UTC: 2.1 minutes for the servers to start, then two passes of
about 3.4 minutes.

## What SGLang does under memory pressure

In PD mode (fork `99f39f2b5`):
- **The decode server admits a request only when it can reserve its KV**
  (`disaggregation/decode.py`, `pop_preallocated` and `_allocatable_tokens`):
  its prompt plus `num_reserved_decode_tokens` (512) must fit in the free
  tokens. Those exclude a reserve: the larger of 512 for every request the
  server holds and room for one running request to finish if all the others
  were retracted. The requests already retracted count against it too.
- **The prefill server waits for that before it prefills.** A request sits in
  its bootstrap queue until the decode server has allocated room and sent the
  metadata. So under pressure a request waits before its prefill, not with
  its KV in flight.
- **When running decodes outgrow the cache, some are retracted**
  (`managers/scheduler.py`, `check_decode_mem` and `retract_decode`). In decode
  mode a retracted request's KV goes to CPU memory (`offload_kv_cache`) and the
  request to the retracted queue. `resume_retracted_reqs` brings it back once
  there is room for another 512 decode steps.
- **While any request is retracted, the decode server admits nothing and
  doesn't check for arrived KV** (`process_decode_queue`, `decode.py:1516`).

## What it shows

From `python src/pd_load_report.py results/gptoss-pd-pressure-nixl-20261103`:

- **The cache stayed full.** Its usage reached 1.00, 18 to 19 requests ran
  (median), 15 waited for room on average (up to 26 and 32), and up to 10 sat
  retracted. SGLang retracted 59 times, one request each; 29 and 30 load
  requests were retracted at least once, and no probe was.
- **The wait came before the transfer.** The probes waited 25.5 and 23.7 s
  (median, by pass) in the prefill server's bootstrap queue, the same time as
  the decode server took to admit them (25.4 and 23.6 s).
- **The transfer itself then took 117 ms (median; p90 247) against 3 to 4 ms
  without pressure,** with a tail D of 61 ms and 7.2 ms of posting per extra
  request (passes without the profiler).

Why the transfer grew:
- **Admission comes in bursts,** whenever memory frees up, and the prefill
  server runs a burst as one step. Probe `pressure-probe-00-0-000`, one prompt
  of 128 tokens, shared its prefill step with 8 load requests of 1,024 tokens.
- **A step's KVs land only after its posting is done.** That step's posts
  spread over 202 ms. Seven of its 9 KVs, the probe's among them, landed within
  0.5 ms of each other, 5 ms after the step's last post, together with a
  request from the step before, posted 531 ms earlier. The two posted last
  landed 12 ms later.
  The decode server found each probe's KV 5.2 ms (median) after the prefill
  server's poll saw its transfer done.
- **Posting got slow.** NIXL building and posting the transfers
  (`add_transfer_request`) took 14.2 ms of the prefill scheduler per request,
  against 1.3 to 1.9 in the load run, almost all in NIXL's `transfer()` call
  for the full-attention KV. This run doesn't show why. A guess: with the
  cache full, a request's slots on the decode server are scattered, so its KV
  goes in many small pieces.

From the profiles:
- **The prefill scheduler spent 75% of its loop in `pop_bootstrapped`,**
  polling the requests waiting for the decode server: mostly
  `torch.distributed` all-reduces in `poll_and_all_reduce`.
- **The decode scheduler spent 92.5% in `run_batch`,** 84% on
  `triton_backend.py:1519`, the GPU read that makes the decode loop check for
  KV once per step (see the load run's README). A decode step of 18 requests
  took 16.0 ms.

With 1 or 2 steps per probe in each pass, the two passes' transfer times
differ widely by probe; the report shows both.
