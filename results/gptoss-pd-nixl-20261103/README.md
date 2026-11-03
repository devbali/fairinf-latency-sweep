# SGLang PD disaggregation over NVLink: the NIXL backend

`src/pd_runner.py` on `fairinf-matteo-fork-4xa100` on 2026-11-03, with
SGLang's other built-in transfer backend, `--disaggregation-transfer-backend
nixl`. UCX moves the KV cache from the prefill server's GPU to the decode
server's over NVLink. Everything else is as in
`results/gptoss-pd-mooncake-tcp-20261103`: the gpt-oss server command of
`vm/README.md` on fairinf-sglang main (`99f39f2b5`), with PD mode, the backend
and a port added, laid out as the fork's single-node PD recipe
(`docs/advanced_features/pd_disaggregation.md`): every GPU visible, prefill on
GPU 0, decode on `--base-gpu-id 1`. `vm/README.md` has the exact commands, and
`run.log` the setup.

The whole plan ran, and every request got both answers: 23,564 requests in 18
minutes. These are the 334 batches of `src/pd_batches.json` and its 10 bursts,
each sent 3 times, after a 4-batch warmup.

- `results.jsonl.gz`: one line per request (see `pd_runner.py`).
- `prefill.log.gz`, `decode.log.gz`: the two servers' logs.
- `run.log`: the setup, and the NVLink counters before and after the run.

`pd_runner.py` also wrote `requests.jsonl`, the same lines without the log
stages. It was left out: `results.jsonl` holds all of it.

The transfer model below comes from `python src/compare_fits.py
results/gptoss-pd-nixl-20261103/results.jsonl.gz`, every other number from
`python src/pd_fits.py` on the same file.

## Setup

- **NIXL:** `nixl-cu13` 1.0.1, NIXL for CUDA 13 with its own UCX, and `nixl`
  1.0.1, the module SGLang imports. 1.0.1 was the current release when the
  fork branched from SGLang (2026-05-06), and the fork's CI installs the
  current one. `nixl` was installed without its dependency `nixl-cu12`, a
  second build that would overwrite some of the same files. SGLang's default
  NIXL backend is UCX.
- **All of the KV cache went over NVLink.** During the run GPU 0 sent 161.06
  GiB over its NVLinks and GPU 1 received 161.06 GiB, against the 161.08 GiB
  of KV the prefill server reports sending.
- **The rest is as in the mooncake run.** Prefill runs eagerly, since PD mode
  turns piecewise CUDA graphs off. `pd_runner.py` added `--enable-metrics`,
  `--skip-tokenizer-init` and `--enable-request-time-stats-logging`, for its
  timings.

## What delays a request

From the end of its prefill step (its result processed) to the decode server
finding its KV cache in place:

**1. Moving the bytes costs next to nothing.** One request alone takes
3.6 ms at 16 tokens and 5.6 ms at 8,192 (195 MB). Median of 3 samples:

| prompt tokens | 16 | 128 | 512 | 1,024 | 2,048 | 4,096 | 8,192 |
|---|---|---|---|---|---|---|---|
| delay (ms) | 3.6 | 3.7 | 3.9 | 3.9 | 4.1 | 4.6 | 5.6 |

- The 192 MB between 16 and 8,192 tokens add 2 ms, about 100 GB/s.
- The 29 irregular lengths of the random batches took 3.6 to 10.6 ms
  (median 5.1), up to 10.6 ms at 7,522 tokens against 5.6 ms at 8,192. The
  likeliest cause is that their KV slots, freed by earlier requests of other
  sizes, are in more pieces, and each piece is a separate transfer. KV is
  allocated one token at a time (page size 1).
- A prompt longer than one prefill step goes in chunks, and only its last
  chunk's transfer comes after the forward: 6.0 ms for 12,288 tokens and
  10.5 ms for 32,768.

**2. Each request of a step costs about 2.2 ms of the prefill scheduler's
time.** It processes a step's results one request at a time, and for each it
builds the NIXL transfer descriptors and posts the transfer, before the next
request's. Consecutive requests of a step end 2.6 ms apart (median over steps;
p90 5.0 ms). The mooncake backend only queues the request there, at 0.7 ms
each.

**3. So a step's transfer time depends on N, not S.** A prefill step's
transfer time T runs from its first end to the last of its requests reaching
the decode server. `src/compare_fits.py` fits it with the four polynomial
regressions of the step-latency models, in the step's N, S and M, with no
floor and no bends. The 1,720 prefill steps of the plan's batches come in 938
sets of prompt lengths; each set is one batch, fitted on the median of its
steps. Held out means 5-fold cross-validation over those sets:

| model | params | held-out R^2 | held-out RMSE | mean error | median | max |
|---|---|---|---|---|---|---|
| linear | 4 | 0.9739 | 7.9 ms | 28.6% | 15.0% | 173% |
| degree 2 | 10 | 0.9729 | 8.1 ms | 28.9% | 15.2% | 475% |
| degree 2 + sum l^2 | 11 | 0.9731 | 8.1 ms | 28.6% | 15.1% | 470% |
| selected | 4 | 0.9739 | 7.9 ms | 28.6% | 15.0% | 173% |

- **Backward elimination keeps the linear model,** as do four of the five
  folds of the nested cross-validation:
  T = 7.92 + 2.111 N + 0.00079 S - 0.00100 M ms. The N term is 2.11 ms per
  request (95% interval [2.07, 2.14]); dropping it more than triples the
  error, while dropping S or M adds 2 points.
- **S hardly matters.** A full 8,192-token step adds at most 6.5 ms through
  S, about what three more requests add through N.
- **The relative errors come from steps of one prompt,** 177 of the 938
  (mean error 71%): they take 4-6 ms and the model gives them about 10. For
  steps of several prompts the mean error is 17-19%.

**4. Bootstrap delays prefill.** A prompt can be prefilled only once the
decode server has told the prefill server where its KV goes. SGLang does
that per request, and the prefill server picks bootstrapped requests up only
between steps:

| requests sent together | 1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|---|
| bootstrap, median (ms) | 1 | 1 | 1 | 77 | 78 | 80 | 84 | 93 | 599 |
| bootstrap, p90 (ms) | 2 | 77 | 81 | 102 | 108 | 86 | 95 | 1,295 | 2,301 |

Only 29% of the sent batches ran as one prefill step. The requests that
finish bootstrap first are prefilled together, and the rest join the next
step, about 80 ms later, one eager prefill step's CPU time. In 39% of the
sent batches of several prompts the first step held one request. So a sent
batch's time, from its first end to its last request's KV, also holds the
prefill of its later steps, which is why the model above is per prefill step.
The same S split among more requests takes longer (median ms):

| S | N = 1 | N = 2 | N = 8 | N = 32 | N = 64 |
|---|---|---|---|---|---|
| 1,024 | 3.9 | 19.5 | 21.7 | 79.4 | 140.3 |
| 4,096 | 4.6 | 28.9 | 80.3 | 133.2 | 209.9 |
| 8,192 | 5.6 | 127.7 | 202.4 | 258.9 | 312.0 |

**5. The decode server finds a transfer at its next poll.** A request joins a
decode step 0.9 ms after the decode server finds its KV (median; p90 3.5 ms,
p99 10.7 ms). The server polls once per scheduler loop, and its loops get
longer with more requests waiting.

## Saturation: bursts

A burst sends more prompts of one length than one prefill step holds: 512
short prompts, or 64k tokens. Prefill and transfer then run at once.

**The transfer kept up with prefill at every prompt length.** It drained the
KV at least as fast as prefill produced it (medians over 3 samples):

| prompt | requests | prefill steps | prefill (tok/s) | transfer (tok/s) | transfer (req/s) | median delay (ms) | max delay (ms) |
|---|---|---|---|---|---|---|---|
| 16 | 512 | 3 | 4,775 | 5,628 | 351.7 | 279.5 | 872 |
| 32 | 512 | 4 | 8,244 | 9,353 | 292.3 | 197.3 | 566 |
| 64 | 512 | 5 | 10,407 | 11,865 | 185.4 | 160.5 | 444 |
| 128 | 512 | 8 | 15,476 | 17,102 | 133.6 | 96.5 | 306 |
| 256 | 256 | 9 | 21,783 | 24,817 | 96.9 | 93.0 | 233 |
| 512 | 128 | 9 | 25,151 | 29,658 | 57.9 | 53.2 | 145 |
| 1,024 | 64 | 9 | 26,501 | 30,458 | 29.7 | 32.2 | 78 |
| 2,048 | 32 | 9 | 24,807 | 26,522 | 13.0 | 23.1 | 343 |
| 4,096 | 16 | 9 | 22,021 | 24,257 | 5.9 | 15.7 | 382 |
| 8,192 | 8 | 8 | 17,495 | 20,385 | 2.5 | 481.2 | 486 |

- **Short prompts are limited by the prefill server's CPU, not the
  transfer.** For short prompts, the 2-3 ms per request of item 2 happens on
  the prefill scheduler's own thread. So it also limits prefill: 16-token
  prompts were prefilled at 4,775 tokens/s, 300 requests per second.
- **Delays grow with load.** In a short-prompt burst they run to hundreds of
  ms, because all requests are queued behind each other in the prefill
  scheduler and the decode server's loop.
- **A large transfer posted while the next step computes can wait for that
  step.**
  - In the 8,192-token bursts, every step is one prompt, and with overlap
    scheduling a step's results are processed after the next step is
    launched.
  - Of the 21 transfers of over 150 MB posted while the next step computed,
    13 (62%) took over 100 ms, median 482 ms: one 8,192-token forward,
    instead of the 10 ms they take otherwise. Each request's KV reached the
    decode server just after the next step ended.
  - Smaller transfers rarely did: 2 of the 134 of 30 to 150 MB (1%). Those
    under 30 MB come in steps of many requests, whose delays item 2 sets.
  - The likeliest cause: a large transfer needs UCX to make progress, which
    NIXL does when the scheduler calls into it, and the scheduler doesn't
    while it waits for the next step's forward.

`results/gptoss-pd-mooncake-tcp-20261103` has the same plan on SGLang's
default backend, mooncake, which on this VM falls back to TCP. In its smoke
test (`smoke/` there), a burst of 64 prompts of 1,024 tokens drained at 10.4k
tokens/s against prefill's 21k: there the transfer was what saturated.
