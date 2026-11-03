# SGLang PD disaggregation with the fork's defaults: mooncake over TCP

`src/pd_runner.py` on `fairinf-matteo-fork-4xa100` on 2026-11-03, with
SGLang's PD disaggregation set up as the fork does by default: the mooncake
transfer backend, its queues and threads, and the bootstrap port 8998. The
prefill server ran on GPU 0 and the decode server on GPU 1, each the gpt-oss
server command of `vm/README.md` on fairinf-sglang main (`99f39f2b5`) with
`--disaggregation-mode` and a port added. `vm/README.md` has the exact
commands, and `run.log` the setup.

**This run is kept as the record of why the PD model doesn't use the fork's
default backend.** It broke 2.8 minutes into the plan, after 52 of the 334
batches of `src/pd_batches.json` (each sent 3 times). The PD transfer model
comes from `results/gptoss-pd-nixl-20261103`, the full plan over NVLink.

- `results.jsonl`: 970 requests, one line each (see `pd_runner.py`): 810 got
  both answers, 160 failed. `pd_runner.py`'s `requests.jsonl`, the same lines
  without the servers' log stages, was left out.
- `prefill.log`, `decode.log`: the two servers' logs.
- `smoke/`: the smoke test just before, with the same setup and
  `pd_runner.py --samples 1` on a short plan: after the warmup, the batches
  `[16]`, `[1024]`, `[8192]`, 16 x `[64]`, 256 x `[16]` and `[6144, 16, 16,
  16]`, and one burst of 64 x `[1024]`.

The numbers below come from `python src/pd_fits.py` on `results.jsonl` and
on `smoke/results.jsonl`.

## How the KV cache moved

The VM has no RDMA cards, so mooncake falls back to TCP: its log says "Found 0
HCAs", then "TcpTransport: listen on port ...". Each transfer goes from GPU 0
to host memory, over a loopback TCP connection, and from host memory to GPU 1.
PD mode also turns piecewise CUDA graphs off (`server_args.py`, "13. PD
disaggregation"), so prefill runs eagerly.

## Why it broke

At 01:37:38 UTC one transfer failed: `TcpTransport::getConnection failed to
create connection to 10.0.1.13:16544. Error: connect: Cannot assign requested
address`. The VM had run out of ephemeral ports.

- **Mooncake's TCP transport opens a new connection for every slice of every
  transfer** and closes it afterwards, unless `MC_TCP_ENABLE_CONNECTION_POOL=1`
  is set (`tcp_transport.h` in mooncake 0.3.10.post2: `bool
  enable_connection_pool_ = false`). A closed connection holds its port in
  TIME_WAIT for 60 s, and a request's transfer is dozens of slices: the K and
  V of each of 24 layers, per contiguous run of KV slots, plus its first
  token's data.
- **SGLang gives up on a decode server after one failed transfer**
  (`mooncake/conn.py`: "if the session fails once, mark it as failed"). Every
  later request failed with "Decode instance could be dead, remote mooncake
  session ... is not alive", and `pd_runner.py` stopped after five failed
  batches in a row.
- **Times turned erratic first.** From batch 42 (8 prompts, 1.9 minutes in),
  the three samples of a batch differ several-fold: 145, 339 and 647 ms for
  8 x 128 tokens. Before that they agree within a few percent. The kernel's
  search for a free port slows as TIME_WAIT sockets pile up.

So the fork's default PD setup can't serve a sustained load on a machine
without RDMA.

## What it measured before it broke

- **A request carries 24 KB x (l + min(l, 128)),** as over NVLink: the K and
  V of the 12 full-attention layers for each of its l prompt tokens, plus
  those of the 12 sliding-window layers for its last 128 tokens, gpt-oss's
  window.
- **One request alone is up to 30 times slower than over NVLink,** about
  1.2 GB/s: 1.2 times at 16 tokens, 30 times at 8,192. From the end of its
  prefill step to the decode server finding its KV cache, median of 3
  samples:

  | prompt tokens | 16 | 128 | 512 | 1,024 | 2,048 | 4,096 | 8,192 |
  |---|---|---|---|---|---|---|---|
  | delay (ms) | 4.5 | 8.7 | 18.0 | 28.4 | 49.4 | 89.4 | 169.8 |

- **Transfers go one at a time.** Mooncake shards transfers into its 4 queues
  by the decode server's port, so with one decode server they all share one
  queue and one thread.
- **Under load the transfer saturated.** The run broke before its bursts, but
  the smoke test sent one: 64 prompts of 1,024 tokens. Prefill produced
  21,210 tokens/s while the transfer drained 10,417 (10.2 requests/s), so the
  last request's KV arrived 3.6 s after its prefill. Over NVLink the same
  burst drained 30,458 tokens/s, faster than prefill.
