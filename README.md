# fairinf-latency-sweep

This repo measures how long one forward step of SGLang takes on batches built
by hand. It fits latency models in the style of FairInference's linear ones
([arXiv 2609.18112](https://arxiv.org/abs/2609.18112), §5.4) to those times,
piecewise linear where the hardware changes regime, and checks how well they
predict batches they were not fitted on.

The current target is **gpt-oss-20b in BF16 on one A100 80GB**, running the
FairInference fork of SGLang pinned to commit `4e84427ab`
(`matteo/fairbatching`).

| file | what it does |
|---|---|
| `runner.py` | Loads SGLang's `ModelRunner` once, then times every batch in `--batches`: 1 warmup run, then 3 samples. |
| `batches.py` | The batch plan: `PREFILL_BATCHES`, `DECODE_BATCHES` and `HYBRID_BATCHES`. |
| `fit.py` | Fits the piecewise-linear models below and reports R² and errors, in-sample and cross-validated. |
| `compare_fits.py` | Compares linear, degree-2 and degree-2 + sum l^2 regressions, plus the terms that matter picked by backward elimination, and prints markdown tables. |
| `sanity_batches.py` | A short plan of 88 batches from `batches.py`, 1.5 minutes of timing, for checking that a VM matches the latest run. |
| `compare_runs.py` | Compares a run with a reference run batch by batch, and with `fit.py`'s model, and compares the setup lines of their logs. |
| `vm/` | Setup and run scripts for the A100 VM. |

## What one step is

`runner.py` uses the same code path as `python -m sglang.bench_one_batch`: no
server, scheduler or radix cache. A batch is a list of request lengths, and
each request is filled with random tokens.

- **Prefill** (`time_prefill`): one forward pass over new prompts. Request i
  has `lens[i]` prompt tokens.
- **Decode** (`time_decode`): one decode step for requests that already hold
  `lens[i]` tokens of context.
  - No prefill runs first. SGLang allocates the context's KV cache but nothing
    fills it, because a decode step costs the same whatever the cached values
    are.
  - In an A/B test on Llama, this ran 0–7% faster than decoding right after a
    real prefill. The gap came from the GPU's power state, not the cache
    contents.
- **Hybrid** (`time_hybrid`): one mixed step, built exactly as SGLang's
  scheduler builds it under `--enable-mixed-chunk`. It holds new prompts plus
  one decode token for every running request, in a single forward pass
  (`prepare_for_decode` on the running batch, then `mix_with_running`).

A time is host wall clock from batch preparation through sampling, with the
GPU synced before and after. Each batch runs once as a warmup, then 3 more
times as samples. The 3 samples usually agree within a few percent: the median
spread is 0.6% for prefill, 2.9% for decode and 0.3% for hybrid.

## The models to fit

| | N | S | M |
|---|---|---|---|
| prefill | requests | total prompt tokens | longest prompt |
| decode | requests, padded to the CUDA-graph size | total context tokens | longest context |
| hybrid | `N_p`, `N_d` | `S_p`, `S_d` | `M_p`, `M_d` |

`_p` is the prefill part of a hybrid step and `_d` the decode part. `Σl²` is
the sum of squared prompt lengths, and `(x)+` means `max(0, x)`.

```
prefill:  T = max(F_p, c + α·S + β·M + γ·N + η_N·N·M + η_S·S·M + η_Q·Σl²)

decode:   T = c + α·S + β·M + γ·N + Σ_k δ_k·(N − k)+        k = 2, 4, 8, …, 128

hybrid:   T = max(F_h, c + α_p·S_p + β_p·M_p + γ_p·N_p
                         + η_N·N_p·M_p + η_S·S_p·M_p + η_Q·Σl_p²
                         + α_d·S_d + β_d·M_d + γ_d·N_d
                         + θ_N·N_d·M_p + θ_S·S_d·M_p)
```

Each model is linear in its coefficients, so least squares fits it, and
piecewise linear where the hardware changes regime.

**Prefill and hybrid have a floor.** These steps run without CUDA graphs.
Below about 1k tokens the CPU launching kernels is the bottleneck, and a step
takes 67-74 ms whatever its size. The model takes the larger of the floor F
and the GPU time. `fit.py` fits the two pieces by alternating: the linear part
on the steps above the floor, the floor on the rest, until no step changes
side.

**Decode bends with batch size.** Decode runs as CUDA graphs and has no floor.
Each of the first few requests adds a lot (1.6 ms each up to 2 requests, 1.2 ms
up to 4), most likely because each new token pulls in more of gpt-oss's 32
experts. After that each request adds 0.03-0.15 ms. A breakpoint at every
power of two lets the slope change at each doubling.

**The attention terms come from SGLang's Triton extend-attention kernel,**
which gpt-oss must use on A100:
- It launches `ceil(M / BLOCK_M)` query blocks for every request in the step,
  sized by the longest prompt (`triton_ops/extend_attention.py:609`).
- Each block loops over its request's cached context (line 337), then over its
  new tokens up to the block's end (line 433). A block past the request's last
  token does the same work with every row masked; the kernel has no early
  exit.

So a prompt of l tokens costs about `l·M − l²/2` key reads, which sum to
`S·M − Σl²/2` over the step, plus a fixed cost per block, `N·M`. The fit
agrees: η_Q comes out at −0.46·η_S for prefill and −0.51·η_S for hybrid, close
to the −½ this predicts.

In a hybrid step each decode is a 1-token prompt with a long cached context.
It also gets `ceil(M_p / BLOCK_M)` blocks, each looping over that whole
context, hence θ_N·N_d·M_p and θ_S·S_d·M_p. If the prefill and decode parts
just added up, these cross terms would be zero. Measured on gpt-oss:

| step | time |
|---|---|
| a 2,048-token prompt alone | ~100 ms |
| + 64 decodes with 1 token of context | 316 ms |
| + 64 decodes with 512 tokens of context | 444 ms |

## Checking the fit

```bash
python fit.py results/<run>/results.jsonl
```

For each phase, `fit.py` fits least squares on the median of each batch's
samples. It prints:
- **the coefficients**, in ms per unit, each with a 95% interval from 200
  bootstrap refits. An interval that spans zero marks a term the data can't
  pin down.
- **R², adjusted R², RMSE, MAE and relative error** on the batches it was
  fitted on.
- **the same on held-out batches** from 5-fold cross-validation: each batch is
  predicted by a model fitted without it. This is the number to judge a model
  by.
- **the noise:** how far each batch's 3 samples spread. An error near that
  level is measurement noise, not model error.
- **the held-out error by batch size**, in quartiles of tokens per step, or
  requests for decode.

R² is dominated by the largest steps. Hybrid steps range from 84 ms to 3.6 s,
so a hybrid model that misses small steps by 30% still scores 0.996. Relative
error is the better guide.

The latest run (`results/gptoss-clean-20261028/`), on held-out batches:

| phase | batches | parameters | R² | RMSE | mean error | p90 | max | sample noise (median) |
|---|---|---|---|---|---|---|---|---|
| prefill | 618 | 8 | 0.9988 | 3.8 ms | 1.3% | 2.8% | 7.3% | 0.5% |
| decode | 565 | 11 | 0.9841 | 0.76 ms | 2.9% | 6.1% | 22.0% | 2.8% |
| hybrid | 448 | 13 | 0.9991 | 12.7 ms | 2.1% | 4.3% | 14.3% | 0.3% |

`compare_fits.py` fits linear and full degree-2 polynomial regressions to the
same batches, with and without the sum l^2 term, and picks the terms that
matter by backward elimination. Its results, every coefficient, and where the
sum l^2 term comes from are in that run's
[README](results/gptoss-clean-20261028/README.md).

## Planning batches

The plan in `batches.py` follows three rules.

**1. Vary S, M and N independently.**
- In uniform batches S = N·M, so a fit on them alone cannot separate α from β.
  The attention terms tie up the same way: N·M = S and Σl² = S·M.
- Skewed batches (one long request plus short ones) move M at nearly fixed S.
- Decode needs batches in every power-of-two range of N, one per piece of its
  model. Drawing N log-uniformly covers them.

**2. Stay inside what the server would actually run.** These limits are for
gpt-oss-20b on A100 80GB at `--mem-fraction-static 0.7`:

| limit | value | why |
|---|---|---|
| prefill step | S ≤ 8,192 tokens | `--chunked-prefill-size`, picked from GPU memory |
| hybrid step | S_p + N_d ≤ 8,192 tokens | each decode uses one token of the same budget |
| decode batch | N ≤ 256 | CUDA graphs exist for 1, 2, 4, 8, 12, 16, 24, 32, then every 8 up to 256; other sizes are padded up |
| KV cache | S + N ≤ ~300k tokens | the runner never evicts sliding-window tokens, so the smaller pool (309,703 tokens) is the limit |

**3. Sample enough to test the model, not just fit it.** Every phase combines
a structured design with a larger seeded random sample.
- **How a random batch is drawn:** first its total, then its request count,
  then a split of the total among the requests. The split uses lognormal
  weights, ranging from even to one dominant request.
- **Prefill and hybrid** draw the total uniformly, so batches near the 8,192
  budget, which a loaded server runs, are well covered.
- **Decode** draws the total log-uniformly, because contexts span four orders
  of magnitude.

| phase | design | batches |
|---|---|---|
| prefill | uniform: N ∈ {1, …, 64} × L ∈ {16, …, 8192}, N·L ≤ 8,192 | 49 |
| | skewed: one long request (512–6,144) + N − 1 short ones (16, 64 or 256), N ∈ {2, …, 32} | 69 |
| | random: total 16–8,192, N ≤ 64 | 500 |
| decode | uniform: N ∈ {1, …, 256} × context ∈ {64, 256, 1k, 4k, 16k, 64k}, within the KV limit | 42 |
| | skewed: one long context (8k–96k) + N − 1 short ones, N ∈ {8, 32, 128, 256} | 23 |
| | random: total 32–300k, N ≤ 256, contexts up to 128k | 500 |
| hybrid | 6 prefill parts × 8 decode parts, each part also timed on its own | 48 |
| | random: decode part up to 64k context tokens, prefill part fills the rest of the budget | 400 |

That makes 1,631 batches, each run 4 times: about 30 minutes on one A100.
- Hybrid steps take the most time, because of the attention interaction above.
  For that reason random hybrids keep their decode part to 64k context tokens.
- To change the plan, edit the lists or counts (`N_RANDOM`). Any list can be
  left out, and `--batches` can point at another file.

## How SGLang builds hybrid steps

A real server forms hybrid ("mixed") steps only when chunked prefill is on
**and** `--enable-mixed-chunk` is set (`scheduler.py:1092`). The FairInference
experiments don't set it, so every step they ran was pure prefill or pure
decode.

- **One token budget per step:** `--chunked-prefill-size`.
- **Decodes are paid for first.** Each running decode request uses one token
  of the budget before any prefill is added (`scheduler.py:2900`,
  `schedule_policy.py:418, 425`).
- **The prefill part is capped at `chunked-prefill-size − N_d`.** Free KV
  memory can lower that further (`schedule_policy.py:480`).
  - A partly prefilled prompt continues first.
  - Waiting prompts are then added whole while they fit.
  - The first one that doesn't fit is cut to the remainder; the rest of it
    runs in later steps.
- **Decodes have no separate cap.** Every running request joins the step. If
  N_d ≥ the budget, no prefill fits and the step is pure decode.
- **The merged step is one forward pass** (`ForwardMode.MIXED`), with each
  decode treated as a 1-token prefill on its context. Hybrid steps never use
  CUDA graphs.

Confirmed on a gpt-oss-20b server with `--chunked-prefill-size 2048`. 64
requests were decoding when a 7,000-token prompt arrived:

```
Prefill batch, #new-seq: 1, #new-token: 1984, ... #running-req: 64   ← 2048 − 64
Prefill batch, #new-seq: 1, #new-token: 1984, ... #running-req: 64
Prefill batch, #new-seq: 1, #new-token: 1984, ... #running-req: 64
Prefill batch, #new-seq: 1, #new-token: 1048, ... #running-req: 64   ← 7000 − 3 × 1984
```

While the 64 decoders were being admitted, the steps carried 2,047, 2,027 and
2,007 new tokens with 1, 21 and 41 requests running. That is 2,048 − N_d every
time.

## Running on the A100 VM

The VM is `fairinf-matteo-fork-1xa100-eu1` in europe-west4-a: a spot A100
80GB at about $3.10/hr. Keep one VM running at a time, and stop it when idle.

```bash
gcloud compute instances start fairinf-matteo-fork-1xa100-eu1 --zone europe-west4-a
gcloud compute instances stop fairinf-matteo-fork-1xa100-eu1 --zone europe-west4-a --discard-local-ssd=true

# copy the repo over
gcloud compute scp --recurse --zone europe-west4-a runner.py batches.py sanity_batches.py vm \
    devbali@fairinf-matteo-fork-1xa100-eu1:~/latency-test/fairinf-latency-sweep/
```

On the VM, from `~/latency-test/fairinf-latency-sweep`:

```bash
nohup bash vm/run_clean.sh batches.py results.jsonl > run.log 2>&1 &   # ~30 min; survives SSH drops
```

`vm/run_clean.sh` runs `runner.py` with the flags of the latest run:
`--dtype bfloat16 --mem-fraction-static 0.7 --sampling-backend pytorch
--disable-piecewise-cuda-graph`. Flags added after the output file go to
SGLang and override these. It needs three things under `~/latency-test`, all
in place on eu1:
- **SGLang at `4e84427ab`**, unpacked from the VM's own fairinf-sglang repo
  (step 1 of `vm/setup.sh` below).
- **A fresh venv** with the fork's own package pins, installed unmodified by
  `bash vm/fresh_venv.sh`: torch 2.11.0, the real sglang-kernel 0.4.2.post1,
  flashinfer 0.6.8.post1 and transformers 5.6.0.
- **`lmsys/gpt-oss-20b-bf16`** in the Hugging Face cache (step 3 of
  `vm/setup.sh`).

It appends one JSON line per sample to `results.jsonl`:

- `phase`: `prefill`, `decode` or `hybrid`
- `batch`: the lengths, or `[prefill lengths, decode lengths]` for hybrid
- `sample`: 0, 1 or 2
- `latency_s`

Copy `results.jsonl` and `run.log` back into `results/<run>/` and run `fit.py`
on it.

### Checking a VM against the latest run

`sanity_batches.py` is a short sweep for checking that a VM runs with the same
setup the latency models were fitted on. Its 88 batches are all in
`batches.py`: 20 prefill, 20 decode and 48 hybrid, covering the CPU floor, the
attention terms, decode from 1 to 256 requests, and the hybrid design. Timing
them takes about 1.5 minutes, after about 2 minutes of loading the model.

```bash
nohup bash vm/run_clean.sh sanity_batches.py results.jsonl > run.log 2>&1 &   # on the VM
python compare_runs.py results/gptoss-clean-20261028/results.jsonl results/<run>/results.jsonl
```

`compare_runs.py` compares each batch's median time with the latest run's and
with `fit.py`'s model. When `run.log` sits next to each `results.jsonl`, it
also compares the setup lines of the two logs: package versions, GPU, MoE
kernel, KV cache and CUDA-graph sizes. On a matching setup the setup lines
agree and the times differ by about the sample noise.
`results/gptoss-sanity-check-20261029/` is this check on eu1: the setup lines
matched, and the times differed by 0.1-0.5% on average in each phase.

### The older hooks-venv setup

The first gpt-oss sweep ran in `hooks-venv`, the FairInference experiments'
venv, through `vm/setup.sh` and `vm/run.sh`. Its GPU-side coefficients matched
the latest run's, but its floor was about 89 ms instead of 70. `vm/setup.sh`:

1. **Unpacks SGLang at `4e84427ab`** from the VM's own fairinf-sglang repo,
   leaving that checkout alone.
2. **Builds a side directory**, `~/latency-test/pydeps`, which `vm/run.sh`
   puts first on `PYTHONPATH`. `hooks-venv` itself stays untouched.
   - **transformers 5.6.0.** The fork pins it; the venv has 4.51.
   - **triton 3.6.0.** torch 2.11 (the fork's pin) ships with it, and
     `triton_kernels` needs 3.5 or newer.
   - **A copy of the venv's `sgl_kernel` stub,** with `is_arch_support_pdl()`
     returning `False`. The stub returns an empty CUDA tensor, which breaks
     Triton attention.
   - **A copy of `triton_kernels`.** It reads the shared-memory limit from the
     Triton driver, because torch 2.5.1 lacks `shared_memory_per_block_optin`.
     It also caches device queries it would otherwise repeat on every matmul,
     at about 1 ms each.
3. **Downloads `lmsys/gpt-oss-20b-bf16`.**

### Serving gpt-oss on this fork

`runner.py` never starts a server. A real gpt-oss server on this fork, for
example to check hybrid steps as above, needs
`--scheduling-policy-path noop_cache_policy.NoOpWithCachePolicy`, with `vm/`
on `PYTHONPATH`.

- **Why it's needed:** the fork's scheduler calls a cache hook on every
  admission (`scheduler.py:2941`), but only `RadixCache` defines it
  (`mem_cache/radix_cache.py:366`). gpt-oss's sliding-window layers get
  `SWARadixCache`, which crashes on the first request.
- **How the policy fixes it:** it attaches a no-op hook to whatever cache is in
  use (`scheduler.py:1131`). Scheduling stays SGLang's default.
- **What doesn't work:** `--disable-hybrid-swa-memory`. That path stores KV
  inside the rotary embedding, which the fallback RoPE the fork forces on A100
  rejects.

## Known effects on this stack

- **Prefill and hybrid steps have a CPU floor.** Below about 1k tokens they
  take 67-74 ms whatever their size; the fitted floor is 70 ms for prefill and
  71 ms for hybrid.
  - The overhead is Python plus kernel launches. Piecewise CUDA graphs would
    cut it, but they are off (`--disable-piecewise-cuda-graph`): on A100 the
    fork sends RMSNorm through FlashInfer's kernel (`layernorm.py:261`), which
    torch's compiler can't trace. In `hooks-venv` they can't work either, as
    torch 2.5.1's compiler can't trace `ForwardMode.is_extend()`.
  - Most of gpt-oss's overhead is its MoE path, about 8 Triton launches per
    layer. Llama's floor is 41–48 ms.
  - Decode uses regular CUDA graphs and doesn't pay this.
- **The attention terms above are real cost,** not a fitting artifact, in plain
  prefill steps as well as hybrid ones.
- **Prompts here have no cached prefix.** A continued chunk on a real server
  has one, and every block of that request loops over it as well. That adds a
  prefix·M term these runs don't measure.
- **Decode times step at the CUDA-graph sizes.** A batch of 100 runs as 104;
  `fit.py` uses the padded size.
- **Why gpt-oss in BF16:**
  - The hooks venv's `sgl_kernel` is a stub on A100. Its MoE routing,
    alignment and INT8/FP8 GEMM kernels all return empty tensors. That ruled
    out Gemma 4 and Nemotron there; the fresh venv has the real one.
  - gpt-oss's MoE goes through `triton_kernels` instead and avoids the stub.
  - Its MXFP4 weights are converted to BF16 in software on A100, which runs
    the experts about 44× slower. Hence the BF16 checkpoint.
