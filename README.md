# fairinf-latency-sweep

This repo measures how long one forward step of SGLang takes on batches built
by hand. It fits latency models in the style of FairInference's linear ones
([arXiv 2609.18112](https://arxiv.org/abs/2609.18112), §5.4) to those times,
piecewise linear where the hardware changes regime, and checks how well they
predict batches they were not fitted on.

The current target is **gpt-oss-20b in BF16 on one A100 80GB**, running the
FairInference fork of SGLang pinned to commit `4e84427ab`
(`matteo/fairbatching`).

| path | what it does |
|---|---|
| `sanity_latency.py` | Given the command that starts an SGLang server, runs `src/sanity_batches.json` in its place and prints how the setup and times differ from the reference run, `results/gptoss-sanity-cuda-graphs-20261029`. |
| `sanity_accuracy.py` | Given the same command, starts that server, asks it the GPQA Diamond questions with OpenAI's own gpt-oss eval code, and checks the score against OpenAI's published one. |
| `src/runner.py` | Loads SGLang's `ModelRunner` once, then times every batch in `--batches`: 1 warmup run, then 3 samples. |
| `src/batches.json` | The batch plan, every batch written out: `prefill`, `decode` and `hybrid` lists. |
| `src/generate_batches.py` | Writes both plans, `src/batches.json` and `src/sanity_batches.json`, from the designs and seeded random draws below. |
| `src/compare_fits.py` | Fits the four models below: linear, degree-2 and degree-2 + sum l^2 regressions, and the terms that matter picked by backward elimination. Prints their accuracy and coefficients as markdown tables. |
| `src/fit.py` | Helpers the scripts share: loading results, least squares with the CPU floor, cross-validation and the bootstrap. |
| `src/sanity_batches.json` | A short plan of 88 batches from `src/batches.json`, for `sanity_latency.py`. |
| `src/compare_runs.py` | Compares a run with a reference run batch by batch, and compares the setup lines of their logs. |
| `src/server_command.py` | Reads the server command the sanity checks take, and runs something else in the server's place. |
| `vm/` | Setup and run scripts for the A100 VM. `vm/README.md` lists the exact setup, server command and commands used on it. |
| `results/` | One directory per run: its samples, its log, and a README on what it shows. |

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
spread is 0.5% for prefill, 2.8% for decode and 0.3% for hybrid.

## The models

| | N | S | M |
|---|---|---|---|
| prefill | requests | total prompt tokens | longest prompt |
| decode | requests, padded to the CUDA-graph size | total context tokens | longest context |
| hybrid | `N_p`, `N_d` | `S_p`, `S_d` | `M_p`, `M_d` |

`_p` is the prefill part of a hybrid step and `_d` the decode part. sum l^2 is
the sum of the squared prompt lengths, and `(x)+` means `max(0, x)`.

`src/compare_fits.py` fits four regressions, each with a polynomial part f:

```
linear               f = c + a*N + b*S + d*M
degree 2             f = linear + every square and product of N, S and M
degree 2 + sum l^2   f = degree 2 + q * sum l^2
selected             f = the terms of degree 2 + sum l^2 that matter most
```

All four wrap f the same way:

```
prefill, hybrid:  T = max(F, f)
decode:           T = f + sum_k delta_k * (N - k)+        k = 2, 4, 8, ..., 128
```

- **Hybrid steps** use all six variables, so their degree-2 polynomial has 21
  squares and products. Their sum l^2 is over the prefill part's prompts.
- **The selected model** comes from backward elimination. Starting from
  degree 2 + sum l^2, it drops one term at a time, each time the one whose
  removal leaves the lowest held-out error. Of the models along the way, it
  keeps the smallest whose error is within one standard error of the best
  one's.
- **Every model is linear in its coefficients,** so least squares fits it.

**Prefill and hybrid have a floor without CUDA graphs.** Run eagerly, below
about 1k tokens the CPU launching kernels is the bottleneck, and a step takes
67-74 ms whatever its size. The models take the larger of the floor F and the
GPU time f. The fit alternates: f on the steps above the floor, F on the rest,
until no step changes side. With piecewise CUDA graphs there is no floor: the
fits put F at 0, and a 16-token prompt takes about 19 ms.

**Decode bends with batch size.** Decode runs as CUDA graphs and has no floor.
Each of the first few requests adds a lot, most likely because each new token
pulls in more of gpt-oss's 32 experts. In the linear model, a request adds
1.6 ms up to 2 requests and 1.2 ms up to 4, then 0.03-0.15 ms. A breakpoint at
every power of two lets the slope change at each doubling.

**The attention terms come from SGLang's Triton extend-attention kernel,**
which gpt-oss must use on A100:
- It launches `ceil(M / BLOCK_M)` query blocks for every request in the step,
  sized by the longest prompt (`triton_ops/extend_attention.py:609`).
- Each block loops over its request's cached context (line 337), then over its
  new tokens up to the block's end (line 433). A block past the request's last
  token does the same work with every row masked; the kernel has no early
  exit.

So a prompt of l tokens costs about `l*M - l^2/2` key reads, which sum to
`S*M - sum l^2 / 2` over the step, plus a fixed cost per block, `N*M`. N*M and
S*M are in the degree-2 polynomial, but sum l^2 is not, and no polynomial in
N, S and M can express it. Hence the extra term. The selected model agrees:
its sum l^2 coefficient comes out at -0.45 times its S*M one for prefill, and
-0.52 times for hybrid, close to the -1/2 this predicts.

In a hybrid step each decode is a 1-token prompt with a long cached context.
It also gets `ceil(M_p / BLOCK_M)` blocks, each looping over that whole
context, hence the N_d*M_p and S_d*M_p terms. If the prefill and decode parts
just added up, these cross terms would be zero. Measured on gpt-oss:

| step | time |
|---|---|
| a 2,048-token prompt alone | ~100 ms |
| + 64 decodes with 1 token of context | 316 ms |
| + 64 decodes with 512 tokens of context | 444 ms |

## Fitting the models

```bash
python src/compare_fits.py results/<run>/results.jsonl
```

For each phase, `src/compare_fits.py` fits every model by least squares on the
median of each batch's samples. It prints markdown tables of:
- **accuracy:** R^2 on the batches each model was fitted on, then R^2, RMSE
  and relative error on held-out batches from 5-fold cross-validation. Each
  held-out batch is predicted by a model fitted without it. This is the number
  to judge a model by.
- **the noise:** how far each batch's 3 samples spread. An error near that
  level is measurement noise, not model error.
- **the elimination path** that picks the selected model, and the same
  selection repeated inside each training fold (nested cross-validation), to
  check that picking the terms didn't overfit.
- **every coefficient,** in ms per unit, with a 95% interval from 200
  bootstrap refits for the selected model. An interval that spans zero marks a
  term the data can't pin down.

R^2 is dominated by the largest steps. Hybrid steps range from 69 ms to 3.6 s,
so a hybrid model that misses small steps by 30% still scores 0.996. Relative
error is the better guide.

Mean error on held-out batches, in the two full runs:

| run | phase | batches | linear | degree 2 | degree 2 + sum l^2 | selected | sample noise (median) |
|---|---|---|---|---|---|---|---|
| no CUDA graphs, `results/gptoss-clean-20261028` | prefill | 618 | 6.2% | 1.6% | 1.3% | 1.3% | 0.5% |
| | decode | 565 | 2.9% | 2.7% | 2.7% | 2.7% | 2.8% |
| | hybrid | 448 | 43.8% | 2.3% | 2.2% | 2.0% | 0.3% |
| CUDA graphs, `results/gptoss-cuda-graphs-20261029` | prefill | 2,122 | 11.1% | 2.8% | 2.7% | 2.7% | 0.4% |
| | decode | 1,985 | 2.9% | 2.7% | 2.6% | 2.6% | 2.9% |
| | hybrid | 1,357 | 56.9% | 3.2% | 3.1% | 3.0% | 0.2% |

- **Without CUDA graphs,** the selected models have 7, 9 and 12 parameters,
  against 12, 18 and 30 for degree 2 + sum l^2. Every coefficient, the full
  accuracy tables, and where the sum l^2 term comes from are in that run's
  [README](results/gptoss-clean-20261028/README.md).
- **With CUDA graphs,** the errors come from the smallest steps and from hybrid
  steps with a tiny prompt and many decodes, which the floor used to hide. That
  run's [README](results/gptoss-cuda-graphs-20261029/README.md) has the
  details, and how a padded token count helps the prefill model.

## Planning batches

`src/batches.json` lists every batch of the plan. `src/generate_batches.py`
writes it from the designs below and seeded random draws. The plan follows
three rules.

**1. Vary S, M and N independently.**
- In uniform batches S = N*M, so a fit on them alone cannot separate the S
  term from the M term. The attention terms tie up the same way: N*M = S and
  sum l^2 = S*M.
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
| prefill | uniform: N in {1, 2, 4, ..., 64} x L in {16, 32, ..., 8192}, N*L <= 8,192 | 49 |
| | skewed: one long prompt (512-6,144) + N - 1 short ones (16, 64 or 256), N in {2, 4, ..., 32} | 69 |
| | random: total 16-8,192, N <= 64 | 500 |
| | one prompt: every 16 tokens up to 256, every 32 up to 1,024, every 128 up to 8,192 | 86 |
| | uniform, denser: N in {1, 2, 3, 4, 6, 8, ..., 48, 64} x L in {16, 24, 32, 48, ..., 6,144, 8,192} | 89 |
| | skewed, denser: one long prompt (256-6,144) + N - 1 short ones (16-128), N in {2, 3, 4, 6, ..., 64} | 330 |
| | random, more of the same | 999 |
| decode | uniform: N in {1, 2, 4, ..., 256} x context in {64, 256, 1k, 4k, 16k, 64k}, within the KV limit | 42 |
| | skewed: one long context (8k-96k) + N - 1 short ones, N in {8, 32, 128, 256} | 23 |
| | random: total 32-300k, N <= 256, contexts up to 128k | 500 |
| | uniform, denser: every N up to 16, then 20, 24, 28, 32, 40, ..., 224, 256 x every power of two from 64 to 128k | 221 |
| | skewed, denser: one long context (8k-128k) + N - 1 short ones (64-4k), N in {2, 4, ..., 256} | 199 |
| | random, more of the same | 1,000 |
| hybrid | 6 prefill parts x 8 decode parts, each part also timed on its own | 48 |
| | random: decode part up to 64k context tokens, prefill part fills the rest of the budget | 400 |
| | 13 prefill parts x 27 decode parts of up to 64k context tokens | 309 |
| | random, more of the same | 600 |

That makes 5,464 batches, each run 4 times: about 80 minutes on one A100 at
the clean run's step times. With CUDA graphs, timing them took 64 minutes
(`results/gptoss-cuda-graphs-20261029`).
- **The plan of `results/gptoss-clean-20261028` comes first:** in each phase,
  the rows down to the first random one, drawn exactly as then. So a new run
  compares with that one batch by batch. A batch already in an earlier row is
  left out of later ones.
- **Hybrid steps take the most time,** because of the attention interaction
  above. For that reason the newer hybrid designs keep their decode part to
  64k context tokens.
- **To change the plan,** edit `src/generate_batches.py` and run
  `python src/generate_batches.py`. `--batches` can also point at another
  file, such as `src/sanity_batches.json`.

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
`vm/README.md` has, in one place, what is installed where on it, the server
command every run used, and the exact commands of the sweep and both checks.

```bash
gcloud compute instances start fairinf-matteo-fork-1xa100-eu1 --zone europe-west4-a
gcloud compute instances stop fairinf-matteo-fork-1xa100-eu1 --zone europe-west4-a --discard-local-ssd=true

# copy the repo over
gcloud compute scp --recurse --zone europe-west4-a src vm \
    devbali@fairinf-matteo-fork-1xa100-eu1:~/latency-test/fairinf-latency-sweep/
```

On the VM, from `~/latency-test/fairinf-latency-sweep`, the latest sweep
(`results/gptoss-cuda-graphs-20261029`) ran:

```bash
PIECEWISE=inductor nohup bash vm/run_clean.sh src/batches.json results.jsonl > run.log 2>&1 &   # ~70 min; survives SSH drops
```

`vm/run_clean.sh` runs `src/runner.py` with `--dtype bfloat16
--mem-fraction-static 0.7 --sampling-backend pytorch`. `PIECEWISE=inductor`
runs prefill and hybrid steps as piecewise CUDA graphs; the script says why
that takes torch inductor. Without it, the script passes
`--disable-piecewise-cuda-graph`, the setup of `results/gptoss-clean-20261028`.
Flags added after the output file go to SGLang and override these. It needs
three things under `~/latency-test`, all in place on eu1:
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

Copy `results.jsonl` and `run.log` back into `results/<run>/` and run
`src/compare_fits.py` on it.

### Checking a system's latency

`sanity_latency.py` checks that a system runs SGLang with the setup the latency
models were fitted on. Give it the command that starts the SGLang server
there. It runs `src/runner.py` in the server's place, with the same
environment, Python and flags. With the reference run's setup, that is:

```bash
SERVER="TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
PYTHONPATH=/path/to/fairinf-sglang/python:/path/to/fairinf-latency-sweep/vm \
python3 -m sglang.launch_server --model-path lmsys/gpt-oss-20b-bf16 --dtype bfloat16 \
--mem-fraction-static 0.7 --sampling-backend pytorch --piecewise-cuda-graph-compiler inductor \
--scheduling-policy-path noop_cache_policy.NoOpWithCachePolicy --port 30000"

python sanity_latency.py "$SERVER"
```

`sanity_accuracy.py` below takes the same command. `vm/README.md` explains
each part of it.

It times the 88 batches of `src/sanity_batches.json`, all also in
`src/batches.json`: 20 prefill, 20 decode and 48 hybrid. They cover the
smallest steps, the attention terms, decode from 1 to 256 requests, and the
hybrid design. Then it compares them with the reference run,
`results/gptoss-sanity-cuda-graphs-20261029`: the same batches on eu1, with
piecewise CUDA graphs as above (`PIECEWISE=inductor` in `vm/run_clean.sh`). It
prints:
- the setup lines of the two run logs that differ: package versions, GPU, MoE
  kernel, KV cache and CUDA-graph sizes;
- every batch's time in both runs, with `!` on differences beyond the sample
  noise;
- a verdict, also given as the exit code: 0 if the run matches, 1 if not.

The run matches if the setup lines do and no phase's times differ. A phase
differs if its batches are more than 5% slower or faster on average, or if
more than 10% of them differ by more than noise. A setup difference moves
many batches; a single small step can be off by chance. An identical rerun on
eu1 had one 10 ms decode step 9% slow
(`results/gptoss-sanity-latency-eu1-20261029`).

The 5% leaves room for drift. Runs of the same setup on eu1 differ by at most
1.2% on average in any phase. Added to the noise of those runs, a shift of up
to 3% in one phase always passes, and one of 6% or more always fails. Setup
differences are larger: prefill without CUDA graphs is 53% slower.

Notes on running it:
- **The command** may carry environment variables, `nohup` and a trailing
  redirect or `&`, as the FairInference launch scripts do.
- **Server-only flags** such as `--port` are accepted and change nothing.
- **`--dry-run`** prints what would run.
- **Output:** the run's `results.jsonl` and `run.log` go to a new directory
  under `results/`.
- **Requirements:** it needs this repo, with the reference run, and numpy.

By hand, the same check is `vm/run_clean.sh` on the plan, then
`src/compare_runs.py`, which prints the comparison as markdown:

```bash
PIECEWISE=inductor nohup bash vm/run_clean.sh src/sanity_batches.json results.jsonl > run.log 2>&1 &   # on the VM
python src/compare_runs.py results/gptoss-sanity-cuda-graphs-20261029/results.jsonl results/<run>/results.jsonl
```

`results/gptoss-sanity-check-20261029/` is the earlier check of the setup
without CUDA graphs, against `results/gptoss-clean-20261028`.

### Checking accuracy on a system

`sanity_accuracy.py` checks that a system's SGLang setup computes the right
answers, not only that it runs at the right speed. It takes the same server
command as `sanity_latency.py`, but starts that server for real. Then it asks
the server the 198 GPQA Diamond questions with OpenAI's own eval code for
gpt-oss: `gpt_oss.evals` from
[github.com/openai/gpt-oss](https://github.com/openai/gpt-oss), which OpenAI's
[guide to verifying gpt-oss implementations](https://developers.openai.com/cookbook/articles/gpt-oss/verifying-implementations)
runs.

```bash
python sanity_accuracy.py "$SERVER"   # SERVER as for sanity_latency.py above
```

The eval's prompt, answer grading and client run unchanged, with their
settings: temperature 1, up to 131,072 tokens per answer, every question in
flight at once. Only the number of attempts differs. OpenAI's script, `python
-m gpt_oss.evals --eval gpqa`, asks each question 8 times; the check asks once
by default, and `--attempts 8` matches the script. The score is compared with
OpenAI's for gpt-oss-20b, from the model card
([arXiv 2508.10925](https://arxiv.org/abs/2508.10925), Table 3, GPQA Diamond
without tools): 56.8% at low reasoning effort, 66.0% at medium and 71.5% at
high. `--effort` picks one; low is the default.

- **The score passes** if it is at most max(5 points, 3 standard errors) below
  OpenAI's, with the standard error over the 198 questions: 46.2% at low
  effort. A broken setup scores near chance, 25%.
- **One attempt per question is enough.** Drawing one answer per question from
  the 8-attempt run below, 20,000 times over, gives 53.5% to 63.1% (1st to 95th
  percentile), and never below 46.2%.
- **It takes about 6 minutes** on eu1: 5 to start the server with CUDA
  graphs, under 1 for the questions.
- **Two flags are added** to a command that lacks them. Neither changes how
  the model runs.
  - `--reasoning-parser gpt-oss` keeps the model's reasoning out of the
    response's content, where the eval looks for the answer.
  - `--allow-auto-truncate` is needed because the eval asks for 131,072 new
    tokens, the model's whole context. Without it this fork rejects every such
    request (`tokenizer_manager.py`, `_validate_one_request`), and the eval
    scores each rejection as a wrong answer. With it, SGLang lowers max_tokens
    to what fits after the prompt.
- **The eval's code** is a checkout of github.com/openai/gpt-oss: `--gpt-oss
  DIR`, or by default a clone at commit `2770201` (main on 2026-10-29) in
  `~/.cache/sanity-accuracy/`.
- **The eval's packages:** the eval runs with the Python that runs the check,
  or `--python`. It needs gpt-oss's eval packages: openai, pandas, numpy,
  jinja2, tqdm and blobfile. A venv of their own keeps them apart from the
  server's:

  ```bash
  python3 -m venv evals-venv && evals-venv/bin/pip install openai pandas numpy jinja2 tqdm blobfile
  evals-venv/bin/python sanity_accuracy.py "$SERVER"
  ```
- **A gpt-oss server on this fork** needs the cache policy in the command,
  `--scheduling-policy-path noop_cache_policy.NoOpWithCachePolicy` with `vm/`
  on `PYTHONPATH`; see "Serving gpt-oss on this fork" below.
  `sanity_latency.py` accepts it too, and it changes nothing there.
- **Output:** a new directory under `results/` holds the server's log, the
  eval's output, every question, answer and grade (`gpqa-*-answers.json.gz`)
  and `scores.json`. The exit code is 0 if the score passes, 1 if it is too
  low, and 2 if the check could not run.

On eu1, with the CUDA-graph setup at low effort, both runs matched OpenAI's
56.8%:

| run | attempts per question | score |
|---|---|---|
| `results/gptoss-sanity-accuracy-eu1-20261029` | 1 | 54.5% |
| `results/gptoss-openai-evals-cuda-graphs-20261029` | 8 | 59.0% |

Before this, the check ran SGLang's eval script, `sglang.test.run_eval`, on the
same server command. It scored GPQA Diamond 36.4% with CUDA graphs and 33.3%
without, and MMLU 68.4% and 68.0%, against OpenAI's 56.8% and 80.4%
(`results/gptoss-accuracy-cuda-graphs-20261029`,
`results/gptoss-accuracy-clean-20261029`). Since OpenAI's code gets OpenAI's
score from the same server, that script, not the setup, was off. Its GPQA
prompt and grading differ from OpenAI's: it asks for a last line `Answer:
$LETTER` and counts only the first `Answer: X` in the response, while
OpenAI's grader accepts the letter in many forms. Which of the two caused the
gap was not checked.

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

- **Without CUDA graphs, prefill and hybrid steps have a CPU floor.** Below
  about 1k tokens they take 67-74 ms whatever their size; the fitted floor is
  70 ms for prefill and 71 ms for hybrid.
  - The overhead is Python plus kernel launches. Piecewise CUDA graphs remove
    it, with flags only (`PIECEWISE=inductor` in `vm/run_clean.sh`). The
    default eager compiler can't trace the fork's RMSNorm, FlashInfer's kernel
    (`layernorm.py:261`), so they need `--piecewise-cuda-graph-compiler
    inductor`, which runs it as torch code, and `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`.
    In `hooks-venv` they can't work at all, as torch 2.5.1's compiler can't
    trace `ForwardMode.is_extend()`.
  - With them, steps of 1k tokens or more run 3-4% slower, mostly because a
    step is padded up to the next of the 58 captured token counts.
  - Most of gpt-oss's overhead is its MoE path, about 8 Triton launches per
    layer. Llama's floor is 41–48 ms.
  - Decode uses regular CUDA graphs and doesn't pay this.
- **The attention terms above are real cost,** not a fitting artifact, in plain
  prefill steps as well as hybrid ones.
- **Prompts here have no cached prefix.** A continued chunk on a real server
  has one, and every block of that request loops over it as well. That adds a
  prefix·M term these runs don't measure.
- **Decode times step at the CUDA-graph sizes.** A batch of 100 runs as 104;
  the models use the padded size.
- **Why gpt-oss in BF16:**
  - The hooks venv's `sgl_kernel` is a stub on A100. Its MoE routing,
    alignment and INT8/FP8 GEMM kernels all return empty tensors. That ruled
    out Gemma 4 and Nemotron there; the fresh venv has the real one.
  - gpt-oss's MoE goes through `triton_kernels` instead and avoids the stub.
  - Its MXFP4 weights are converted to BF16 in software on A100, which runs
    the experts about 44× slower. Hence the BF16 checkpoint.
