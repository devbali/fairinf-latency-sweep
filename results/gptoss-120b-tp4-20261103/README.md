# gpt-oss-120b on four A100s with tensor parallelism: latency models

The final eval of 2026-11-03: the full sweep, `src/batches.json`, on
gpt-oss-120b with `--tp 4` over the four A100s of `fairinf-matteo-fork-4xa100`.
Everything else is the setup of `results/gptoss-cuda-graphs-20261029`,
including piecewise CUDA graphs for prefill and hybrid steps.

- `results.jsonl`: 16,392 samples (5,464 batches, each timed 3 times).
- `run.log`: the runner's output, from rank 0.

Every fit number below comes from running, in the repo root:

```bash
python src/compare_fits.py results/gptoss-120b-tp4-20261103/results.jsonl
```

## The run

| | |
|---|---|
| date | 2026-11-03, 05:08 to 06:01 UTC: 11 minutes to load the model and capture graphs, then 42 minutes of timing |
| VM | `fairinf-matteo-fork-4xa100`: four A100-SXM4-80GB, every pair joined by NVLink (NV12) |
| SGLang, packages | fairinf-sglang main `99f39f2b5`, which differs from the 20b runs' `4e84427ab` only in retraction code that `runner.py` never calls; the fresh venv, installed unmodified |
| model | `lmsys/gpt-oss-120b-bf16` at revision `3609e677`: BF16, with the experts stored as FP8 E5M2, which holds gpt-oss's MXFP4 values exactly. SGLang loads all of it as BF16, 54.6 GB per GPU. |
| command | `MODEL=/mnt/ram/gpt-oss-120b-bf16 FORK=$LT/fairinf-sglang-99f39f2b5 CUDA_VISIBLE_DEVICES=0,1,2,3 PIECEWISE=inductor bash vm/run_clean.sh src/batches.json OUT --tp 4 --mem-fraction-static 0.819` |
| flags | those of the 20b sweep, `--dtype bfloat16 --sampling-backend pytorch --piecewise-cuda-graph-compiler inductor` with `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`, plus `--tp 4`. `--mem-fraction-static 0.819` is SGLang's own default for four 80 GB GPUs. The 20b runs' 0.7 would leave about 1.4 GB per GPU for the KV cache after the weights, room for about 70k tokens. |
| how it times | `src/runner.py` runs one process per GPU, as `sglang.bench_one_batch` does. Every rank builds the same batches from the same seed and runs them in step; rank 0 records the times, which include waiting for the other ranks at each all-reduce. |
| KV cache | 469,904 tokens in the sliding-window layers and 587,380 in the full-attention ones, per GPU; the plan's largest batch needs 299,180 |
| CUDA graphs | decode at batch sizes 1 to 512, every 8 up to 256 as before; the plan's decode batches go up to 256, so they are padded as in the 20b runs. Prefill and hybrid steps as piecewise graphs at the same 58 token counts, 4 to 8,192. |
| batches | `src/batches.json`, unchanged: 2,122 prefill, 1,985 decode and 1,357 hybrid. Its limits hold here too: 8,192 tokens per step is SGLang's default chunked-prefill size for these GPUs, and the KV cache is larger. |

The model was downloaded into a RAM disk, as the VM's boot disk has 6 GB free;
`vm/README.md` has the commands.

## Against gpt-oss-20b on one A100

Batch by batch against `results/gptoss-cuda-graphs-20261029`, the same plan
with the same flags on one GPU. Median step time:

| tokens in the step | prefill, 20b | prefill, 120b | ratio | hybrid, 20b | hybrid, 120b | ratio |
|---|---|---|---|---|---|---|
| under 256 | 27.7 ms | 27.0 ms | 0.96 | 33.8 ms | 28.9 ms | 0.83 |
| 256-1,023 | 47.8 ms | 46.1 ms | 0.98 | 65.6 ms | 58.0 ms | 0.81 |
| 1,024-2,047 | 83.6 ms | 74.6 ms | 0.89 | 104.9 ms | 85.3 ms | 0.79 |
| 2,048-4,095 | 142.0 ms | 110.8 ms | 0.77 | 200.0 ms | 134.3 ms | 0.67 |
| 4,096-8,192 | 287.1 ms | 183.7 ms | 0.63 | 443.6 ms | 244.8 ms | 0.55 |

| decode requests | 20b | 120b | ratio |
|---|---|---|---|
| 1 | 10.3 ms | 7.5 ms | 0.73 |
| 2-4 | 14.5 ms | 10.2 ms | 0.74 |
| 5-16 | 15.6 ms | 13.0 ms | 0.84 |
| 17-64 | 19.8 ms | 16.9 ms | 0.85 |
| 65-256 | 27.6 ms | 26.8 ms | 0.97 |

A hybrid step's tokens are S_p + N_d. The ratio is the median of each batch's
120b time over its 20b time; over all batches it is 0.72 for prefill, 0.86
for decode and 0.62 for hybrid.

- **Small prefill steps cost what they cost on one GPU:** a 16-token prompt
  takes 20.6 ms, against 20.3 ms for 20b.
- **The more work a step holds, the more the four GPUs gain.** 120b has 1.4
  times 20b's active parameters per token, split over four GPUs, plus two
  all-reduces per layer. Prefill steps of 4k-8k tokens take 0.63 times as
  long as on one GPU with 20b, hybrid ones 0.55 times.
- **Decode is 15-27% faster up to 64 requests,** as each GPU reads a quarter
  of the weights a token uses, and about as fast from 65 to 256 requests.
  There the tokens touch most experts, and a quarter of 120b's weights is
  more than all of 20b's.
- **The noise is as before:** a batch's samples spread by 0.4% (prefill), 2.8%
  (decode) and 0.3% (hybrid), median.

## The models

Held-out mean error, max in brackets, with the 20b run's selected model for
comparison:

| | linear | degree 2 | degree 2 + sum l^2 | selected | selected, 20b |
|---|---|---|---|---|---|
| prefill | 5.8% (38%) | 4.6% (49%) | 4.6% (49%) | 4.6% (51%), 10 params | 2.7% (36%) |
| decode | 3.8% (19%) | 3.4% (17%) | 3.4% (17%) | 3.4% (17%), 14 params | 2.6% (29%) |
| hybrid | 37.7% (381%) | 4.2% (61%) | 4.1% (60%) | 4.1% (64%), 14 params | 3.0% (43%) |

- **No floor:** every model fits F = 0 but the linear hybrid one.
- **The selection keeps the kernel's terms again:** N*M, S*M and sum l^2 for
  prefill, and all five for hybrid, of which M_p*N_d matters most (dropping
  it takes the held-out error to 31%). The sum l^2 coefficient is -0.56
  times S*M's for prefill and -0.55 for hybrid, near the kernel's -0.5.
- **The errors are higher than on one GPU.** The largest misses are the
  smallest prefill steps: 2 to 6 prompts of 16 to 64 tokens take 20-25 ms,
  and the selected model gives them about 30. Its constant, 28.8 ms, is set
  by the larger steps.
- **Decode needs its bends:** without them, its held-out error is 6.6%
  instead of 3.4%.
- **Held out** means 5-fold cross-validation over the batches; the nested
  cross-validation, which also chooses the terms within each training fold,
  gives the same errors to 0.1 points.

## Full fit output

## prefill: 2122 batches, samples spread by 0.4% (median)

With the floor; errors are on held-out batches.

| model | params | R^2 | held-out R^2 | held-out RMSE | mean error | median | p90 | max |
|---|---|---|---|---|---|---|---|---|
| linear | 5 | 0.9759 | 0.9757 | 9.8 ms | 5.8% | 4.4% | 12.2% | 37.8% |
| degree 2 | 11 | 0.9926 | 0.9925 | 5.4 ms | 4.6% | 3.1% | 9.8% | 48.6% |
| degree 2 + sum l^2 | 12 | 0.9928 | 0.9927 | 5.4 ms | 4.6% | 3.0% | 9.7% | 49.2% |
| selected | 10 | 0.9928 | 0.9927 | 5.4 ms | 4.6% | 3.0% | 9.6% | 51.3% |

With the terms also chosen inside each training fold (nested cross-validation), the selected model's held-out error is mean 4.6%, median 3.0%, p90 9.7%, max 51.3%.
- fold 0 chose 8 terms: the same ones
- fold 1 chose 8 terms: the same ones
- fold 2 chose 8 terms: the same ones
- fold 3 chose 7 terms: -N*S
- fold 4 chose 8 terms: the same ones

Held-out error by the step's prompts: mean, max.

| model | one prompt (246) | equal prompts (127) | unequal prompts (1749) |
|---|---|---|---|
| linear | 8.3%, 37.8% | 12.0%, 35.5% | 5.0%, 28.6% |
| degree 2 | 6.3%, 48.6% | 12.9%, 46.5% | 3.8%, 32.8% |
| degree 2 + sum l^2 | 6.3%, 49.2% | 12.7%, 46.9% | 3.7%, 33.2% |
| selected | 6.5%, 51.3% | 12.8%, 48.4% | 3.7%, 34.4% |

The polynomial alone, without the floor.

| model | params | R^2 | held-out R^2 | held-out RMSE | mean error | median | p90 | max |
|---|---|---|---|---|---|---|---|---|
| linear | 4 | 0.9759 | 0.9757 | 9.8 ms | 5.8% | 4.4% | 12.2% | 37.8% |
| degree 2 | 10 | 0.9926 | 0.9925 | 5.4 ms | 4.6% | 3.1% | 9.8% | 48.6% |
| degree 2 + sum l^2 | 11 | 0.9928 | 0.9927 | 5.4 ms | 4.6% | 3.0% | 9.7% | 49.2% |

Backward elimination from degree 2 + sum l^2: held-out error after each drop. The selected model is the smallest with mean error at most 4.68%: the best, 4.57%, plus one standard error.

| params | dropped | mean error | max |
|---|---|---|---|
| 12 | none | 4.57% | 49.2% |
| 11 | M^2 | 4.57% | 49.3% |
| 10 (selected) | N^2 | 4.60% | 51.3% |
| 9 | N*S | 4.71% | 55.7% |
| 8 | M | 4.80% | 57.0% |
| 7 | N | 4.92% | 59.7% |
| 6 | sum l^2 | 5.01% | 60.1% |
| 5 | N*M | 6.64% | 60.7% |
| 4 | S^2 | 8.39% | 97.5% |
| 3 | S*M | 9.18% | 65.4% |
| 2 | S | 62.62% | 552.5% |

The selected model, in ms per unit.

| term | coefficient | 95% interval | held-out mean error without it |
|---|---|---|---|
| floor F | 0 | [0, 0] | |
| const | 28.79 | [28.1, 29.5] |  |
| N | 0.2314 | [0.2, 0.267] | 4.90% |
| S | 0.02342 | [0.0229, 0.024] | 11.61% |
| M | 0.003101 | [0.00226, 0.00387] | 4.72% |
| N*S | -3.558e-05 | [-4.46e-05, -2.58e-05] | 4.71% |
| N*M | 0.0003235 | [0.000299, 0.000341] | 5.18% |
| S^2 | -5.422e-07 | [-6.29e-07, -4.63e-07] | 4.98% |
| S*M | 1.941e-06 | [1.72e-06, 2.16e-06] | 4.76% |
| sum l^2 | -1.094e-06 | [-1.31e-06, -8.88e-07] | 4.73% |

Coefficients in ms per unit.

| term | linear | degree 2 | degree 2 + sum l^2 | selected |
|---|---|---|---|---|
| floor F | 0 | 0 | 0 | 0 |
| const | 25.7 | 27.99 | 28.18 | 28.79 |
| N | 0.5231 | 0.422 | 0.3559 | 0.2314 |
| S | 0.01985 | 0.02254 | 0.02305 | 0.02342 |
| M | 0.01126 | 0.004105 | 0.00363 | 0.003101 |
| N^2 |  | -0.004009 | -0.002349 |  |
| N*S |  | -9.223e-06 | -2.868e-05 | -3.558e-05 |
| N*M |  | 0.000311 | 0.0003145 | 0.0003235 |
| S^2 |  | -5.743e-07 | -5.289e-07 | -5.422e-07 |
| S*M |  | 1.672e-06 | 1.901e-06 | 1.941e-06 |
| M^2 |  | -7.944e-07 | -8.697e-08 |  |
| sum l^2 |  |  | -9.906e-07 | -1.094e-06 |

sum l^2 / S*M: -0.564 in the selected model (95% interval [-0.630, -0.502]), -0.521 in degree 2 + sum l^2

## decode: 1985 batches, samples spread by 2.8% (median)

With the bends; errors are on held-out batches.

| model | params | R^2 | held-out R^2 | held-out RMSE | mean error | median | p90 | max |
|---|---|---|---|---|---|---|---|---|
| linear | 11 | 0.9850 | 0.9848 | 0.9 ms | 3.8% | 3.0% | 7.8% | 19.3% |
| degree 2 | 17 | 0.9891 | 0.9888 | 0.8 ms | 3.4% | 2.8% | 7.1% | 17.3% |
| degree 2 + sum l^2 | 18 | 0.9893 | 0.9890 | 0.8 ms | 3.4% | 2.8% | 7.1% | 17.1% |
| selected | 14 | 0.9890 | 0.9887 | 0.8 ms | 3.4% | 2.9% | 7.3% | 16.6% |

With the terms also chosen inside each training fold (nested cross-validation), the selected model's held-out error is mean 3.4%, median 2.9%, p90 7.3%, max 16.6%.
- fold 0 chose 12 terms: +N^2, +N*S, -M^2, -sum l^2, -(N-128)+
- fold 1 chose 13 terms: the same ones
- fold 2 chose 13 terms: the same ones
- fold 3 chose 13 terms: the same ones
- fold 4 chose 13 terms: the same ones

The polynomial alone, without the bends.

| model | params | R^2 | held-out R^2 | held-out RMSE | mean error | median | p90 | max |
|---|---|---|---|---|---|---|---|---|
| linear | 4 | 0.9466 | 0.9461 | 1.8 ms | 9.2% | 7.0% | 21.3% | 44.0% |
| degree 2 | 10 | 0.9761 | 0.9756 | 1.2 ms | 6.6% | 5.1% | 14.0% | 30.0% |
| degree 2 + sum l^2 | 11 | 0.9767 | 0.9763 | 1.2 ms | 6.6% | 4.9% | 14.1% | 30.3% |

Backward elimination from degree 2 + sum l^2: held-out error after each drop. The selected model is the smallest with mean error at most 3.43%: the best, 3.37%, plus one standard error.

| params | dropped | mean error | max |
|---|---|---|---|
| 18 | none | 3.37% | 17.1% |
| 17 | N^2 | 3.37% | 17.1% |
| 16 | (N-2)+ | 3.39% | 17.1% |
| 15 | S*M | 3.40% | 17.1% |
| 14 (selected) | N*S | 3.42% | 16.6% |
| 13 | (N-16)+ | 3.46% | 16.2% |
| 12 | (N-32)+ | 3.48% | 16.5% |
| 11 | (N-64)+ | 3.51% | 18.3% |
| 10 | S^2 | 3.59% | 19.5% |
| 9 | N*M | 3.64% | 19.7% |
| 8 | M^2 | 3.81% | 20.2% |
| 7 | sum l^2 | 3.93% | 21.1% |
| 6 | (N-4)+ | 4.13% | 21.0% |
| 5 | (N-128)+ | 4.76% | 22.7% |
| 4 | S | 6.21% | 29.3% |
| 3 | (N-8)+ | 11.07% | 49.4% |
| 2 | M | 22.42% | 86.5% |
| 1 | N | 40.35% | 145.5% |

The selected model, in ms per unit.

| term | coefficient | 95% interval | held-out mean error without it |
|---|---|---|---|
| const | 6.467 | [6.37, 6.55] |  |
| N | 0.7957 | [0.764, 0.836] | 4.85% |
| S | 3.748e-05 | [3.46e-05, 4.04e-05] | 4.12% |
| M | 0.0001073 | [0.000101, 0.000113] | 4.62% |
| N*M | 8.98e-08 | [6.63e-08, 1.29e-07] | 3.52% |
| S^2 | -6.878e-11 | [-7.89e-11, -5.68e-11] | 3.50% |
| M^2 | 2.402e-10 | [1.86e-10, 3.21e-10] | 3.54% |
| sum l^2 | -2.258e-10 | [-3.21e-10, -1.69e-10] | 3.53% |
| (N-4)+ | -0.4004 | [-0.462, -0.354] | 3.72% |
| (N-8)+ | -0.2385 | [-0.268, -0.204] | 3.59% |
| (N-16)+ | -0.07682 | [-0.1, -0.0564] | 3.46% |
| (N-32)+ | 0.04741 | [0.0346, 0.0607] | 3.47% |
| (N-64)+ | -0.04242 | [-0.0526, -0.0326] | 3.48% |
| (N-128)+ | -0.02938 | [-0.0354, -0.0228] | 3.50% |

Coefficients in ms per unit.

| term | linear | degree 2 | degree 2 + sum l^2 | selected |
|---|---|---|---|---|
| const | 6.36 | 6.272 | 6.247 | 6.467 |
| N | 0.8996 | 0.8995 | 0.9298 | 0.7957 |
| S | 1.764e-05 | 3.435e-05 | 3.558e-05 | 3.748e-05 |
| M | 0.0001185 | 0.0001121 | 0.0001103 | 0.0001073 |
| N^2 |  | -1.385e-05 | -8.571e-06 |  |
| N*S |  | 5.334e-08 | 3.985e-08 |  |
| N*M |  | 5.325e-08 | 4.576e-08 | 8.98e-08 |
| S^2 |  | -8.294e-11 | -8.022e-11 | -6.878e-11 |
| S*M |  | -1.579e-11 | 3.961e-11 |  |
| M^2 |  | 2.581e-11 | 1.44e-10 | 2.402e-10 |
| sum l^2 |  |  | -1.723e-10 | -2.258e-10 |
| (N-2)+ | -0.1673 | -0.1451 | -0.1721 |  |
| (N-4)+ | -0.286 | -0.3365 | -0.3498 | -0.4004 |
| (N-8)+ | -0.281 | -0.2532 | -0.2466 | -0.2385 |
| (N-16)+ | -0.07395 | -0.08256 | -0.08118 | -0.07682 |
| (N-32)+ | 0.03776 | 0.04558 | 0.04754 | 0.04741 |
| (N-64)+ | -0.04138 | -0.04359 | -0.04372 | -0.04242 |
| (N-128)+ | -0.02996 | -0.02745 | -0.02801 | -0.02938 |

## hybrid: 1357 batches, samples spread by 0.3% (median)

With the floor; errors are on held-out batches.

| model | params | R^2 | held-out R^2 | held-out RMSE | mean error | median | p90 | max |
|---|---|---|---|---|---|---|---|---|
| linear | 8 | 0.8309 | 0.8221 | 72.2 ms | 37.7% | 22.9% | 75.4% | 380.6% |
| degree 2 | 29 | 0.9981 | 0.9980 | 7.7 ms | 4.2% | 2.3% | 8.2% | 61.2% |
| degree 2 + sum l^2 | 30 | 0.9982 | 0.9980 | 7.6 ms | 4.1% | 2.3% | 8.0% | 60.5% |
| selected | 14 | 0.9980 | 0.9979 | 7.9 ms | 4.1% | 2.3% | 8.3% | 63.8% |

With the terms also chosen inside each training fold (nested cross-validation), the selected model's held-out error is mean 4.2%, median 2.4%, p90 8.3%, max 63.8%.
- fold 0 chose 12 terms: the same ones
- fold 1 chose 12 terms: the same ones
- fold 2 chose 12 terms: +M_p, -N_p*M_p
- fold 3 chose 11 terms: -sum l_p^2
- fold 4 chose 11 terms: -N_p*M_p

Held-out error by the step's prompts: mean, max.

| model | one prompt (327) | equal prompts (165) | unequal prompts (865) |
|---|---|---|---|
| linear | 73.8%, 380.6% | 31.2%, 214.7% | 25.3%, 231.2% |
| degree 2 | 8.0%, 61.2% | 3.5%, 10.0% | 2.9%, 23.3% |
| degree 2 + sum l^2 | 7.9%, 60.5% | 3.5%, 9.8% | 2.8%, 23.4% |
| selected | 8.1%, 63.8% | 2.9%, 14.4% | 2.8%, 22.5% |

The polynomial alone, without the floor.

| model | params | R^2 | held-out R^2 | held-out RMSE | mean error | median | p90 | max |
|---|---|---|---|---|---|---|---|---|
| linear | 7 | 0.7868 | 0.7800 | 80.3 ms | 33.1% | 20.3% | 87.4% | 335.0% |
| degree 2 | 28 | 0.9981 | 0.9980 | 7.7 ms | 4.2% | 2.3% | 8.2% | 61.2% |
| degree 2 + sum l^2 | 29 | 0.9982 | 0.9980 | 7.6 ms | 4.1% | 2.3% | 8.0% | 60.5% |

Backward elimination from degree 2 + sum l^2: held-out error after each drop. The selected model is the smallest with mean error at most 4.15%: the best, 3.97%, plus one standard error.

| params | dropped | mean error | max |
|---|---|---|---|
| 30 | none | 4.11% | 60.5% |
| 29 | S_d^2 | 4.09% | 60.6% |
| 28 | N_d^2 | 4.07% | 64.1% |
| 27 | S_p*N_d | 4.05% | 63.3% |
| 26 | N_p*N_d | 4.01% | 63.1% |
| 25 | S_p*S_d | 3.98% | 62.9% |
| 24 | M_d^2 | 3.98% | 62.9% |
| 23 | N_p*M_d | 3.97% | 62.9% |
| 22 | M_p^2 | 3.97% | 62.9% |
| 21 | N_p*S_d | 3.98% | 62.8% |
| 20 | M_p*M_d | 3.98% | 62.8% |
| 19 | S_d*M_d | 3.98% | 62.2% |
| 18 | N_d*S_d | 3.99% | 60.7% |
| 17 | S_d | 3.98% | 60.6% |
| 16 | N_d*M_d | 4.00% | 60.4% |
| 15 | S_p*M_d | 4.04% | 61.1% |
| 14 (selected) | M_p | 4.10% | 63.8% |
| 13 | sum l_p^2 | 4.17% | 64.5% |
| 12 | N_p^2 | 4.20% | 64.9% |
| 11 | N_p*S_p | 4.48% | 70.9% |
| 10 | N_p*M_p | 4.78% | 70.4% |
| 9 | N_p | 5.11% | 74.5% |
| 8 | M_d | 5.82% | 83.1% |
| 7 | S_p^2 | 6.94% | 108.4% |
| 6 | S_p*M_p | 7.17% | 85.8% |
| 5 | N_d | 7.79% | 97.5% |
| 4 | M_p*S_d | 14.63% | 101.2% |
| 3 | M_p*N_d | 36.62% | 112.1% |
| 2 | S_p | 102.64% | 931.4% |

The selected model, in ms per unit.

| term | coefficient | 95% interval | held-out mean error without it |
|---|---|---|---|
| floor F | 0 | [0, 0] | |
| const | 28.59 | [27.3, 29.8] |  |
| N_p | 0.6915 | [0.508, 0.84] | 4.45% |
| S_p | 0.02544 | [0.0248, 0.0261] | 14.93% |
| N_d | 0.1433 | [0.126, 0.163] | 5.72% |
| M_d | 0.001024 | [0.000935, 0.00111] | 4.97% |
| N_p^2 | -0.002771 | [-0.00433, -0.000737] | 4.18% |
| N_p*S_p | -0.0001315 | [-0.000161, -0.000103] | 4.31% |
| N_p*M_p | 0.000432 | [0.000353, 0.000533] | 4.20% |
| S_p^2 | -9.146e-07 | [-1.02e-06, -8.26e-07] | 5.40% |
| S_p*M_p | 3.26e-06 | [2.85e-06, 3.7e-06] | 4.36% |
| M_p*N_d | 0.0005546 | [0.000539, 0.000567] | 31.12% |
| M_p*S_d | 8.97e-07 | [8.86e-07, 9.06e-07] | 10.82% |
| sum l_p^2 | -1.78e-06 | [-2.19e-06, -1.39e-06] | 4.17% |

Coefficients in ms per unit.

| term | linear | degree 2 | degree 2 + sum l^2 | selected |
|---|---|---|---|---|
| floor F | 94.86 | 0 | 0 | 0 |
| const | -219.8 | 27.89 | 27.63 | 28.59 |
| N_p | -4.036 | 0.835 | 0.8379 | 0.6915 |
| S_p | 0.04529 | 0.02273 | 0.02373 | 0.02544 |
| M_p | 0.04523 | 0.003454 | 0.002769 |  |
| N_d | 1.921 | 0.1836 | 0.1836 | 0.1433 |
| S_d | 0.001546 | -1.496e-05 | -9.205e-06 |  |
| M_d | 0.006105 | 0.001059 | 0.001092 | 0.001024 |
| N_p^2 |  | -0.005618 | -0.00429 | -0.002771 |
| N_p*S_p |  | -7.975e-05 | -0.0001355 | -0.0001315 |
| N_p*M_p |  | 0.0004817 | 0.0003972 | 0.000432 |
| N_p*N_d |  | 0.0001815 | 0.0001477 |  |
| N_p*S_d |  | -4.76e-07 | -3.374e-07 |  |
| N_p*M_d |  | 3.866e-06 | 3.854e-06 |  |
| S_p^2 |  | -8.158e-07 | -7.209e-07 | -9.146e-07 |
| S_p*M_p |  | 2.45e-06 | 2.966e-06 | 3.26e-06 |
| S_p*N_d |  | 6.636e-06 | 6.952e-06 |  |
| S_p*S_d |  | 1.072e-08 | 9.595e-09 |  |
| S_p*M_d |  | -7.787e-09 | -1.684e-08 |  |
| M_p^2 |  | -1.148e-06 | 1.379e-08 |  |
| M_p*N_d |  | 0.0005455 | 0.0005454 | 0.0005546 |
| M_p*S_d |  | 8.898e-07 | 8.891e-07 | 8.97e-07 |
| M_p*M_d |  | -7.353e-08 | -6.074e-08 |  |
| N_d^2 |  | -0.0002415 | -0.0002437 |  |
| N_d*S_d |  | -5.765e-07 | -6.025e-07 |  |
| N_d*M_d |  | 1.029e-05 | 1.037e-05 |  |
| S_d^2 |  | 1.871e-10 | 2.054e-10 |  |
| S_d*M_d |  | 4.143e-09 | 2.406e-09 |  |
| M_d^2 |  | -1.376e-08 | -1.251e-08 |  |
| sum l_p^2 |  |  | -1.802e-06 | -1.78e-06 |

sum l_p^2 / S_p*M_p: -0.546 in the selected model (95% interval [-0.595, -0.481]), -0.608 in degree 2 + sum l^2

## Summary, held out, with the floor and bends (transfer has neither)

| | linear | degree 2 | degree 2 + sum l^2 | selected |
|---|---|---|---|---|
| prefill parameters | 5 | 11 | 12 | 10 |
| prefill held-out R^2 | 0.9757 | 0.9925 | 0.9927 | 0.9927 |
| prefill mean error | 5.8% | 4.6% | 4.6% | 4.6% |
| prefill max error | 37.8% | 48.6% | 49.2% | 51.3% |
| decode parameters | 11 | 17 | 18 | 14 |
| decode held-out R^2 | 0.9848 | 0.9888 | 0.9890 | 0.9887 |
| decode mean error | 3.8% | 3.4% | 3.4% | 3.4% |
| decode max error | 19.3% | 17.3% | 17.1% | 16.6% |
| hybrid parameters | 8 | 29 | 30 | 14 |
| hybrid held-out R^2 | 0.8221 | 0.9980 | 0.9980 | 0.9979 |
| hybrid mean error | 37.7% | 4.2% | 4.1% | 4.1% |
| hybrid max error | 380.6% | 61.2% | 60.5% | 63.8% |

## Summary, held out, the polynomial alone

| | linear | degree 2 | degree 2 + sum l^2 |
|---|---|---|---|
| prefill held-out R^2 | 0.9757 | 0.9925 | 0.9927 |
| prefill mean error | 5.8% | 4.6% | 4.6% |
| prefill max error | 37.8% | 48.6% | 49.2% |
| decode held-out R^2 | 0.9461 | 0.9756 | 0.9763 |
| decode mean error | 9.2% | 6.6% | 6.6% |
| decode max error | 44.0% | 30.0% | 30.3% |
| hybrid held-out R^2 | 0.7800 | 0.9980 | 0.9980 |
| hybrid mean error | 33.1% | 4.2% | 4.1% |
| hybrid max error | 335.0% | 61.2% | 60.5% |
