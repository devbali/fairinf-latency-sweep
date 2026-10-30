# gpt-oss-20b on one A100 with CUDA graphs: latency models

The latency sweep of 2026-10-29: the setup of `results/gptoss-clean-20261028`
plus piecewise CUDA graphs for prefill and hybrid steps, on the denser plan.

- `results.jsonl`: 16,392 samples (5,464 batches, each timed 3 times).
- `run.log`: the runner's output.

Every fit number below comes from running, in the repo root:

```bash
python src/compare_fits.py results/gptoss-cuda-graphs-20261029/results.jsonl
```

## The run

| | |
|---|---|
| date | 2026-10-29, 17:42 to 18:51 UTC: 3.5 minutes to load the model and capture graphs, then 64 minutes of timing |
| VM, SGLang, packages | as in `results/gptoss-clean-20261028`: eu1, fairinf-sglang `4e84427ab`, the fresh venv, installed unmodified |
| command | `PIECEWISE=inductor bash vm/run_clean.sh src/batches.json OUT` |
| flags | `--dtype bfloat16 --mem-fraction-static 0.7 --sampling-backend pytorch --piecewise-cuda-graph-compiler inductor`, with `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` in the environment |
| CUDA graphs | decode as before, at batch sizes 1 to 256. Prefill and hybrid steps as piecewise graphs split at attention, captured at 58 token counts from 4 to 8,192; a step runs as the next count up. |
| batches | `src/batches.json`: 2,122 prefill, 1,985 decode and 1,357 hybrid. The first 618, 565 and 448 are the clean run's batches. |

No code changed to get the graphs, only flags and an environment variable.
SGLang's default compiler for them, eager, can't trace the fork's RMSNorm
(FlashInfer's). Inductor runs RMSNorm and SGLang's other small ops as torch
code instead. With inductor, SGLang's compile hook hands torch 2.11's graph
cache a stand-in object it no longer accepts; with inductor's caches off,
torch never looks at it. Before this run, a check on the same VM compared the
graphs' logits with the eager path's on 14 prefill and hybrid batches: they
agreed to 0.5-3% (bf16 rounding), and the top token matched for 97-100% of
the requests.

## CUDA graphs against no CUDA graphs

On the 1,631 batches both runs timed, median step time:

| tokens in the step | prefill, graphs | prefill, no graphs | hybrid, graphs | hybrid, no graphs |
|---|---|---|---|---|
| under 256 | 26.5 ms | 68.5 ms | 50.7 ms | 70.7 ms |
| 256-1,023 | 46.5 ms | 70.2 ms | 71.4 ms | 73.1 ms |
| 1,024-2,047 | 82.7 ms | 77.2 ms | 130.2 ms | 123.8 ms |
| 2,048-4,095 | 135.7 ms | 129.8 ms | 213.5 ms | 205.1 ms |
| 4,096-8,192 | 287.3 ms | 278.2 ms | 458.0 ms | 446.2 ms |

- **The CPU floor is gone.** Without graphs, every prefill or hybrid step
  under about 1k tokens took about 70 ms. With them, a 16-token prompt takes
  about 19 ms.
- **Steps of 1k tokens or more are 3-4% slower, mostly from padding.** A step
  of 1,025 tokens runs as 1,280. For prefill, each 1% of padding adds 0.42% to
  the step; batches with no padding are 2.9% faster than without graphs.
  Hybrid steps are padded by 6% on average and run 3.4% slower.
- **Decode is unchanged:** the median ratio is 1.005.
- **The noise is as before:** a batch's samples spread by 0.4% (prefill), 2.9%
  (decode) and 0.2% (hybrid), median.

## The models

Held-out mean error, max in brackets, with the clean run's selected model for
comparison:

| | linear | degree 2 | degree 2 + sum l^2 | selected | selected, no graphs |
|---|---|---|---|---|---|
| prefill | 11.1% (137%) | 2.8% (31%) | 2.7% (32%) | 2.7% (36%), 8 params | 1.3% (7%) |
| decode | 2.9% (25%) | 2.7% (28%) | 2.6% (28%) | 2.6% (29%), 11 params | 2.7% (21%) |
| hybrid | 56.9% (647%) | 3.2% (47%) | 3.1% (48%) | 3.0% (43%), 11 params | 2.0% (14%) |

- **No floor:** every model but the linear one fits F = 0.
- **The selection keeps the kernel's terms again:** N*M, S*M and sum l^2 for
  prefill, and all five for hybrid. The sum l^2 coefficient is -0.42 times
  S*M's for prefill and -0.58 for hybrid, near the kernel's -0.5.
- **The errors are higher than without graphs,** in two places. The smallest
  steps: a 24-token prompt takes 18.4 ms against 24.2 predicted. And hybrid
  steps with a tiny prompt and many decodes: 16 tokens plus 256 decodes of
  128 takes 139 ms against 73 predicted (129 ms without graphs). Neither had
  shown before, because the floor covered the small steps.
- **The padded token count helps prefill.** Adding it and its square to
  degree 2 + sum l^2 lowers the held-out mean error from 2.7% to 1.95% (p90
  from 6.1% to 4.3%). It barely changes hybrid's.

## Full fit output

## prefill: 2122 batches, samples spread by 0.4% (median)

With the floor; errors are on held-out batches.

| model | params | R^2 | held-out R^2 | held-out RMSE | mean error | median | p90 | max |
|---|---|---|---|---|---|---|---|---|
| linear | 5 | 0.9610 | 0.9605 | 23.8 ms | 11.1% | 7.0% | 25.2% | 137.4% |
| degree 2 | 11 | 0.9968 | 0.9967 | 6.9 ms | 2.8% | 2.1% | 6.1% | 30.7% |
| degree 2 + sum l^2 | 12 | 0.9971 | 0.9969 | 6.6 ms | 2.7% | 1.9% | 6.1% | 32.1% |
| selected | 8 | 0.9971 | 0.9969 | 6.6 ms | 2.7% | 1.9% | 6.2% | 35.7% |

With the terms also chosen inside each training fold (nested cross-validation), the selected model's held-out error is mean 2.7%, median 1.9%, p90 6.2%, max 35.7%.
- fold 0 chose 6 terms: the same ones
- fold 1 chose 6 terms: the same ones
- fold 2 chose 7 terms: +M
- fold 3 chose 6 terms: the same ones
- fold 4 chose 6 terms: the same ones

Held-out error by the step's prompts: mean, max.

| model | one prompt (246) | equal prompts (127) | unequal prompts (1749) |
|---|---|---|---|
| linear | 18.5%, 137.4% | 24.6%, 117.0% | 9.1%, 63.9% |
| degree 2 | 3.1%, 30.7% | 5.6%, 17.2% | 2.5%, 13.4% |
| degree 2 + sum l^2 | 3.2%, 32.1% | 5.3%, 18.0% | 2.4%, 14.6% |
| selected | 3.4%, 35.7% | 6.0%, 19.4% | 2.4%, 14.7% |

The polynomial alone, without the floor.

| model | params | R^2 | held-out R^2 | held-out RMSE | mean error | median | p90 | max |
|---|---|---|---|---|---|---|---|---|
| linear | 4 | 0.9582 | 0.9578 | 24.6 ms | 12.1% | 6.7% | 27.9% | 100.0% |
| degree 2 | 10 | 0.9968 | 0.9967 | 6.9 ms | 2.8% | 2.1% | 6.1% | 30.7% |
| degree 2 + sum l^2 | 11 | 0.9971 | 0.9969 | 6.6 ms | 2.7% | 1.9% | 6.1% | 32.1% |

Backward elimination from degree 2 + sum l^2: held-out error after each drop. The selected model is the smallest with mean error at most 2.71%: the best, 2.66%, plus one standard error.

| params | dropped | mean error | max |
|---|---|---|---|
| 12 | none | 2.67% | 32.1% |
| 11 | N^2 | 2.66% | 32.3% |
| 10 | M^2 | 2.66% | 32.0% |
| 9 | S^2 | 2.66% | 31.3% |
| 8 (selected) | M | 2.71% | 35.7% |
| 7 | N*S | 2.89% | 44.1% |
| 6 | N | 3.07% | 46.3% |
| 5 | sum l^2 | 3.17% | 35.3% |
| 4 | N*M | 7.24% | 46.5% |
| 3 | S*M | 12.41% | 74.9% |
| 2 | S | 97.90% | 950.2% |

The selected model, in ms per unit.

| term | coefficient | 95% interval | held-out mean error without it |
|---|---|---|---|
| floor F | 0 | [0, 0] | |
| const | 23.61 | [23, 24.4] |  |
| N | 0.2123 | [0.175, 0.25] | 3.07% |
| S | 0.03156 | [0.0311, 0.0319] | 25.12% |
| N*S | -4.124e-05 | [-5.24e-05, -2.57e-05] | 2.89% |
| N*M | 0.0008949 | [0.000857, 0.000927] | 6.92% |
| S*M | 5.045e-06 | [4.68e-06, 5.43e-06] | 4.31% |
| sum l^2 | -2.126e-06 | [-2.49e-06, -1.79e-06] | 2.98% |

Coefficients in ms per unit.

| term | linear | degree 2 | degree 2 + sum l^2 | selected |
|---|---|---|---|---|
| floor F | 43.47 | 0 | 0 | 0 |
| const | -18.88 | 22.43 | 22.82 | 23.61 |
| N | 1.303 | 0.3832 | 0.2478 | 0.2123 |
| S | 0.03668 | 0.03009 | 0.03113 | 0.03156 |
| M | 0.02603 | 0.002517 | 0.001546 |  |
| N^2 |  | -0.003753 | -0.0003545 |  |
| N*S |  | 4.596e-06 | -3.523e-05 | -4.124e-05 |
| N*M |  | 0.0008727 | 0.0008797 | 0.0008949 |
| S^2 |  | -7.339e-08 | 1.948e-08 |  |
| S*M |  | 4.497e-06 | 4.965e-06 | 5.045e-06 |
| M^2 |  | -1.64e-06 | -1.924e-07 |  |
| sum l^2 |  |  | -2.027e-06 | -2.126e-06 |

sum l^2 / S*M: -0.421 in the selected model (95% interval [-0.458, -0.382]), -0.408 in degree 2 + sum l^2

## decode: 1985 batches, samples spread by 2.9% (median)

With the bends; errors are on held-out batches.

| model | params | R^2 | held-out R^2 | held-out RMSE | mean error | median | p90 | max |
|---|---|---|---|---|---|---|---|---|
| linear | 11 | 0.9844 | 0.9842 | 0.8 ms | 2.9% | 2.3% | 6.3% | 25.2% |
| degree 2 | 17 | 0.9879 | 0.9875 | 0.7 ms | 2.7% | 2.1% | 5.8% | 28.4% |
| degree 2 + sum l^2 | 18 | 0.9887 | 0.9883 | 0.7 ms | 2.6% | 2.0% | 5.7% | 27.8% |
| selected | 11 | 0.9885 | 0.9883 | 0.7 ms | 2.6% | 2.0% | 5.6% | 28.6% |

With the terms also chosen inside each training fold (nested cross-validation), the selected model's held-out error is mean 2.6%, median 2.0%, p90 5.7%, max 28.7%.
- fold 0 chose 10 terms: the same ones
- fold 1 chose 9 terms: -(N-2)+
- fold 2 chose 9 terms: -(N-2)+
- fold 3 chose 10 terms: the same ones
- fold 4 chose 10 terms: the same ones

The polynomial alone, without the bends.

| model | params | R^2 | held-out R^2 | held-out RMSE | mean error | median | p90 | max |
|---|---|---|---|---|---|---|---|---|
| linear | 4 | 0.9150 | 0.9143 | 1.8 ms | 7.9% | 6.0% | 18.5% | 39.2% |
| degree 2 | 10 | 0.9664 | 0.9660 | 1.1 ms | 5.3% | 3.7% | 11.0% | 28.8% |
| degree 2 + sum l^2 | 11 | 0.9677 | 0.9673 | 1.1 ms | 5.2% | 3.5% | 11.1% | 27.9% |

Backward elimination from degree 2 + sum l^2: held-out error after each drop. The selected model is the smallest with mean error at most 2.65%: the best, 2.60%, plus one standard error.

| params | dropped | mean error | max |
|---|---|---|---|
| 18 | none | 2.61% | 27.8% |
| 17 | (N-16)+ | 2.61% | 27.9% |
| 16 | S*M | 2.60% | 27.9% |
| 15 | N*M | 2.60% | 28.0% |
| 14 | N*S | 2.60% | 28.0% |
| 13 | N^2 | 2.61% | 28.0% |
| 12 | (N-32)+ | 2.61% | 28.4% |
| 11 (selected) | (N-8)+ | 2.62% | 28.6% |
| 10 | (N-2)+ | 2.67% | 28.7% |
| 9 | S^2 | 2.77% | 25.2% |
| 8 | M^2 | 2.88% | 25.8% |
| 7 | sum l^2 | 3.01% | 25.9% |
| 6 | (N-128)+ | 3.14% | 25.5% |
| 5 | (N-64)+ | 4.82% | 32.9% |
| 4 | S | 6.56% | 31.3% |
| 3 | (N-4)+ | 9.63% | 43.9% |
| 2 | M | 15.79% | 62.5% |
| 1 | N | 28.21% | 97.8% |

The selected model, in ms per unit.

| term | coefficient | 95% interval | held-out mean error without it |
|---|---|---|---|
| const | 8.405 | [8.27, 8.51] |  |
| N | 1.596 | [1.51, 1.67] | 3.24% |
| S | 3.796e-05 | [3.59e-05, 3.98e-05] | 3.45% |
| M | 5.645e-05 | [5.23e-05, 6.21e-05] | 3.04% |
| S^2 | -4.75e-11 | [-5.53e-11, -3.92e-11] | 2.71% |
| M^2 | 3.315e-10 | [2.85e-10, 4.01e-10] | 2.78% |
| sum l^2 | -2.763e-10 | [-3.54e-10, -2.29e-10] | 2.76% |
| (N-2)+ | -0.4382 | [-0.544, -0.32] | 2.67% |
| (N-4)+ | -1.046 | [-1.09, -0.996] | 3.80% |
| (N-64)+ | -0.04909 | [-0.0529, -0.0446] | 2.95% |
| (N-128)+ | -0.03065 | [-0.0348, -0.0268] | 2.74% |

Coefficients in ms per unit.

| term | linear | degree 2 | degree 2 + sum l^2 | selected |
|---|---|---|---|---|
| const | 8.507 | 8.443 | 8.405 | 8.405 |
| N | 1.569 | 1.549 | 1.596 | 1.596 |
| S | 2.254e-05 | 3.576e-05 | 3.766e-05 | 3.796e-05 |
| M | 6.746e-05 | 5.965e-05 | 5.685e-05 | 5.645e-05 |
| N^2 |  | 4.365e-05 | 5.176e-05 |  |
| N*S |  | 2.789e-08 | 7.199e-09 |  |
| N*M |  | 1.153e-10 | -1.139e-08 |  |
| S^2 |  | -5.348e-11 | -4.93e-11 | -4.75e-11 |
| S*M |  | -8.229e-11 | 2.741e-12 |  |
| M^2 |  | 1.354e-10 | 3.169e-10 | 3.315e-10 |
| sum l^2 |  |  | -2.645e-10 | -2.763e-10 |
| (N-2)+ | -0.3905 | -0.3427 | -0.3842 | -0.4382 |
| (N-4)+ | -1.092 | -1.139 | -1.16 | -1.046 |
| (N-8)+ | 0.05628 | 0.0732 | 0.08334 |  |
| (N-16)+ | -0.02101 | -0.02686 | -0.02474 |  |
| (N-32)+ | -0.0114 | -0.009248 | -0.006248 |  |
| (N-64)+ | -0.04413 | -0.04933 | -0.04953 | -0.04909 |
| (N-128)+ | -0.03333 | -0.04148 | -0.04233 | -0.03065 |

## hybrid: 1357 batches, samples spread by 0.2% (median)

With the floor; errors are on held-out batches.

| model | params | R^2 | held-out R^2 | held-out RMSE | mean error | median | p90 | max |
|---|---|---|---|---|---|---|---|---|
| linear | 8 | 0.8247 | 0.8157 | 181.0 ms | 56.9% | 31.7% | 125.5% | 647.1% |
| degree 2 | 29 | 0.9985 | 0.9983 | 17.2 ms | 3.2% | 2.0% | 6.6% | 47.2% |
| degree 2 + sum l^2 | 30 | 0.9986 | 0.9984 | 16.9 ms | 3.1% | 2.0% | 6.5% | 47.7% |
| selected | 11 | 0.9984 | 0.9983 | 17.6 ms | 3.0% | 2.1% | 5.8% | 43.2% |

With the terms also chosen inside each training fold (nested cross-validation), the selected model's held-out error is mean 3.0%, median 2.1%, p90 6.3%, max 29.6%.
- fold 0 chose 10 terms: +S_p^2
- fold 1 chose 10 terms: +S_p^2, +M_p^2, -sum l_p^2
- fold 2 chose 10 terms: +S_p^2
- fold 3 chose 10 terms: +S_p^2
- fold 4 chose 10 terms: +N_p*S_p

Held-out error by the step's prompts: mean, max.

| model | one prompt (327) | equal prompts (165) | unequal prompts (865) |
|---|---|---|---|
| linear | 107.8%, 647.1% | 57.2%, 348.8% | 37.6%, 367.2% |
| degree 2 | 4.4%, 47.2% | 3.9%, 18.2% | 2.5%, 30.5% |
| degree 2 + sum l^2 | 4.4%, 47.7% | 4.0%, 15.9% | 2.5%, 30.1% |
| selected | 4.6%, 43.2% | 3.4%, 14.9% | 2.4%, 17.3% |

The polynomial alone, without the floor.

| model | params | R^2 | held-out R^2 | held-out RMSE | mean error | median | p90 | max |
|---|---|---|---|---|---|---|---|---|
| linear | 7 | 0.7606 | 0.7527 | 209.7 ms | 49.5% | 30.1% | 100.0% | 512.8% |
| degree 2 | 28 | 0.9985 | 0.9983 | 17.2 ms | 3.2% | 2.0% | 6.6% | 47.2% |
| degree 2 + sum l^2 | 29 | 0.9986 | 0.9984 | 16.9 ms | 3.1% | 2.0% | 6.5% | 47.7% |

Backward elimination from degree 2 + sum l^2: held-out error after each drop. The selected model is the smallest with mean error at most 3.05%: the best, 2.95%, plus one standard error.

| params | dropped | mean error | max |
|---|---|---|---|
| 30 | none | 3.09% | 47.7% |
| 29 | N_d^2 | 3.03% | 41.7% |
| 28 | S_d^2 | 3.00% | 41.5% |
| 27 | N_p*S_d | 2.99% | 41.4% |
| 26 | S_p*M_d | 2.98% | 41.4% |
| 25 | S_p*S_d | 2.96% | 41.6% |
| 24 | N_p*M_d | 2.96% | 41.5% |
| 23 | S_d*M_d | 2.96% | 41.5% |
| 22 | M_p*M_d | 2.96% | 41.4% |
| 21 | N_p*N_d | 2.95% | 40.2% |
| 20 | N_p^2 | 2.95% | 40.1% |
| 19 | M_p^2 | 2.95% | 40.4% |
| 18 | S_p^2 | 2.96% | 40.0% |
| 17 | M_d^2 | 2.98% | 40.3% |
| 16 | N_d*M_d | 2.98% | 38.0% |
| 15 | S_p*N_d | 3.02% | 23.5% |
| 14 | N_d*S_d | 3.00% | 29.7% |
| 13 | N_p*S_p | 2.99% | 37.1% |
| 12 | S_d | 2.99% | 37.1% |
| 11 (selected) | M_p | 3.05% | 43.2% |
| 10 | sum l_p^2 | 3.29% | 28.1% |
| 9 | N_p | 3.39% | 27.8% |
| 8 | M_d | 3.79% | 40.6% |
| 7 | N_p*M_p | 4.57% | 39.7% |
| 6 | N_d | 7.01% | 78.8% |
| 5 | S_p*M_p | 10.00% | 131.0% |
| 4 | M_p*S_d | 21.49% | 141.8% |
| 3 | M_p*N_d | 60.23% | 207.3% |
| 2 | S_p | 178.15% | 1896.4% |

The selected model, in ms per unit.

| term | coefficient | 95% interval | held-out mean error without it |
|---|---|---|---|
| floor F | 0 | [0, 0] | |
| const | 25.67 | [24.5, 26.8] |  |
| N_p | 0.2273 | [0.168, 0.268] | 3.32% |
| S_p | 0.02965 | [0.0292, 0.0302] | 21.79% |
| N_d | 0.3098 | [0.266, 0.358] | 6.64% |
| M_d | 0.001014 | [0.000815, 0.00117] | 3.55% |
| N_p*M_p | 0.001018 | [0.000887, 0.00126] | 3.37% |
| S_p*M_p | 8.091e-06 | [7.05e-06, 9e-06] | 4.15% |
| M_p*N_d | 0.001465 | [0.00142, 0.0015] | 49.32% |
| M_p*S_d | 2.419e-06 | [2.39e-06, 2.45e-06] | 17.17% |
| sum l_p^2 | -4.719e-06 | [-5.57e-06, -3.78e-06] | 3.29% |

Coefficients in ms per unit.

| term | linear | degree 2 | degree 2 + sum l^2 | selected |
|---|---|---|---|---|
| floor F | 146.7 | 0 | 0 | 0 |
| const | -763.1 | 24.28 | 23.67 | 25.67 |
| N_p | -14.59 | 0.4439 | 0.4506 | 0.2273 |
| S_p | 0.1113 | 0.02788 | 0.03025 | 0.02965 |
| M_p | 0.1226 | 0.002785 | 0.001169 |  |
| N_d | 5.368 | 0.3521 | 0.3521 | 0.3098 |
| S_d | 0.004405 | -6.775e-05 | -5.418e-05 |  |
| M_d | 0.01655 | 0.001463 | 0.00154 | 0.001014 |
| N_p^2 |  | -0.003002 | 0.0001309 |  |
| N_p*S_p |  | 2.044e-05 | -0.000111 |  |
| N_p*M_p |  | 0.001267 | 0.001068 | 0.001018 |
| N_p*N_d |  | 0.001654 | 0.001574 |  |
| N_p*S_d |  | -1.004e-06 | -6.774e-07 |  |
| N_p*M_d |  | 4.962e-06 | 4.935e-06 |  |
| S_p^2 |  | -5.299e-07 | -3.059e-07 |  |
| S_p*M_p |  | 7.062e-06 | 8.278e-06 | 8.091e-06 |
| S_p*N_d |  | 2.956e-05 | 3.031e-05 |  |
| S_p*S_d |  | 1.602e-08 | 1.336e-08 |  |
| S_p*M_d |  | 4.746e-08 | 2.611e-08 |  |
| M_p^2 |  | -3.315e-06 | -5.759e-07 |  |
| M_p*N_d |  | 0.001436 | 0.001436 | 0.001465 |
| M_p*S_d |  | 2.399e-06 | 2.397e-06 | 2.419e-06 |
| M_p*M_d |  | -1.457e-09 | 2.872e-08 |  |
| N_d^2 |  | -0.0005553 | -0.0005606 |  |
| N_d*S_d |  | -6.915e-07 | -7.529e-07 |  |
| N_d*M_d |  | 2.122e-05 | 2.141e-05 |  |
| S_d^2 |  | 1.151e-10 | 1.583e-10 |  |
| S_d*M_d |  | 1.953e-08 | 1.543e-08 |  |
| M_d^2 |  | -1.114e-07 | -1.085e-07 |  |
| sum l_p^2 |  |  | -4.25e-06 | -4.719e-06 |

sum l_p^2 / S_p*M_p: -0.583 in the selected model (95% interval [-0.623, -0.525]), -0.513 in degree 2 + sum l^2

## Summary, held out, with the floor and bends

| | linear | degree 2 | degree 2 + sum l^2 | selected |
|---|---|---|---|---|
| prefill parameters | 5 | 11 | 12 | 8 |
| prefill held-out R^2 | 0.9605 | 0.9967 | 0.9969 | 0.9969 |
| prefill mean error | 11.1% | 2.8% | 2.7% | 2.7% |
| prefill max error | 137.4% | 30.7% | 32.1% | 35.7% |
| decode parameters | 11 | 17 | 18 | 11 |
| decode held-out R^2 | 0.9842 | 0.9875 | 0.9883 | 0.9883 |
| decode mean error | 2.9% | 2.7% | 2.6% | 2.6% |
| decode max error | 25.2% | 28.4% | 27.8% | 28.6% |
| hybrid parameters | 8 | 29 | 30 | 11 |
| hybrid held-out R^2 | 0.8157 | 0.9983 | 0.9984 | 0.9983 |
| hybrid mean error | 56.9% | 3.2% | 3.1% | 3.0% |
| hybrid max error | 647.1% | 47.2% | 47.7% | 43.2% |

## Summary, held out, the polynomial alone

| | linear | degree 2 | degree 2 + sum l^2 |
|---|---|---|---|
| prefill held-out R^2 | 0.9578 | 0.9967 | 0.9969 |
| prefill mean error | 12.1% | 2.8% | 2.7% |
| prefill max error | 100.0% | 30.7% | 32.1% |
| decode held-out R^2 | 0.9143 | 0.9660 | 0.9673 |
| decode mean error | 7.9% | 5.3% | 5.2% |
| decode max error | 39.2% | 28.8% | 27.9% |
| hybrid held-out R^2 | 0.7527 | 0.9983 | 0.9984 |
| hybrid mean error | 49.5% | 3.2% | 3.1% |
| hybrid max error | 512.8% | 47.2% | 47.7% |
