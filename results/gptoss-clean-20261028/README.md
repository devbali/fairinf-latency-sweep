# gpt-oss-20b on one A100: latency models

The latency sweep of 2026-10-28, and the models fitted to it.

- `results.jsonl`: 4,893 samples, one JSON line each (1,631 batches, each timed 3 times).
- `run.log`: the runner's output.

Every number in this file comes from running, in the repo root:

```bash
python compare_fits.py results/gptoss-clean-20261028/results.jsonl
```

## The run

| | |
|---|---|
| model | gpt-oss-20b in BF16 (`lmsys/gpt-oss-20b-bf16`) |
| GPU | one A100-SXM4-80GB, VM `fairinf-matteo-fork-1xa100-eu1` |
| SGLang | the FairInference fork, `matteo/fairbatching` at `4e84427ab` |
| packages | torch 2.11.0+cu130, sglang-kernel 0.4.2.post1, flashinfer 0.6.8.post1, triton 3.6.0, transformers 5.6.0, installed unmodified by `vm/fresh_venv.sh` |
| flags | `--dtype bfloat16 --mem-fraction-static 0.7 --sampling-backend pytorch --disable-piecewise-cuda-graph` |
| kernels | SGLang's defaults for gpt-oss on A100: Triton attention, `triton_kernel` MoE |
| batches | `batches.py`: 618 prefill, 565 decode and 448 hybrid. Each ran once as a warmup, then 3 times. |

A batch's latency is the median of its 3 samples, in ms.

## The models

The variables are the same as in `fit.py`:

| | prefill | decode | hybrid |
|---|---|---|---|
| N | requests | requests, padded up to the CUDA-graph batch size | N_p prompts, N_d decodes |
| S | total prompt tokens | total context tokens | S_p, S_d |
| M | longest prompt | longest context | M_p, M_d |
| l_i | length of prompt i | length of context i | prompt lengths in the prefill part |

Each model has a polynomial part f:

```
linear               f = c + a*N + b*S + d*M
degree 2             f = linear + e1*N^2 + e2*N*S + e3*N*M + e4*S^2 + e5*S*M + e6*M^2
degree 2 + sum l^2   f = degree 2 + q * sum_i l_i^2
selected             f = the terms of degree 2 + sum l^2 that matter most
```

- **Degree 2** is the full second-order polynomial: the terms of scikit-learn's
  `PolynomialFeatures(degree=2)`.
- **Hybrid steps** use all six variables, so degree 2 adds 21 squares and
  products. The extra term is `sum l_p^2`, over the prefill part's prompts.
- **Selected** comes from backward elimination. Starting from degree 2 +
  sum l^2, it drops one term at a time: each time, the one whose removal
  leaves the lowest held-out mean error. Of the models along the way, it keeps
  the smallest whose error is within one standard error of the best one's.
  That's the one-standard-error rule (Hastie, Tibshirani and Friedman, *The
  Elements of Statistical Learning*, 2nd ed., section 7.10). Every term can go
  except the constant and the floor, including decode's bends.

The selected terms, besides the constant and (prefill, hybrid) the floor:

| phase | terms kept |
|---|---|
| prefill | N, S, N*M, S*M, sum l^2 |
| decode | N, S, M, N^2, S^2, and the bends at N = 2, 4 and 64 |
| hybrid | S_p, N_d, M_d, N_p*M_p, S_p*M_p, M_p*N_d, M_p*S_d, S_p*N_d, N_d^2, sum l_p^2 |

All four models wrap f the same way:

```
prefill, hybrid:  T = max(F, f)
decode:           T = f + sum_k delta_k * (N - k)+       k = 2, 4, 8, ..., 128;  (x)+ = max(0, x)
```

- **Floor F.** Prefill and hybrid steps run without CUDA graphs. Below about 2k
  tokens, the CPU launching kernels sets the step time. As in `fit.py`, f is
  fitted to the steps above the floor and F to the rest, alternating until no
  step changes side.
- **Bends.** Decode runs as CUDA graphs and has no floor, but each added
  request costs less as the batch grows. The hinge terms let the slope in N
  change at each power of two. The selected decode model keeps three of them.

Every model is linear in its coefficients and fitted by least squares.

## Accuracy

All errors here are held out: each batch is predicted by a model fitted without
it, using 5-fold cross-validation with `fit.py`'s folds. Error is
|predicted - measured| / measured. R^2 is dominated by the largest steps, so
the relative errors are the better guide.

| | linear | degree 2 | degree 2 + sum l^2 | selected | fit.py's model |
|---|---|---|---|---|---|
| prefill parameters | 5 | 11 | 12 | 7 | 8 |
| prefill held-out R^2 | 0.9716 | 0.9982 | 0.9988 | 0.9988 | 0.9988 |
| prefill mean error | 6.2% | 1.6% | 1.3% | 1.3% | 1.3% |
| prefill max error | 29.4% | 9.0% | 6.8% | 7.2% | 7.3% |
| decode parameters | 11 | 17 | 18 | 9 | 11 |
| decode held-out R^2 | 0.9841 | 0.9855 | 0.9863 | 0.9857 | 0.9841 |
| decode mean error | 2.9% | 2.7% | 2.7% | 2.7% | 2.9% |
| decode max error | 22.0% | 21.9% | 21.9% | 21.0% | 22.0% |
| hybrid parameters | 8 | 29 | 30 | 12 | 13 |
| hybrid held-out R^2 | 0.7148 | 0.9990 | 0.9990 | 0.9991 | 0.9991 |
| hybrid mean error | 43.8% | 2.3% | 2.2% | 2.0% | 2.1% |
| hybrid max error | 463.5% | 32.2% | 28.6% | 13.9% | 14.3% |

`fit.py`'s model is the one the repo's `fit.py` fits, for reference; for
decode it's the linear one. A batch's 3 samples spread by 0.5% (prefill), 2.8%
(decode) and 0.3% (hybrid), median. An error near that level is measurement
noise.

The first three polynomials alone, without the floor (prefill, hybrid) or the
bends (decode):

| | linear | degree 2 | degree 2 + sum l^2 |
|---|---|---|---|
| prefill held-out R^2 | 0.9517 | 0.9952 | 0.9959 |
| prefill mean error | 12.6% | 4.7% | 4.5% |
| prefill max error | 84.1% | 25.2% | 25.1% |
| decode held-out R^2 | 0.8987 | 0.9628 | 0.9639 |
| decode mean error | 8.6% | 5.6% | 5.5% |
| decode max error | 41.5% | 29.0% | 29.1% |
| hybrid held-out R^2 | 0.6908 | 0.9988 | 0.9989 |
| hybrid mean error | 45.9% | 3.6% | 3.5% |
| hybrid max error | 442.6% | 36.2% | 34.8% |

- **Linear regression isn't enough.** It misses prefill steps by 6.2% and
  hybrid steps by 44% on average: attention cost isn't linear in N, S and M.
- **Degree 2 fixes most of that.** Adding sum l^2 helps prefill further: mean
  error falls from 1.6% to 1.3%, and the worst case from 9.0% to 6.8%.
- **The selected model is as accurate with far fewer terms:** 7 parameters
  instead of 12 for prefill, 9 instead of 18 for decode, and 12 instead of 30
  for hybrid. On hybrid it's better. The 18 terms it drops make predictions
  worse on unseen batches, and without them the worst error falls from 28.6%
  to 13.9%.
- **The selection doesn't overfit.** Choosing the terms with the same folds
  that score them could flatter the selected model. So the whole selection was
  repeated inside each training fold (nested cross-validation). That gives
  mean errors of 1.3%, 2.7% and 2.1%, within 0.1 points of the table, and worst
  cases of 7.2%, 21.6% and 16.1%.
- **Decode is at the noise level with any of the four.** What matters there
  are the bends; without them the error is 5.5 to 8.6%.

## Where sum l^2 comes from

In short: attention is quadratic in each request's own length, not in the
batch's. A step's attention work is a sum of per-request squares, sum l^2. No
polynomial in N, S and M can express that, so the model gets it as one extra
term.

### Causal attention is quadratic per request

In causal attention, token t of a prompt attends to the t tokens up to and
including itself. A prompt of l tokens therefore scores

```
1 + 2 + ... + l = l(l + 1)/2
```

query-key pairs. Requests in a batch never attend to each other, so a prefill
step scores

```
sum_i l_i(l_i + 1)/2 = (sum l^2 + S)/2
```

pairs, where sum l^2 = sum_i l_i^2.

### N, S and M don't determine it

These two steps have the same N, S and M but different attention work:

| prompt lengths | N | S | M | sum l^2 |
|---|---|---|---|---|
| 1000, 500, 500 | 3 | 2,000 | 1,000 | 1,500,000 |
| 1000, 999, 1 | 3 | 2,000 | 1,000 | 1,998,002 |

So no polynomial in N, S and M, of any degree, matches attention cost on every
step. The degree-2 terms match sum l^2 only in special cases: it equals
S^2 = M^2 for a single prompt, and S*M for N equal prompts. In general,
S^2 = sum_i sum_j l_i*l_j also counts pairs of tokens from different requests,
which attention never computes.

### SGLang's kernel pads every request to the longest

On A100, SGLang runs gpt-oss's attention with its Triton kernel
`extend_attention_fwd`, in
`python/sglang/srt/layers/attention/triton_ops/extend_attention.py` at
`4e84427ab`:

- **One grid for the whole step.** It launches `(requests, heads, ceil(M / 128))`
  programs, each covering 128 query rows (line 609). Every request gets
  ceil(M / 128) blocks, sized by the step's longest prompt rather than its own
  length.
- **No early exit.** Block b of a request loops over its keys up to
  `min(l, 128 * (b + 1))` (lines 433-438). A block that starts past the
  request's last token still loops over all l keys, with every row masked.

So each request costs as if it had M query rows, each reading all l of its
keys. The exception is its real rows: the causal mask spares them the upper
half of their triangle.

```
l*M - l^2/2  =  l^2/2         its causal triangle
              + l*(M - l)     the masked blocks past its end
```

Summed over the step:

```
sum_i (l_i*M - l_i^2/2)  =  S*M - 0.5 * sum l^2
```

There's also a fixed cost for each of the step's N * ceil(M / 128) blocks: the
N*M term. N*M and S*M are already in the degree-2 polynomial. Adding sum l^2
completes the kernel's cost, and predicts its coefficient: -0.5 times that of
S*M.

gpt-oss alternates full-attention layers with 128-token sliding-window layers.
In a prefill step both kinds do this work, because the kernel applies the
window as a mask inside the same loop (lines 468-473).

### Hybrid and decode steps

- **Hybrid.** The prefill part pays the same S_p*M_p - 0.5 * sum l_p^2. Each
  decode in the step is a 1-token prompt with a long cached context. It also
  gets ceil(M_p / 128) blocks, and in full-attention layers each block first
  loops over that whole context (line 337). This adds S_d*M_p, plus N_d*M_p for
  the blocks: the M_p*S_d and M_p*N_d terms of the degree-2 polynomial. The
  linear model has neither.
- **Decode.** Each request computes one new token, which reads its cached
  context once, so attention costs about S in total. Nothing is quadratic, and
  the selection drops sum l^2 from the decode model.

### What the fit shows

- **Backward elimination keeps the kernel's terms.**
  - Prefill keeps N*M, S*M and sum l^2. It drops M, N^2, N*S, S^2 and M^2 with
    no loss in accuracy, and every training fold of the nested
    cross-validation chose the same terms.
  - Hybrid keeps all five of the kernel's terms: N_p*M_p, S_p*M_p, sum l_p^2,
    M_p*N_d and M_p*S_d. So did every training fold. Of the other 17
    second-order terms it keeps two, S_p*N_d and N_d^2. They matter least of
    the terms it keeps, and the folds disagree about them.
  - `fit.py`'s model, derived from the kernel by hand, is nearly the same. For
    prefill the selected terms are `fit.py`'s minus M.
- **sum l^2 earns its place.** Dropping it from the selected model raises the
  held-out mean error from 1.3% to 1.9% for prefill, and from 2.0% to 2.2% for
  hybrid.
- **Its coefficient comes out close to the prediction.** The ratio of
  sum l^2's coefficient to S*M's is -0.45 in the selected prefill model (95%
  interval -0.49 to -0.42). For hybrid, sum l_p^2 to S_p*M_p is -0.52 (-0.58
  to -0.45). The kernel predicts -0.5.
- **sum l^2 helps exactly where N, S and M fall short.** Held-out prefill
  error by the step's prompts (mean, max):

| model | one prompt (54) | equal prompts (39) | unequal prompts (525) |
|---|---|---|---|
| linear | 9.0%, 20.1% | 8.4%, 29.4% | 5.8%, 29.0% |
| degree 2 | 1.3%, 4.8% | 2.7%, 9.0% | 1.6%, 8.2% |
| degree 2 + sum l^2 | 1.3%, 4.8% | 2.2%, 3.8% | 1.3%, 6.8% |
| selected | 1.2%, 4.8% | 2.2%, 4.2% | 1.3%, 7.2% |

With one prompt, sum l^2 = S^2 = M^2 is already in the polynomial, so the extra
term changes nothing. With several prompts, no polynomial term tracks it across
all steps, and adding it lowers the error.

## Coefficients and full accuracy, by phase

Coefficients are in ms per unit. N counts requests; S, M and l count tokens.
So S*M is in ms per token^2.

- **R^2** is measured on the batches the model was fitted on. The held-out
  columns use cross-validation, as above.
- **The elimination path** shows the held-out error after each term is dropped,
  starting from degree 2 + sum l^2.
- **Intervals** are bootstrap 95% intervals for the selected model, from 200
  refits on resampled batches. An interval that contains zero marks a term the
  data can't pin down.
- **Error without it** is the held-out mean error of the selected model refitted
  without that one term: how much the term matters.

### Prefill: 618 batches

| model | params | R^2 | held-out R^2 | held-out RMSE | mean error | median | p90 | max |
|---|---|---|---|---|---|---|---|---|
| linear | 5 | 0.9724 | 0.9716 | 18.7 ms | 6.2% | 4.7% | 14.3% | 29.4% |
| degree 2 | 11 | 0.9984 | 0.9982 | 4.8 ms | 1.6% | 1.3% | 3.4% | 9.0% |
| degree 2 + sum l^2 | 12 | 0.9989 | 0.9988 | 3.8 ms | 1.3% | 1.1% | 2.8% | 6.8% |
| selected | 7 | 0.9989 | 0.9988 | 3.8 ms | 1.3% | 1.0% | 2.8% | 7.2% |
| fit.py's model | 8 | 0.9989 | 0.9988 | 3.8 ms | 1.3% | 1.1% | 2.8% | 7.3% |

With the terms also chosen inside each training fold, the selected model's
held-out error is mean 1.3%, median 1.0%, p90 2.8%, max 7.2%. All five folds
chose the same terms.

The elimination path. The selected model is the smallest with mean error at
most 1.37%: the best, 1.32%, plus one standard error.

| params | dropped | mean error | max |
|---|---|---|---|
| 12 | none | 1.34% | 6.8% |
| 11 | M^2 | 1.34% | 6.8% |
| 10 | N*S | 1.33% | 7.2% |
| 9 | S^2 | 1.33% | 7.3% |
| 8 | M | 1.32% | 7.2% |
| 7 (selected) | N^2 | 1.32% | 7.2% |
| 6 | N | 1.53% | 10.8% |
| 5 | sum l^2 | 1.94% | 10.7% |
| 4 | N*M | 5.59% | 41.0% |
| 3 | S*M | 9.95% | 43.2% |
| 2 | S | 70.02% | 195.8% |

The selected model:

| term | coefficient | 95% interval | held-out mean error without it |
|---|---|---|---|
| floor F | 70.13 | [69.8, 70.4] | |
| const | 22.37 | [21.8, 23.1] |  |
| N | 0.128 | [0.096, 0.166] | 1.53% |
| S | 0.02917 | [0.0289, 0.0294] | 21.87% |
| N*M | 0.0008816 | [0.000789, 0.000947] | 3.65% |
| S*M | 5.627e-06 | [5.28e-06, 6.06e-06] | 3.20% |
| sum l^2 | -2.554e-06 | [-2.98e-06, -2.21e-06] | 1.89% |

All four models:

| term | linear | degree 2 | degree 2 + sum l^2 | selected |
|---|---|---|---|---|
| floor F | 72.43 | 70.16 | 70.13 | 70.13 |
| const | -24.64 | 18.7 | 20.09 | 22.37 |
| N | 1.067 | 0.4704 | 0.2439 | 0.128 |
| S | 0.03662 | 0.02838 | 0.02921 | 0.02917 |
| M | 0.02562 | 0.002347 | 0.001326 |  |
| N^2 |  | -0.006219 | -0.0005501 |  |
| N*S |  | 3.559e-05 | -1.693e-05 |  |
| N*M |  | 0.0008766 | 0.0008811 | 0.0008816 |
| S^2 |  | -1.235e-07 | 5.251e-08 |  |
| S*M |  | 4.845e-06 | 5.419e-06 | 5.627e-06 |
| M^2 |  | -1.795e-06 | -3.933e-08 |  |
| sum l^2 |  |  | -2.514e-06 | -2.554e-06 |

### Decode: 565 batches

| model | params | R^2 | held-out R^2 | held-out RMSE | mean error | median | p90 | max |
|---|---|---|---|---|---|---|---|---|
| linear | 11 | 0.9850 | 0.9841 | 0.8 ms | 2.9% | 2.1% | 6.1% | 22.0% |
| degree 2 | 17 | 0.9870 | 0.9855 | 0.7 ms | 2.7% | 2.0% | 5.9% | 21.9% |
| degree 2 + sum l^2 | 18 | 0.9881 | 0.9863 | 0.7 ms | 2.7% | 2.0% | 5.8% | 21.9% |
| selected | 9 | 0.9865 | 0.9857 | 0.7 ms | 2.7% | 2.0% | 5.8% | 21.0% |

With the terms also chosen inside each training fold, the selected model's
held-out error is mean 2.7%, median 2.1%, p90 5.8%, max 21.6%. Four folds chose
the same terms; one chose (N-128)+ instead of N^2.

The elimination path. The selected model is the smallest with mean error at
most 2.75%: the best, 2.64%, plus one standard error.

| params | dropped | mean error | max |
|---|---|---|---|
| 18 | none | 2.67% | 21.9% |
| 17 | (N-8)+ | 2.66% | 20.7% |
| 16 | (N-128)+ | 2.66% | 20.7% |
| 15 | (N-32)+ | 2.65% | 20.6% |
| 14 | (N-16)+ | 2.64% | 20.6% |
| 13 | S*M | 2.64% | 20.7% |
| 12 | N*M | 2.64% | 20.8% |
| 11 | N*S | 2.68% | 20.7% |
| 10 | sum l^2 | 2.69% | 20.9% |
| 9 (selected) | M^2 | 2.70% | 21.0% |
| 8 | (N-64)+ | 2.79% | 21.7% |
| 7 | (N-2)+ | 2.87% | 21.8% |
| 6 | S^2 | 2.95% | 22.6% |
| 5 | S | 4.52% | 23.2% |
| 4 | N^2 | 6.91% | 24.8% |
| 3 | (N-4)+ | 10.45% | 45.3% |
| 2 | M | 14.71% | 62.1% |
| 1 | N | 28.72% | 99.0% |

The selected model:

| term | coefficient | 95% interval | held-out mean error without it |
|---|---|---|---|
| const | 8.408 | [8.27, 8.55] |  |
| N | 1.577 | [1.47, 1.69] | 3.43% |
| S | 3.15e-05 | [2.75e-05, 3.6e-05] | 3.48% |
| M | 5.81e-05 | [5.39e-05, 6.24e-05] | 4.33% |
| N^2 | -0.0001596 | [-0.000196, -0.000123] | 2.87% |
| S^2 | -4.203e-11 | [-5.86e-11, -2.83e-11] | 2.81% |
| (N-2)+ | -0.429 | [-0.606, -0.243] | 2.80% |
| (N-4)+ | -1.019 | [-1.11, -0.938] | 4.04% |
| (N-64)+ | -0.03847 | [-0.0504, -0.027] | 2.79% |

All four models:

| term | linear | degree 2 | degree 2 + sum l^2 | selected |
|---|---|---|---|---|
| const | 8.44 | 8.496 | 8.444 | 8.408 |
| N | 1.614 | 1.554 | 1.608 | 1.577 |
| S | 2.058e-05 | 3.449e-05 | 3.713e-05 | 3.15e-05 |
| M | 6.229e-05 | 4.31e-05 | 4.034e-05 | 5.81e-05 |
| N^2 |  | -0.0001568 | -0.0001425 | -0.0001596 |
| N*S |  | -9.708e-10 | -2.123e-08 |  |
| N*M |  | -3.053e-08 | -5.543e-08 |  |
| S^2 |  | -5.134e-11 | -4.981e-11 | -4.203e-11 |
| S*M |  | 1.117e-11 | 1.178e-10 |  |
| M^2 |  | 1.173e-10 | 3.464e-10 |  |
| sum l^2 |  |  | -3.36e-10 |  |
| (N-2)+ | -0.4579 | -0.4028 | -0.4452 | -0.429 |
| (N-4)+ | -1.056 | -1.059 | -1.095 | -1.019 |
| (N-8)+ | 0.0491 | 0.0608 | 0.07911 |  |
| (N-16)+ | -0.01457 | -0.019 | -0.01578 |  |
| (N-32)+ | -0.02912 | -0.01227 | -0.01019 |  |
| (N-64)+ | -0.03761 | -0.03106 | -0.03239 | -0.03847 |
| (N-128)+ | -0.03746 | -8.528e-05 | -0.001832 |  |

A decode step's cost per request is the N coefficient plus the bends below N.
In the linear model that is 1.6 ms per request up to 2 requests, 1.2 ms up to
4, then 0.03 to 0.15 ms.

### Hybrid: 448 batches

| model | params | R^2 | held-out R^2 | held-out RMSE | mean error | median | p90 | max |
|---|---|---|---|---|---|---|---|---|
| linear | 8 | 0.7934 | 0.7148 | 225.9 ms | 43.8% | 29.6% | 110.1% | 463.5% |
| degree 2 | 29 | 0.9993 | 0.9990 | 13.6 ms | 2.3% | 1.6% | 4.9% | 32.2% |
| degree 2 + sum l^2 | 30 | 0.9993 | 0.9990 | 13.1 ms | 2.2% | 1.5% | 4.7% | 28.6% |
| selected | 12 | 0.9992 | 0.9991 | 12.4 ms | 2.0% | 1.4% | 4.0% | 13.9% |
| fit.py's model | 13 | 0.9992 | 0.9991 | 12.7 ms | 2.1% | 1.6% | 4.3% | 14.3% |

Held-out error by the prefill part's prompts (mean, max):

| model | one prompt (82) | equal prompts (24) | unequal prompts (342) |
|---|---|---|---|
| linear | 65.1%, 463.5% | 69.1%, 398.6% | 36.9%, 200.1% |
| degree 2 | 2.6%, 32.2% | 4.3%, 12.0% | 2.1%, 19.5% |
| degree 2 + sum l^2 | 2.5%, 28.6% | 4.1%, 11.4% | 2.0%, 15.1% |
| selected | 2.2%, 8.7% | 3.2%, 10.4% | 1.9%, 13.9% |
| fit.py's model | 2.1%, 9.4% | 3.2%, 7.9% | 2.0%, 14.3% |

With the terms also chosen inside each training fold, the selected model's
held-out error is mean 2.1%, median 1.4%, p90 4.4%, max 16.1%. The folds chose
10 to 14 terms. All kept the kernel's five; they differ only on smaller terms
that involve the decode part, such as S_d, M_d and S_d*M_d.

The elimination path. The selected model is the smallest with mean error at
most 2.02%: the best, 1.93%, plus one standard error. Dropping the first 14
terms lowers the error.

| params | dropped | mean error | max |
|---|---|---|---|
| 30 | none | 2.21% | 28.6% |
| 29 | S_p*S_d | 2.17% | 18.8% |
| 28 | N_p*S_d | 2.12% | 16.8% |
| 27 | N_p*N_d | 2.08% | 16.8% |
| 26 | N_p^2 | 2.06% | 16.8% |
| 25 | M_d^2 | 2.04% | 15.0% |
| 24 | N_p*S_p | 2.02% | 14.8% |
| 23 | S_d^2 | 2.01% | 16.5% |
| 22 | N_d*M_d | 1.99% | 13.5% |
| 21 | M_p | 1.96% | 13.4% |
| 20 | S_p*M_d | 1.95% | 13.4% |
| 19 | M_p^2 | 1.94% | 13.3% |
| 18 | S_p^2 | 1.93% | 13.3% |
| 17 | N_p | 1.93% | 13.3% |
| 16 | N_p*M_d | 1.93% | 13.3% |
| 15 | N_d*S_d | 1.94% | 12.8% |
| 14 | S_d*M_d | 1.95% | 13.1% |
| 13 | M_p*M_d | 1.98% | 13.9% |
| 12 (selected) | S_d | 2.00% | 13.9% |
| 11 | N_d^2 | 2.07% | 14.1% |
| 10 | S_p*N_d | 2.10% | 13.9% |
| 9 | M_d | 2.34% | 14.5% |
| 8 | sum l_p^2 | 2.60% | 17.2% |
| 7 | N_p*M_p | 3.61% | 19.3% |
| 6 | N_d | 4.62% | 27.3% |
| 5 | S_p*M_p | 7.62% | 33.1% |
| 4 | M_p*S_d | 19.93% | 71.3% |
| 3 | S_p | 59.28% | 244.5% |
| 2 | M_p*N_d | 132.86% | 505.4% |

The selected model:

| term | coefficient | 95% interval | held-out mean error without it |
|---|---|---|---|
| floor F | 71.15 | [70.6, 72.2] | |
| const | 25.5 | [22.3, 29.1] |  |
| S_p | 0.02765 | [0.0264, 0.0288] | 12.97% |
| N_d | 0.2354 | [0.13, 0.334] | 2.66% |
| M_d | 0.001107 | [0.000715, 0.00142] | 2.31% |
| N_p*M_p | 0.001082 | [0.000917, 0.00138] | 2.43% |
| S_p*M_p | 7.258e-06 | [6.2e-06, 8.13e-06] | 2.95% |
| S_p*N_d | 2.946e-05 | [5.07e-06, 5.77e-05] | 2.10% |
| M_p*N_d | 0.001473 | [0.00143, 0.00151] | 26.21% |
| M_p*S_d | 2.402e-06 | [2.36e-06, 2.47e-06] | 16.09% |
| N_d^2 | -0.0003472 | [-0.00085, 0.000206] | 2.07% |
| sum l_p^2 | -3.805e-06 | [-4.65e-06, -2.83e-06] | 2.22% |

All four models:

| term | linear | degree 2 | degree 2 + sum l^2 | selected |
|---|---|---|---|---|
| floor F | 155.9 | 71.18 | 71.18 | 71.15 |
| const | -643.5 | 24.81 | 24.44 | 25.5 |
| N_p | -14.42 | 1.119 | 0.5919 |  |
| S_p | 0.09809 | 0.02264 | 0.02679 | 0.02765 |
| M_p | 0.1148 | 0.005026 | 0.001336 |  |
| N_d | 5.093 | 0.222 | 0.2359 | 0.2354 |
| S_d | 0.003362 | -0.0001804 | -0.000201 |  |
| M_d | 0.01618 | 0.00182 | 0.00195 | 0.001107 |
| N_p^2 |  | -0.08514 | -0.03265 |  |
| N_p*S_p |  | 0.0005857 | 0.0001293 |  |
| N_p*M_p |  | 0.0004734 | 0.0007782 | 0.001082 |
| N_p*N_d |  | 0.002665 | 0.00297 |  |
| N_p*S_d |  | -3.882e-06 | -3.922e-06 |  |
| N_p*M_d |  | -2.509e-05 | -2.691e-06 |  |
| S_p^2 |  | -5.518e-07 | -8.133e-08 |  |
| S_p*M_p |  | 7.461e-06 | 7.487e-06 | 7.258e-06 |
| S_p*N_d |  | 1.825e-05 | 1.933e-05 | 2.946e-05 |
| S_p*S_d |  | 4.1e-08 | 3.577e-08 |  |
| S_p*M_d |  | 1.089e-07 | 5.89e-08 |  |
| M_p^2 |  | -3.4e-06 | -1.955e-07 |  |
| M_p*N_d |  | 0.001478 | 0.001478 | 0.001473 |
| M_p*S_d |  | 2.378e-06 | 2.381e-06 | 2.402e-06 |
| M_p*M_d |  | -1.16e-07 | -6.198e-08 |  |
| N_d^2 |  | -0.0003432 | -0.0004107 | -0.0003472 |
| N_d*S_d |  | 4.893e-07 | 5.389e-07 |  |
| N_d*M_d |  | 9.532e-06 | 7.983e-06 |  |
| S_d^2 |  | -1.554e-10 | -8.263e-11 |  |
| S_d*M_d |  | 3.13e-08 | 3.232e-08 |  |
| M_d^2 |  | -1.302e-07 | -1.401e-07 |  |
| sum l_p^2 |  |  | -3.75e-06 | -3.805e-06 |

The linear model fits hybrid steps so badly that its fitted floor lands at
156 ms, against 71 ms in the others.
