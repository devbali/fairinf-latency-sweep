# The models

Every model this repo fits on its runs with CUDA graphs, as
`src/compare_fits.py` prints it: four polynomial regressions for each kind of
step. `python src/compare_fits.py results/<run>/results.jsonl` prints every
table below again, along with the elimination path, the nested
cross-validation and the bootstrap intervals; each run's README has that full
output.

## The four regressions

Each is a polynomial f in a step's N, S and M:

| | N | S | M |
|---|---|---|---|
| prefill | requests | total prompt tokens | longest prompt |
| decode | requests, padded to the CUDA-graph size | total context tokens | longest context |
| hybrid | `N_p`, `N_d` | `S_p`, `S_d` | `M_p`, `M_d` |
| transfer | requests of the prefill step | their prompt tokens | longest prompt |

```
linear               f = c + a*N + b*S + d*M
degree 2             f = linear + every square and product of N, S and M
degree 2 + sum l^2   f = degree 2 + q * sum l^2
selected             f = the terms of degree 2 + sum l^2 that matter most
```

`_p` is the prefill part of a hybrid step and `_d` its decode part, sum l^2
the sum of the squared prompt lengths, and `(x)+` means `max(0, x)`. The
selected model comes from backward elimination with the one-standard-error
rule. The time T of a step, in ms, is:

```
prefill, hybrid:  T = max(F, f)
decode:           T = f + sum_k delta_k * (N - k)+        k = 2, 4, 8, ..., 128
transfer:         T = f
```

- **Prefill and hybrid steps can have a floor F:** without CUDA graphs, the
  CPU launching kernels sets the time of steps under about 1k tokens. With
  piecewise CUDA graphs there is none, and the fits put F at 0 for every model
  but some of the linear ones.
- **Decode bends at each power of two in N,** as each of the first requests
  adds the most. These hinge terms are not polynomial; without them, decode's
  held-out error roughly doubles (the second table below).
- **Transfer** is the KV cache transfer of PD disaggregation: from a prefill
  step's first result to the last of its requests' KV caches reaching the
  decode server. Steps of the same prompt lengths count as samples of one
  batch. SGLang turns piecewise CUDA graphs off in PD mode, so the prefill
  server ran its steps eagerly; decode ran as CUDA graphs.
- **Coefficients** are in ms per unit: per request for N, per token for S and
  M, per token squared for sum l^2 and the squares.
- **Held out** means 5-fold cross-validation: each batch is predicted by a
  model fitted without it.

## Held-out mean error, every setup

| setup | phase | linear | degree 2 | degree 2 + sum l^2 | selected | parameters (selected) |
|---|---|---|---|---|---|---|
| gpt-oss-20b, one A100 | prefill | 11.1% | 2.8% | 2.7% | 2.7% | 8 |
| gpt-oss-20b, one A100 | decode | 2.9% | 2.7% | 2.6% | 2.6% | 11 |
| gpt-oss-20b, one A100 | hybrid | 56.9% | 3.2% | 3.1% | 3.0% | 11 |
| gpt-oss-120b, four A100s with tensor parallelism | prefill | 5.8% | 4.6% | 4.6% | 4.6% | 10 |
| gpt-oss-120b, four A100s with tensor parallelism | decode | 3.8% | 3.4% | 3.4% | 3.4% | 14 |
| gpt-oss-120b, four A100s with tensor parallelism | hybrid | 37.7% | 4.2% | 4.1% | 4.1% | 14 |
| PD KV transfer, gpt-oss-20b, two A100s, NIXL over NVLink | transfer | 28.6% | 28.9% | 28.6% | 28.6% | 4 |

The polynomial alone, without the floor and the bends:

| setup | phase | linear | degree 2 | degree 2 + sum l^2 |
|---|---|---|---|---|
| gpt-oss-20b, one A100 | prefill | 12.1% | 2.8% | 2.7% |
| gpt-oss-20b, one A100 | decode | 7.9% | 5.3% | 5.2% |
| gpt-oss-20b, one A100 | hybrid | 49.5% | 3.2% | 3.1% |
| gpt-oss-120b, four A100s with tensor parallelism | prefill | 5.8% | 4.6% | 4.6% |
| gpt-oss-120b, four A100s with tensor parallelism | decode | 9.2% | 6.6% | 6.6% |
| gpt-oss-120b, four A100s with tensor parallelism | hybrid | 33.1% | 4.2% | 4.1% |

## gpt-oss-20b, one A100

`results/gptoss-cuda-graphs-20261029`: `python src/compare_fits.py results/gptoss-cuda-graphs-20261029/results.jsonl`. Held out:

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

### prefill: 2,122 batches

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

### decode: 1,985 batches

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

### hybrid: 1,357 batches

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

## gpt-oss-120b, four A100s with tensor parallelism

`results/gptoss-120b-tp4-20261103`: `python src/compare_fits.py results/gptoss-120b-tp4-20261103/results.jsonl`. Held out:

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

### prefill: 2,122 batches

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

### decode: 1,985 batches

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

### hybrid: 1,357 batches

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

## PD KV transfer, gpt-oss-20b, two A100s, NIXL over NVLink

`results/gptoss-pd-nixl-20261103`: `python src/compare_fits.py results/gptoss-pd-nixl-20261103/results.jsonl.gz`. Held out:

| | linear | degree 2 | degree 2 + sum l^2 | selected |
|---|---|---|---|---|
| transfer parameters | 4 | 10 | 11 | 4 |
| transfer held-out R^2 | 0.9739 | 0.9729 | 0.9731 | 0.9739 |
| transfer mean error | 28.6% | 28.9% | 28.6% | 28.6% |
| transfer max error | 173.1% | 474.9% | 469.5% | 173.1% |

### transfer: 938 batches

| term | linear | degree 2 | degree 2 + sum l^2 | selected |
|---|---|---|---|---|
| const | 7.919 | 8.344 | 8.392 | 7.919 |
| N | 2.111 | 2.069 | 2.052 | 2.111 |
| S | 0.0007932 | 0.00106 | 0.001662 | 0.0007932 |
| M | -0.001005 | -0.001933 | -0.002577 | -0.001005 |
| N^2 |  | 0.0001977 | 0.0003914 |  |
| N*S |  | 1.149e-06 | -7.393e-06 |  |
| N*M |  | 4.248e-05 | 1.192e-05 |  |
| S^2 |  | -9.04e-08 | -6.106e-08 |  |
| S*M |  | 3.49e-07 | 8.945e-07 |  |
| M^2 |  | -2.31e-07 | 1.003e-06 |  |
| sum l^2 |  |  | -1.806e-06 |  |
