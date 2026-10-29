# gpt-oss-20b on one A100: sanity check

A short sweep that checks whether a VM still runs with the setup the latency
models were fitted on. It times 88 batches of the full sweep in
`results/gptoss-clean-20261028` again and compares the two runs.

- `results.jsonl`: 264 samples (88 batches, each timed 3 times).
- `run.log`: the runner's output.

Every number below comes from running, in the repo root:

```bash
python compare_runs.py results/gptoss-clean-20261028/results.jsonl results/gptoss-sanity-check-20261029/results.jsonl
```

## The run

| | |
|---|---|
| date | 2026-10-29 |
| VM | `fairinf-matteo-fork-1xa100-eu1`, one A100-SXM4-80GB: the full sweep's VM |
| code | `runner.py` byte for byte the full sweep's, SGLang `4e84427ab`, the same venv |
| command | `bash vm/run_clean.sh sanity_batches.py OUT` on the VM, with no extra flags |
| flags | `--dtype bfloat16 --mem-fraction-static 0.7 --sampling-backend pytorch --disable-piecewise-cuda-graph`, the defaults of `vm/run_clean.sh` |
| batches | `sanity_batches.py`: 20 prefill, 20 decode and 48 hybrid, all from `batches.py`. Each ran once as a warmup, then 3 times. |
| time | about 4 minutes: 2 to load the model, then 1.5 to time the batches |

## Result: the setup matches

- **The setup lines of the two logs are identical.** They give the SGLang
  copy, the venv's sgl_kernel, the flashinfer, transformers, torch and CUDA
  versions, the GPU, the MoE kernel, the KV cache sizes, the CUDA-graph batch
  sizes, and piecewise CUDA graphs being off. The KV cache sizes follow from
  `--mem-fraction-static`, so they pin that flag too.
- **The times agree to within the noise between samples,** with no shift in
  any phase:

  | phase | batches | mean difference | mean abs difference | sample spread, median |
  |---|---|---|---|---|
  | prefill | 20 | -0.1% | 0.6% | 0.4-0.6% |
  | decode | 20 | -0.1% | 2.3% | 2.0-2.8% |
  | hybrid | 48 | -0.5% | 0.6% | 0.3% |

- **The largest differences are on noisy batches.**
  - Decode 16x16384 was 9.8% slower; its 3 samples in the reference already
    spread by 6.9%.
  - Steps on the CPU floor, which depend on Python and kernel launches rather
    than the GPU, differ by up to 3.0% (prefill) and 4.4% (hybrid).
  - Prefill and hybrid steps above the floor differ by 0.3% on average and
    by at most 1.5%.
- **fit.py's model predicts this run as well as the reference.** Its mean
  error on these batches is 3.6% (prefill), 5.3% (decode) and 3.3% (hybrid)
  for this run, against 3.5%, 5.3% and 2.8% for the reference. That is more
  than its error over the whole reference (1.3%, 2.8% and 2.0%), because these
  are structured batches at the edges of the design, such as single prompts
  on the floor, rather than random ones.

A different setup shows up clearly. In earlier runs on this VM:
- In `hooks-venv`, with older packages, every step on the CPU floor took about
  19 ms longer (a floor of 89 ms instead of 70), while the GPU-side terms
  matched.
- With `--moe-runner-backend triton` instead of the default `triton_kernel`,
  an 8,192-token prefill took 655 ms instead of about 470.

## Full comparison

Every setup line in the run logs matches the reference:

```
sglang       /home/devbali/latency-test/fairinf-sglang-4e84427ab/python/sglang/__init__.py
sgl_kernel   /home/devbali/latency-test/venv/lib/python3.10/site-packages/sgl_kernel/__init__.py
flashinfer   0.6.8.post1
transformers 5.6.0
torch        2.11.0+cu130 cuda 13.0
gpu          NVIDIA A100-SXM4-80GB
Detected GPT-OSS model, enabling triton_kernels MOE kernel.
Load weight end. type=GptOssForCausalLM, mem usage=39.16 GB.
Use sliding window memory pool. full_layer_tokens=387129, swa_layer_tokens=309703
KV Cache is allocated. #tokens: 309703, K size: 3.54 GB, V size: 3.54 GB
KV Cache is allocated. #tokens: 387129, K size: 4.43 GB, V size: 4.43 GB
Capture cuda graph bs [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256]
Disable piecewise CUDA graph because --disable-piecewise-cuda-graph is set
```

Each batch's latency is the median of its samples, in ms. The difference is this run's time against the reference's. The sample spread is how far one batch's samples spread within a run (max - min over the median), so differences near it are measurement noise. The model is fit.py's, fitted on the reference, and its error is relative to the measured time, as in fit.py.

| phase | batches | mean difference | mean abs difference | max abs difference | sample spread, median: reference / this run | model error, mean abs: reference / this run |
|---|---|---|---|---|---|---|
| prefill | 20 | -0.1% | 0.6% | 3.0% | 0.4% / 0.6% | 3.5% / 3.6% |
| decode | 20 | -0.1% | 2.3% | 9.8% | 2.8% / 2.0% | 5.3% / 5.3% |
| hybrid | 48 | -0.5% | 0.6% | 4.4% | 0.3% / 0.3% | 2.8% / 3.3% |

### prefill

| batch (lengths) | reference | this run | difference | model | model error |
|---|---|---|---|---|---|
| 16 | 66.86 | 66.70 | -0.2% | 70.13 | +5.1% |
| 32 | 67.62 | 66.53 | -1.6% | 70.13 | +5.4% |
| 64 | 66.97 | 66.68 | -0.4% | 70.13 | +5.2% |
| 128 | 66.96 | 67.30 | +0.5% | 70.13 | +4.2% |
| 256 | 67.67 | 67.57 | -0.1% | 70.13 | +3.8% |
| 512 | 67.78 | 66.57 | -1.8% | 70.13 | +5.3% |
| 1024 | 68.42 | 70.48 | +3.0% | 70.13 | -0.5% |
| 2048 | 93.45 | 93.63 | +0.2% | 97.20 | +3.8% |
| 4096 | 193.38 | 193.28 | -0.1% | 197.75 | +2.3% |
| 8192 | 468.04 | 468.09 | +0.0% | 474.22 | +1.3% |
| 64x128 | 272.04 | 269.16 | -1.1% | 280.43 | +4.2% |
| 32x256 | 273.76 | 272.32 | -0.5% | 279.18 | +2.5% |
| 16x512 | 278.04 | 276.44 | -0.6% | 283.39 | +2.5% |
| 8x1024 | 288.05 | 288.79 | +0.3% | 295.13 | +2.2% |
| 4x2048 | 311.91 | 312.24 | +0.1% | 320.30 | +2.6% |
| 6144 + 31x16 | 546.95 | 546.55 | -0.1% | 526.08 | -3.7% |
| 4096 + 31x64 | 395.63 | 394.80 | -0.2% | 416.33 | +5.5% |
| 2048 + 15x256 | 262.02 | 262.90 | +0.3% | 279.81 | +6.4% |
| 1024 + 15x16 | 82.38 | 82.02 | -0.4% | 80.26 | -2.1% |
| 512 + 7x256 | 94.81 | 95.29 | +0.5% | 98.70 | +3.6% |

### decode

| batch (lengths) | reference | this run | difference | model | model error |
|---|---|---|---|---|---|
| 1024 | 10.10 | 10.25 | +1.5% | 10.14 | -1.1% |
| 2x1024 | 11.57 | 11.49 | -0.7% | 11.77 | +2.4% |
| 4x1024 | 13.61 | 13.48 | -1.0% | 14.13 | +4.8% |
| 8x1024 | 13.40 | 13.82 | +3.2% | 14.61 | +5.7% |
| 16x1024 | 15.93 | 15.62 | -2.0% | 15.96 | +2.2% |
| 32x1024 | 16.90 | 17.45 | +3.2% | 18.44 | +5.7% |
| 64x1024 | 20.86 | 20.15 | -3.4% | 22.47 | +11.5% |
| 128x1024 | 26.72 | 26.35 | -1.4% | 28.11 | +6.7% |
| 256x1024 | 32.19 | 32.06 | -0.4% | 34.60 | +7.9% |
| 64 | 10.10 | 10.00 | -1.0% | 10.06 | +0.6% |
| 4096 | 10.37 | 10.33 | -0.4% | 10.39 | +0.7% |
| 65536 | 15.74 | 15.65 | -0.6% | 15.48 | -1.1% |
| 256x64 | 29.75 | 29.14 | -2.1% | 29.48 | +1.2% |
| 64x4096 | 24.28 | 25.22 | +3.9% | 26.71 | +5.9% |
| 16x16384 | 19.15 | 21.02 | +9.8% | 21.98 | +4.5% |
| 4x65536 | 19.41 | 19.47 | +0.3% | 23.46 | +20.5% |
| 98304 + 7x128 | 21.35 | 21.51 | +0.8% | 22.54 | +4.8% |
| 98304 + 31x1024 | 26.08 | 24.92 | -4.5% | 26.50 | +6.4% |
| 32768 + 127x1024 | 29.89 | 28.85 | -3.5% | 30.74 | +6.6% |
| 8192 + 255x128 | 29.64 | 28.73 | -3.0% | 30.49 | +6.1% |

### hybrid

| batch (prompts / decodes) | reference | this run | difference | model | model error |
|---|---|---|---|---|---|
| 256 / 512 | 70.09 | 69.44 | -0.9% | 71.27 | +2.6% |
| 256 / 2048 | 69.22 | 68.96 | -0.4% | 71.27 | +3.4% |
| 256 / 8x512 | 70.44 | 67.31 | -4.4% | 71.27 | +5.9% |
| 256 / 8x2048 | 70.31 | 68.05 | -3.2% | 71.27 | +4.7% |
| 256 / 32x512 | 69.92 | 68.60 | -1.9% | 71.27 | +3.9% |
| 256 / 32x2048 | 87.69 | 87.16 | -0.6% | 91.55 | +5.0% |
| 256 / 128x512 | 137.68 | 136.76 | -0.7% | 150.51 | +10.1% |
| 256 / 128x2048 | 266.42 | 265.53 | -0.3% | 265.93 | +0.2% |
| 1024 / 512 | 70.43 | 68.70 | -2.5% | 71.27 | +3.8% |
| 1024 / 2048 | 70.22 | 68.41 | -2.6% | 71.27 | +4.2% |
| 1024 / 8x512 | 78.80 | 78.67 | -0.2% | 81.24 | +3.3% |
| 1024 / 8x2048 | 109.03 | 109.74 | +0.6% | 112.80 | +2.8% |
| 1024 / 32x512 | 143.28 | 143.13 | -0.1% | 153.86 | +7.5% |
| 1024 / 32x2048 | 268.16 | 268.59 | +0.2% | 275.20 | +2.5% |
| 1024 / 128x512 | 412.82 | 413.03 | +0.0% | 444.34 | +7.6% |
| 1024 / 128x2048 | 919.16 | 919.69 | +0.1% | 924.77 | +0.6% |
| 4096 / 512 | 209.04 | 209.11 | +0.0% | 211.15 | +1.0% |
| 4096 / 2048 | 227.43 | 226.66 | -0.3% | 227.94 | +0.6% |
| 4096 / 8x512 | 292.46 | 292.51 | +0.0% | 291.08 | -0.5% |
| 4096 / 8x2048 | 410.90 | 410.76 | -0.0% | 413.90 | +0.8% |
| 4096 / 32x512 | 546.35 | 546.59 | +0.0% | 565.14 | +3.4% |
| 4096 / 32x2048 | 1042.06 | 1043.96 | +0.2% | 1051.48 | +0.7% |
| 4096 / 128x512 | 1571.00 | 1574.66 | +0.2% | 1661.35 | +5.5% |
| 4096 / 128x2048 | 3584.20 | 3592.56 | +0.2% | 3601.80 | +0.3% |
| 4x1024 / 512 | 161.44 | 161.56 | +0.1% | 163.66 | +1.3% |
| 4x1024 / 2048 | 165.69 | 166.07 | +0.2% | 169.04 | +1.8% |
| 4x1024 / 8x512 | 182.80 | 182.13 | -0.4% | 184.84 | +1.5% |
| 4x1024 / 8x2048 | 211.36 | 211.05 | -0.1% | 216.40 | +2.5% |
| 4x1024 / 32x512 | 248.06 | 248.30 | +0.1% | 257.46 | +3.7% |
| 4x1024 / 32x2048 | 372.56 | 373.05 | +0.1% | 378.80 | +1.5% |
| 4x1024 / 128x512 | 516.96 | 516.24 | -0.1% | 547.94 | +6.1% |
| 4x1024 / 128x2048 | 1021.14 | 1022.32 | +0.1% | 1028.37 | +0.6% |
| 8x256 / 512 | 86.87 | 86.42 | -0.5% | 88.61 | +2.5% |
| 8x256 / 2048 | 89.25 | 87.90 | -1.5% | 91.14 | +3.7% |
| 8x256 / 8x512 | 92.48 | 92.55 | +0.1% | 95.10 | +2.8% |
| 8x256 / 8x2048 | 100.32 | 100.75 | +0.4% | 103.85 | +3.1% |
| 8x256 / 32x512 | 110.70 | 109.81 | -0.8% | 117.37 | +6.9% |
| 8x256 / 32x2048 | 143.75 | 142.68 | -0.7% | 147.45 | +3.3% |
| 8x256 / 128x512 | 191.81 | 191.22 | -0.3% | 206.41 | +7.9% |
| 8x256 / 128x2048 | 319.74 | 319.43 | -0.1% | 321.83 | +0.8% |
| 32x128 / 512 | 148.50 | 147.50 | -0.7% | 151.51 | +2.7% |
| 32x128 / 2048 | 150.30 | 150.37 | +0.0% | 153.56 | +2.1% |
| 32x128 / 8x512 | 152.34 | 151.59 | -0.5% | 155.56 | +2.6% |
| 32x128 / 8x2048 | 156.67 | 156.48 | -0.1% | 160.50 | +2.6% |
| 32x128 / 32x512 | 163.15 | 161.93 | -0.7% | 169.43 | +4.6% |
| 32x128 / 32x2048 | 178.51 | 177.78 | -0.4% | 184.30 | +3.7% |
| 32x128 / 128x512 | 211.84 | 211.19 | -0.3% | 224.90 | +6.5% |
| 32x128 / 128x2048 | 277.86 | 277.58 | -0.1% | 279.49 | +0.7% |
