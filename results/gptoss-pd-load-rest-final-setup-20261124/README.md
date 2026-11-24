# PD KV transfer under the prefill, decode and both loads, on the FairInference experiments setup (2026-11-24)

The same `pd_load.py` run as `gptoss-pd-load-idle-final-setup-20261124` with `--scenarios prefill,decode,both`,
after the first run crashed in NIXL. No crash this time; 15,834 probe requests per load.

Setup: the FairInference experiments setup (the fork after its upstream sync, sglang source 66697235d3bb, torch
2.14.1+cu130, nixl and nixl-cu13 1.3.0 as the fork's CI installs them), gpt-oss-20b, a prefill server on one A100 and
a decode server on another, KV over NIXL (UCX over NVLink). Server flags: `--dtype bfloat16 --sampling-backend
pytorch --mem-fraction-static 0.7 --prefill-max-requests 64 --disable-overlap-schedule
--disaggregation-transfer-backend nixl`, decode on `--base-gpu-id 1`. results.jsonl was rebuilt with
`pd_runner.py --parse` after its ReqTimeStats pattern learned the new `attempts=` field (759d5dc).

| load | held-out mean error, selected (old run) | RMSE (old) | transfer new / old, median | one prompt, ms | 33-64 prompts, ms |
|---|---|---|---|---|---|
| prefill | 30.1% (34.5%) | 16.5 ms (30.0) | 1.38 | 5.1 -> 11.7 | 119.4 -> 102.6 |
| decode | 19.9% (19.3%) | 8.3 ms (6.4) | 1.38 | 14.6 -> 23.4 | 108.0 -> 108.0 |
| both | 24.8% (31.3%) | 20.2 ms (34.7) | 1.29 | 15.8 -> 29.2 | 131.3 -> 123.7 |
