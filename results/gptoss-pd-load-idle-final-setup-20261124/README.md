# PD KV transfer under load, idle, on the FairInference experiments setup (2026-11-24)

`src/pd_load.py --probes src/pd_batches.json --samples 3`, as `gptoss-pd-load-full-nixl-20261104`. This run was
meant to cover all four loads; the prefill server crashed with a segmentation fault in NIXL's UCX progress thread
(`nixlUcxSharedThread::run`, `ucp_worker_arm`) early in the prefill load, so only idle is complete (15,834 probe
requests). The prefill, decode and both loads are `gptoss-pd-load-rest-final-setup-20261124`.

Setup: the FairInference experiments setup (the fork after its upstream sync, sglang source 66697235d3bb, torch
2.14.1+cu130, nixl and nixl-cu13 1.3.0 as the fork's CI installs them), gpt-oss-20b, a prefill server on one A100 and
a decode server on another, KV over NIXL (UCX over NVLink). Server flags: `--dtype bfloat16 --sampling-backend
pytorch --mem-fraction-static 0.7 --prefill-max-requests 64 --disable-overlap-schedule
--disaggregation-transfer-backend nixl`, decode on `--base-gpu-id 1`. results.jsonl was rebuilt with
`pd_runner.py --parse` after its ReqTimeStats pattern learned the new `attempts=` field (759d5dc).

Fitted with `compare_fits.py --pd-phase idle`: the selected model's held-out mean error is 19.7% (RMSE 6.8 ms),
against 27.1% (5.1 ms) on the old run. Transfers per prefill step: new / old 1.64 at the median; one prompt 5.3 ->
9.5 ms, 33-64 prompts 93.0 -> 96.0 ms.
