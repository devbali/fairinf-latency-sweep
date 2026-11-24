# PD KV transfer, full PD plan, on the FairInference experiments setup (2026-11-24)

`src/pd_runner.py` on `src/pd_batches.json`, as `gptoss-pd-nixl-20261103`, 23,564 requests.

Setup: the FairInference experiments setup (the fork after its upstream sync, sglang source 66697235d3bb, torch
2.14.1+cu130, nixl and nixl-cu13 1.3.0 as the fork's CI installs them), gpt-oss-20b, a prefill server on one A100 and
a decode server on another, KV over NIXL (UCX over NVLink). Server flags: `--dtype bfloat16 --sampling-backend
pytorch --mem-fraction-static 0.7 --prefill-max-requests 64 --disable-overlap-schedule
--disaggregation-transfer-backend nixl`, decode on `--base-gpu-id 1`. results.jsonl was rebuilt with
`pd_runner.py --parse` after its ReqTimeStats pattern learned the new `attempts=` field (759d5dc).

Against `gptoss-pd-nixl-20261103` (the same plan on fork 99f39f2b5, nixl 1.0.1), per prefill step's transfer
(`pd_fits.transfer_samples`, 312 prompt-length sets in both): new / old 1.57 at the median. One prompt 5.9 -> 9.4 ms,
2-8 prompts 15.3 -> 25.3 ms, 9-32 prompts 41.8 -> 55.6 ms, 33-64 prompts 100.1 -> 98.4 ms.
