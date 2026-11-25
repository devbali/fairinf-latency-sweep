# Large decode steps, gpt-oss-20b, on the FairInference experiments setup (2026-11-24)

`src/runner.py` on `src/decode_large_20b.json`: 576 decode batches, up to 512 requests and 299,912 tokens, the regime the main plan
(`gptoss-20b-final-setup-20261124`) leaves out. Same setup as that run.

export_profile's `gpt-oss-20b-a100-final` fits decode on both runs together, with
bends in N at 256 and 384 as well. Against a fit on the main run alone, the median error on these steps
falls from 10.4% to 8.3%, with the main run's steps unchanged or better.
