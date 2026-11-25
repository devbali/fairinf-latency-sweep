# Large decode steps, gpt-oss-120b-tp4, on the FairInference experiments setup (2026-11-24)

`src/runner.py` on `src/decode_large_120b-tp4.json`: 675 decode batches, up to 512 requests and 464,909 tokens (runner.py's limit: the 470,300-token sliding-window pool), the regime the main plan
(`gptoss-120b-tp4-final-setup-20261124`) leaves out. Same setup as that run.

export_profile's `gpt-oss-120b-tp4-final` fits decode on both runs together, with
bends in N at 256 and 384 as well. Against a fit on the main run alone, the median error on these steps
falls from 6.0% to 3.1%, with the main run's steps unchanged or better.
