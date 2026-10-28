# fairinf-latency-sweep

Times one prefill and one decode forward pass of SGLang (the FairInference
fork) on fake batches. These are the measurements FairInference's latency
model is fitted to ([arXiv 2609.18112](https://arxiv.org/abs/2609.18112),
§5.4).

There are two files:

- **`runner.py`** loads SGLang's `ModelRunner` once on one GPU, then times
  every batch. It uses the same code path as `python -m sglang.bench_one_batch`:
  no server, scheduler or radix cache.
  - `fake_batch` makes one request per prompt length, filled with random tokens.
  - `time_batch` prefills the batch in one forward pass, then decodes one token
    per request. It returns both latencies in seconds.
- **`batches.py`** lists the batches to time in `BATCHES`. Each batch is the
  prompt length of every request in it.

Each batch runs once as a warmup, then three more times as samples.

## Run

On a GPU VM, set up for hooks as in `fairinf-sglang/CLAUDE.md`:

```bash
export PATH=/home/devbali/fairinf/hooks-venv/bin:/usr/local/cuda-12.4/bin:$PATH
export PYTHONPATH=/home/devbali/fairinf/fairinf-sglang/python:$PYTHONPATH
python runner.py \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct --dtype float16 \
    --mem-fraction-static 0.7 --trust-remote-code \
    --attention-backend flashinfer --sampling-backend pytorch \
    --disable-piecewise-cuda-graph \
    --batches batches.py --out results.jsonl
```

`--batches` and `--out` are the runner's own flags; every other flag is an
SGLang server flag. Each sample is appended to `results.jsonl` as one JSON
line with these fields:

- `batch`: the prompt lengths
- `sample`: 0, 1 or 2
- `prefill_s`
- `decode_s`

Both times are host wall clock from batch preparation through sampling, with
the GPU synced before and after, as in `bench_one_batch`. The decode step runs
on the KV cache the prefill just filled, so request i attends over its prompt
plus one token.
