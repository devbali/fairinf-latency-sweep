# Running on eu1

These are the exact setup and commands behind the runs in `results/`, for the
one VM they all used. They only work as written on that VM.

| | |
|---|---|
| VM | `fairinf-matteo-fork-1xa100-eu1`, project `delta-fair-inference`, zone `europe-west4-a`, user `devbali` |
| GPU | one A100-SXM4-80GB, spot, about $3.10/hr; NVIDIA driver 580.178.04 (CUDA 13.0) |
| SGLang | fairinf-sglang at `4e84427ab`, as source in `~/latency-test/fairinf-sglang-4e84427ab/python`, used through PYTHONPATH (not pip-installed) |
| venv | `~/latency-test/venv`, built by `bash vm/fresh_venv.sh`: Python 3.10.12, torch 2.11.0+cu130, sglang-kernel 0.4.2.post1, flashinfer 0.6.8.post1, transformers 5.6.0 |
| model | `lmsys/gpt-oss-20b-bf16`, at `~/.cache/huggingface/hub/models--lmsys--gpt-oss-20b-bf16/snapshots/8de8d3dfa4da071323322a4a3049612203a3b1eb` |
| this repo | `~/latency-test/fairinf-latency-sweep` |
| accuracy check's venv | `~/latency-test/gpt-oss-evals-venv`: openai, pandas, numpy, jinja2, tqdm, blobfile |
| OpenAI's eval code | `~/.cache/sanity-accuracy/gpt-oss-277020131d52`, cloned there by `sanity_accuracy.py` |

## Start, copy, stop

On your machine, from this repo:

```bash
gcloud compute instances start fairinf-matteo-fork-1xa100-eu1 --zone europe-west4-a
gcloud compute scp --recurse --zone europe-west4-a src vm sanity_latency.py sanity_accuracy.py \
    devbali@fairinf-matteo-fork-1xa100-eu1:~/latency-test/fairinf-latency-sweep/
gcloud compute ssh devbali@fairinf-matteo-fork-1xa100-eu1 --zone europe-west4-a

# when done:
gcloud compute instances stop fairinf-matteo-fork-1xa100-eu1 --zone europe-west4-a --discard-local-ssd=true
```

Before a long run, make the VM power itself off at a deadline, in case nobody
stops it (on the VM): `sudo systemd-run --on-active=120min /bin/systemctl poweroff`

## The server

This is the setup all the runs measured and checked. On the VM:

```bash
LT=$HOME/latency-test
MODEL=$HOME/.cache/huggingface/hub/models--lmsys--gpt-oss-20b-bf16/snapshots/8de8d3dfa4da071323322a4a3049612203a3b1eb
SERVER="TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
PYTHONPATH=$LT/fairinf-sglang-4e84427ab/python:$LT/fairinf-latency-sweep/vm \
CUDA_VISIBLE_DEVICES=0 \
$LT/venv/bin/python -m sglang.launch_server --model-path $MODEL \
--dtype bfloat16 --mem-fraction-static 0.7 --sampling-backend pytorch \
--scheduling-policy-path noop_cache_policy.NoOpWithCachePolicy \
--piecewise-cuda-graph-compiler inductor --port 30000"

bash -c "$SERVER"   # serves on port 30000; the checks below take "$SERVER" as is
```

What each part is for:
- `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`: SGLang's compile hook breaks torch
  2.11's compile cache. With the caches off, the CUDA graphs below compile.
- `PYTHONPATH`: SGLang from the fork's source, plus `vm/` for the cache policy.
- `CUDA_VISIBLE_DEVICES=0`: the one GPU.
- `--piecewise-cuda-graph-compiler inductor`: prefill and hybrid steps run as
  CUDA graphs, as decode steps always do. SGLang's default compiler can't
  trace the fork's RMSNorm.
- `--scheduling-policy-path noop_cache_policy.NoOpWithCachePolicy`: without it
  the fork's scheduler crashes on gpt-oss's first request; see
  `noop_cache_policy.py`.
- `--dtype bfloat16 --mem-fraction-static 0.7 --sampling-backend pytorch`: as
  in every run.
- No flag picks the kernels for the experts or attention. SGLang uses its
  defaults for gpt-oss on an A100: `triton_kernel` and `triton`.
- **Startup takes about 5 minutes:** compiling, then capturing graphs for 58
  step sizes up to 8,192 tokens.
- **To serve chat clients, add `--reasoning-parser gpt-oss`.** It keeps the
  reasoning out of the answer. Also add `--allow-auto-truncate` if clients ask
  for the whole context as max_tokens, as OpenAI's eval does. Neither changes
  how the model runs, and `sanity_accuracy.py` adds both.

Before starting it, check that `nvidia-smi` lists no processes and nothing
listens on port 30000. Run one server at a time.

## The commands we ran

On the VM, from `~/latency-test/fairinf-latency-sweep`, with `LT` and `SERVER`
set as above.

**The full sweep**, `results/gptoss-cuda-graphs-20261029` (about 70 minutes):

```bash
PIECEWISE=inductor nohup bash vm/run_clean.sh src/batches.json results.jsonl > run.log 2>&1 &
```

`vm/run_clean.sh` runs no server. It times the steps in-process with
`src/runner.py`, using the same environment and flags. It leaves out the cache
policy, which only the server's scheduler needs. Copy `results.jsonl` and
`run.log` back into `results/<run>/`, then on your machine run
`python src/compare_fits.py results/<run>/results.jsonl`.

**The latency check's reference**, `results/gptoss-sanity-cuda-graphs-20261029`
(about 6 minutes). This is the sweep's command on the short plan:

```bash
mkdir -p results/gptoss-sanity-cuda-graphs-20261029
PIECEWISE=inductor bash vm/run_clean.sh src/sanity_batches.json \
    results/gptoss-sanity-cuda-graphs-20261029/results.jsonl > results/gptoss-sanity-cuda-graphs-20261029/run.log 2>&1
```

**The latency check**, `results/gptoss-sanity-latency-eu1-20261029`: it
matches the reference (about 5.5 minutes).

```bash
$LT/venv/bin/python sanity_latency.py "$SERVER" --out results/gptoss-sanity-latency-eu1-20261029
```

**The accuracy check**, `results/gptoss-sanity-accuracy-eu1-20261029`: 54.5%
on GPQA Diamond against OpenAI's 56.8%, so it matches (about 6 minutes).

```bash
$LT/gpt-oss-evals-venv/bin/python sanity_accuracy.py "$SERVER" --out results/gptoss-sanity-accuracy-eu1-20261029
```

`--attempts 8` asks each question 8 times, as OpenAI's own script does. That
run, `results/gptoss-openai-evals-cuda-graphs-20261029`, scored 59.0%; an
earlier version of the check made it, running OpenAI's script as is. The
evals venv was made once, with:

```bash
python3 -m venv $LT/gpt-oss-evals-venv
$LT/gpt-oss-evals-venv/bin/pip install openai pandas numpy jinja2 tqdm blobfile
```

Each check writes a directory under `results/` on the VM. Copy it back from
your machine:

```bash
gcloud compute scp --recurse --zone europe-west4-a \
    devbali@fairinf-matteo-fork-1xa100-eu1:~/latency-test/fairinf-latency-sweep/results/<dir> results/
```

## Leave alone

- `~/fairinf/hooks-venv` and `~/fairinf/fairinf-sglang`: the FairInference
  hooks experiments use them.
- `/mnt/ssd`.
