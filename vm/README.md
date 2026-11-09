# Running on eu1

These are the exact setup and commands behind the runs in `results/`. All but
the PD runs and the gpt-oss-120b run used eu1; those, at the end, used a
4xA100 VM. They only work as written on those VMs.

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

## PD disaggregation on the 4xA100 VM

`src/pd_runner.py` needs two GPUs, so its runs used another VM:

| | |
|---|---|
| VM | `fairinf-matteo-fork-4xa100`, project `delta-fair-inference`, zone `europe-west4-a`, user `devbali` |
| GPUs | four A100-SXM4-80GB, spot (a2-ultragpu-4g); every pair joined by NVLink (NV12); driver 580.178.04. No RDMA network cards |
| SGLang | fairinf-sglang main at `99f39f2b5`, as source in `~/latency-test/fairinf-sglang-99f39f2b5/python` |
| venv | `~/latency-test/venv`, built by `bash vm/fresh_venv.sh` as on eu1, plus `mooncake-transfer-engine-cuda13==0.3.10.post2`, the version the fork's CI installs for CUDA 13 (`scripts/ci/cuda/ci_install_dependency.sh`), and NIXL: `nixl-cu13==1.0.1`, then `nixl==1.0.1` with `--no-deps` |
| system | Ubuntu's `libibverbs1`, installed with apt: mooncake's engine links to it even when it moves data over TCP |
| model | `lmsys/gpt-oss-20b-bf16` at the same snapshot as eu1 |

The VM was made from the same disk snapshot as eu1, with a 200 GB disk. To
fit the venv and the model, `~/.cache/pip` (27 GB) and the Llama-3-8B
checkpoint (15 GB) were deleted from it; about 7 GB is left.

The two servers are the gpt-oss server command above, one per GPU, with PD
mode and a port added.

**`results/gptoss-pd-mooncake-tcp-20261103`: the fork's defaults.** Everything
else about PD is SGLang's default: the mooncake backend, its queues and
threads, the bootstrap port 8998. With no RDMA cards, mooncake moves the KV
cache over TCP through host memory (its log: "Found 0 HCAs", then
"TcpTransport: listen on port ..."). The run broke 2.8 minutes into the plan,
when the VM ran out of ports for mooncake's TCP connections; the run's README
explains why.

```bash
ENVS="TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 PYTHONPATH=$LT/fairinf-sglang-99f39f2b5/python:$LT/fairinf-latency-sweep/vm"
FLAGS="--model-path $MODEL --dtype bfloat16 --mem-fraction-static 0.7 --sampling-backend pytorch \
--scheduling-policy-path noop_cache_policy.NoOpWithCachePolicy --piecewise-cuda-graph-compiler inductor"
PREFILL="$ENVS CUDA_VISIBLE_DEVICES=0 $LT/venv/bin/python -m sglang.launch_server $FLAGS --disaggregation-mode prefill --port 30000"
DECODE="$ENVS CUDA_VISIBLE_DEVICES=1 $LT/venv/bin/python -m sglang.launch_server $FLAGS --disaggregation-mode decode --port 30001"

$LT/venv/bin/python src/pd_runner.py --prefill "$PREFILL" --decode "$DECODE" --out results/gptoss-pd-mooncake-tcp-20261103
```

**`results/gptoss-pd-nixl-20261103`: NVLink.** The same servers with
`--disaggregation-transfer-backend nixl`, laid out as the fork's single-node PD
recipe (`docs/advanced_features/pd_disaggregation.md`): every GPU visible, and
the decode server on `--base-gpu-id 1`. UCX can only copy from one GPU to
another over NVLink if the process sees both. NIXL's backend is SGLang's
default for it, UCX. The whole plan took 18 minutes.

```bash
$LT/venv/bin/python -m pip download --no-deps nixl-cu13==1.0.1 nixl==1.0.1 -d $LT/nixl-wheels   # checked: no file overlaps the venv
$LT/venv/bin/uv pip install --python $LT/venv/bin/python nixl-cu13==1.0.1
$LT/venv/bin/uv pip install --python $LT/venv/bin/python --no-deps nixl==1.0.1   # its nixl-cu12 dependency ships the same nixl_ep files

FLAGS="$FLAGS --disaggregation-transfer-backend nixl"
PREFILL="$ENVS $LT/venv/bin/python -m sglang.launch_server $FLAGS --disaggregation-mode prefill --port 30000"
DECODE="$ENVS $LT/venv/bin/python -m sglang.launch_server $FLAGS --disaggregation-mode decode --port 30001 --base-gpu-id 1"

$LT/venv/bin/python src/pd_runner.py --prefill "$PREFILL" --decode "$DECODE" --out results/gptoss-pd-nixl-20261103
```

**`results/gptoss-pd-sanity-nixl-20261103`: the PD check's reference.** The
same two servers on the short plan:

```bash
$LT/venv/bin/python src/pd_runner.py --prefill "$PREFILL" --decode "$DECODE" --plan src/pd_sanity_batches.json \
    --out results/gptoss-pd-sanity-nixl-20261103
```

**`results/gptoss-sanity-pd-4xa100-20261103`: the PD check against it,**
right after, with the same commands:

```bash
$LT/venv/bin/python sanity_pd.py "$PREFILL" "$DECODE" --out results/gptoss-sanity-pd-4xa100-20261103
```

**`results/gptoss-pd-load-nixl-20261103` and
`results/gptoss-pd-pressure-nixl-20261103`: transfers under load.**
`src/pd_load.py` with the same two servers, each scenario run again with py-spy
on both schedulers. py-spy went into its own directory, not the venv, and
needs ptrace rights over the servers; the sysctl lasts until the next boot.
The pressure run caps the decode server's KV cache.

```bash
$LT/venv/bin/pip install --target $LT/pyspy py-spy==0.4.0
sudo sysctl -w kernel.yama.ptrace_scope=0
PROFILE="$LT/pyspy/bin/py-spy record --nonblocking --idle --threads --rate 250 --format raw --pid {pid} --output {out}"

$LT/venv/bin/python src/pd_load.py --prefill "$PREFILL" --decode "$DECODE" --profile "$PROFILE" \
    --scenarios idle,prefill,decode,both,idle:profile,prefill:profile,decode:profile,both:profile \
    --out results/gptoss-pd-load-nixl-20261103
$LT/venv/bin/python src/pd_load.py --prefill "$PREFILL" --decode "$DECODE --max-total-tokens 40000" \
    --profile "$PROFILE" --scenarios pressure,pressure:profile --out results/gptoss-pd-pressure-nixl-20261103
```

Both took 19 minutes together. Before them, starting the VM failed for 20
minutes with STOCKOUT (no spot capacity for a2-ultragpu-4g in the zone); a
start every 2 minutes got it.

**`results/gptoss-pd-load-full-nixl-20261104`: the transfer model under each
load.** The same servers, with the PD plan's 334 batches as the probes, 3
times each, under the four loads, without the profiler. It took 66 minutes:

```bash
$LT/venv/bin/python src/pd_load.py --prefill "$PREFILL" --decode "$DECODE" --probes src/pd_batches.json \
    --samples 3 --scenarios idle,prefill,decode,both --out results/gptoss-pd-load-full-nixl-20261104
```

The VM again took about 24 minutes of tries every 2 minutes to start.

PD mode turns piecewise CUDA graphs off (`server_args.py`, "13. PD
disaggregation"), so prefill runs eagerly and the inductor flag does nothing.
`pd_runner.py` adds `--enable-metrics`, `--skip-tokenizer-init` and
`--enable-request-time-stats-logging` to both commands, for its timings; its
docstring says why. Both servers start in about 2 to 3 minutes.

## gpt-oss-120b with tensor parallelism on the 4xA100 VM

`results/gptoss-120b-tp4-20261103` is the full sweep, `src/batches.json`, on
gpt-oss-120b over the VM's four GPUs with `--tp 4`: the setup of
`results/gptoss-cuda-graphs-20261029` otherwise.

| | |
|---|---|
| model | `lmsys/gpt-oss-120b-bf16` at revision `3609e677162887e2926a347645d481475d393746`, 119 GB. Its experts are stored as FP8 E5M2, which holds gpt-oss's MXFP4 values exactly, and everything else as BF16. SGLang loads all of it as BF16: 54.6 GB per GPU. |
| where | a RAM disk, `/mnt/ram`, as the boot disk has 6 GB free. It is gone after every stop, so the model has to be downloaded again after each start, in about 3 minutes. |
| SGLang | fork main `99f39f2b5`, the PD runs' |
| memory | `--mem-fraction-static 0.819`, SGLang's own default for 4 GPUs of 80 GB. The KV cache then holds 469,904 sliding-window tokens, more than the plan's largest batch needs (299,180); 0.7 would leave room for about 70k. |

On the VM, from `~/latency-test/fairinf-latency-sweep`:

```bash
sudo mkdir -p /mnt/ram && sudo mount -t tmpfs -o size=150g,mode=1777 tmpfs /mnt/ram
HF_HOME=/mnt/ram/hf-home HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_XET_HIGH_PERFORMANCE=1 \
    $LT/venv/bin/hf download lmsys/gpt-oss-120b-bf16 --revision 3609e677162887e2926a347645d481475d393746 \
    --local-dir /mnt/ram/gpt-oss-120b-bf16

MODEL=/mnt/ram/gpt-oss-120b-bf16 FORK=$LT/fairinf-sglang-99f39f2b5 CUDA_VISIBLE_DEVICES=0,1,2,3 PIECEWISE=inductor \
    nohup bash vm/run_clean.sh src/batches.json results.jsonl --tp 4 --mem-fraction-static 0.819 > run.log 2>&1 &
```

- **Not `/dev/shm`:** systemd-logind deletes a user's files there when the
  user's last session closes (`RemoveIPC=yes`, the default), and it deleted
  the first download.
- **`HF_HOME` in the RAM disk** keeps Hugging Face's caches off the boot disk.
- **`--tp 4`** makes `src/runner.py` run one process per GPU, as
  `sglang.bench_one_batch` does. They all time the same batches, and rank 0
  records the times.
- **Other defaults change with 4 GPUs:** decode CUDA graphs go up to 512
  requests instead of 256, and the plan's decode batches, up to 256, are
  padded to the same sizes as before.
- **Startup takes about 11 minutes:** 2 to load the weights, 1.5 to capture
  the decode graphs, and 6.5 to compile and capture the piecewise graphs.

## Leave alone

- `~/fairinf/hooks-venv` and `~/fairinf/fairinf-sglang`: the FairInference
  hooks experiments use them.
- `/mnt/ssd`.
