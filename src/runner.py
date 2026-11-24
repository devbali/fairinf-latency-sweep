"""Time single prefill, decode and hybrid forward passes of SGLang on fake batches.

Same low-level path as `python -m sglang.bench_one_batch`: no server,
scheduler or radix cache, just batches built by hand and run through
SGLang's ModelRunner. The batches to time come from a JSON plan
(batches.json by default).

With tensor parallelism (--tp N), it runs as bench_one_batch does: one
process per TP rank, each with its own ModelRunner on its own GPU, all timing
the same batches in step. Rank 0 records the times; a step's collectives make
every rank wait for the slowest, as on a server.
"""

import argparse
import json
import logging
import multiprocessing
import multiprocessing.connection
import os
import sys
import time
import uuid

import numpy as np
import torch

# SGLang moved bench_one_batch to sglang.benchmark.one_batch (upstream #40795) and changed
# how a ModelRunner is built and fed; NEW_API follows that module, the old path 4e84427ab.
try:
    from sglang.benchmark import one_batch as _ob
    from sglang.benchmark.one_batch import TreeCacheNamespace
    NEW_API = True
except ImportError:
    from sglang.bench_one_batch import TreeCacheNamespace
    NEW_API = False
from array import array
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.common import maybe_reindex_device_id

WARMUP_RUNS = 1  # per batch, not recorded
SAMPLES = 3  # per batch
SEED = 0  # of the random prompt tokens, the same on every TP rank


def load_model(server_args, port_args, gpu_id, tp_rank):
    """Load TP rank tp_rank's weights, allocate its KV cache and capture CUDA graphs on gpu_id."""
    if NEW_API:
        _ob.publish(server_args, role="scheduler", ranks=_ob.SpawnRanks(
            world_rank=_ob.spawn_world_rank(server_args, tp_rank=tp_rank, pp_rank=0), gpu_id=gpu_id))
        _ob._init_process_global_configs()
        return _ob.load_model(server_args, port_args, gpu_id, tp_rank)[0].torch_runner
    initialize_moe_config(server_args)  # MoE layers read their kernel choices from it
    return ModelRunner(
        model_config=ModelConfig.from_server_args(server_args),
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        moe_ep_rank=tp_rank // (server_args.tp_size // server_args.ep_size),
        moe_ep_size=server_args.ep_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )


def clear_kv_cache(model_runner):
    """Free every request slot and KV slot, so timings don't depend on each other."""
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()


def fake_batch(model_runner, lens):
    """One request per entry of lens, each with that many random prompt tokens."""
    tokens = sum(lens) + len(lens)  # plus one decoded token per request
    free = model_runner.token_to_kv_pool_allocator.available_size()
    assert tokens <= free, f"batch needs {tokens} KV slots, {free} are free"
    reqs = []
    for n in lens:
        ids = np.random.randint(0, 10000, n).tolist()
        req = Req(
            rid=uuid.uuid4().hex,
            origin_input_text="",
            origin_input_ids=array("q", ids) if NEW_API else ids,
            sampling_params=SamplingParams(temperature=0, max_new_tokens=16),
        )
        req.logprob_start_len = -1
        if NEW_API:
            req.full_untruncated_fill_ids = req.origin_input_ids
            req.extend_end = n
        else:
            req.fill_ids = req.origin_input_ids
            req.set_extend_input_len(n)
        reqs.append(req)
    # bench_one_batch's stand-in for the radix cache: allocation only, no prefix reuse
    if NEW_API:
        tree_cache = TreeCacheNamespace(
            page_size=_ob.get_schedule().page_size,
            device=model_runner.device,
            req_to_token_pool=model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        )
    else:
        tree_cache = TreeCacheNamespace(
            page_size=model_runner.server_args.page_size,
            device=model_runner.device,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        )
    return ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        tree_cache=tree_cache,
        model_config=model_runner.model_config,
        enable_overlap=False,
        spec_algorithm=SpeculativeAlgorithm.NONE,
    )


def decode_batch(model_runner, lens):
    """A batch ready to decode; request i holds lens[i] tokens in the KV cache.

    No prefill runs. prepare_for_extend allocates the context's KV cache and
    sets up the batch, but the model never fills it: a decode step costs the
    same whatever the cached values are, so only the lengths matter.
    """
    batch = fake_batch(model_runner, lens)
    prepare_extend(model_runner, batch)
    # Stand-in for the token each request's prefill would have sampled,
    # recorded on the requests too, as the scheduler does after a prefill.
    next_token_ids = np.random.randint(0, 10000, len(lens))
    for req, token_id in zip(batch.reqs, next_token_ids.tolist()):
        req.output_ids.append(token_id)
    tokens = torch.tensor(next_token_ids, dtype=torch.int64, device=model_runner.device)
    if NEW_API:
        batch.input_ids = tokens  # prepare_for_decode reads the last sampled token here
    else:
        batch.output_ids = tokens
    return batch


def prepare_extend(model_runner, batch):
    """batch.prepare_for_extend(), plus what one_batch.extend does after it on the new API."""
    batch.prepare_for_extend()
    if NEW_API:
        _ob._maybe_prepare_mlp_sync_batch(batch, model_runner)
        if batch.input_ids is None and getattr(batch, "prefill_input_ids_cpu", None) is not None:
            batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True)
            batch.prefill_input_ids_cpu = None


def prepare_decode(model_runner, batch):
    batch.prepare_for_decode()
    if NEW_API:
        _ob._maybe_prepare_mlp_sync_batch(batch, model_runner)


@torch.no_grad()
def forward(model_runner, batch):
    """One forward pass over a prepared batch; samples one token per request."""
    if NEW_API:
        forward_batch = ForwardBatch.init_new(batch, model_runner, return_hidden_states_before_norm=False)
    else:
        forward_batch = ForwardBatch.init_new(batch.get_model_worker_batch(), model_runner)
    logits_output = model_runner.forward(forward_batch).logits_output
    return model_runner.sample(logits_output, forward_batch)


def time_prefill(model_runner, lens):
    """Seconds for one prefill forward pass; request i has lens[i] prompt tokens.

    Wall time from batch preparation through sampling, with the GPU synced on
    both ends, as in bench_one_batch.
    """
    clear_kv_cache(model_runner)
    batch = fake_batch(model_runner, lens)
    torch.cuda.synchronize()
    start = time.perf_counter()
    prepare_extend(model_runner, batch)
    forward(model_runner, batch)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def time_decode(model_runner, lens):
    """Seconds for one decode step; request i has lens[i] tokens of context."""
    clear_kv_cache(model_runner)
    batch = decode_batch(model_runner, lens)
    torch.cuda.synchronize()
    start = time.perf_counter()
    prepare_decode(model_runner, batch)
    forward(model_runner, batch)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def time_hybrid(model_runner, prefill_lens, decode_lens):
    """Seconds for one mixed step: new prompts of prefill_lens[i] tokens, plus
    one decode step for requests with decode_lens[j] tokens of context.

    Built the way the scheduler does it with --enable-mixed-chunk: the running
    batch goes through prepare_for_decode and is merged into the prefill batch
    by mix_with_running (ForwardMode.MIXED), so every decode request becomes a
    one-token extend on top of its context. As in time_decode, the decode
    requests' KV cache is allocated but never filled.

    With one part empty the scheduler mixes nothing: with no running batch it
    runs a prefill step, with no new prompts a decode step (as a CUDA graph).
    So an empty part times that step instead.
    """
    if not decode_lens:
        return time_prefill(model_runner, prefill_lens)
    if not prefill_lens:
        return time_decode(model_runner, decode_lens)
    clear_kv_cache(model_runner)
    running = decode_batch(model_runner, decode_lens)
    batch = fake_batch(model_runner, prefill_lens)
    torch.cuda.synchronize()
    start = time.perf_counter()
    prepare_extend(model_runner, batch)
    prompt_ids, decode_ids = batch.input_ids, running.input_ids
    prepare_decode(model_runner, running)
    batch.mix_with_running(running)
    if NEW_API:
        # mix_with_running leaves input_ids for the scheduler to join at forward entry
        # (overlap_utils.resolve_forward_inputs): the prompts' tokens, then each decode's last token.
        batch.input_ids = torch.cat([prompt_ids, decode_ids])
        batch.mix_running_indices = None
    forward(model_runner, batch)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def recorded_batches(out_path):
    """The (phase, batch) pairs out_path already holds all SAMPLES samples of, so a rerun
    (say, after a spot VM was preempted) picks up where the last one stopped."""
    counts = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:  # a line cut short when the run stopped
                    continue
                key = (row["phase"], json.dumps(row["batch"]))
                counts[key] = counts.get(key, 0) + 1
    return {key for key, n in counts.items() if n >= SAMPLES}


def time_plan(server_args, port_args, plan, out_path, gpu_id=0, tp_rank=0):
    """Times every batch of plan on one TP rank. Rank 0 appends the samples to out_path.
    Batches out_path already holds are skipped, on every rank alike."""
    done = recorded_batches(out_path)  # read before rank 0 appends anything
    # Only rank 0 logs the setup; every rank's errors still show
    logging.basicConfig(format="%(message)s")  # a rank's own process starts unconfigured
    logging.getLogger().setLevel(logging.INFO if tp_rank == 0 else logging.WARNING)
    model_runner = load_model(server_args, port_args, gpu_id, tp_rank)
    # The fork logs a warning on every request-slot allocation (POOL_ALLOC in
    # mem_cache/memory_pool.py). Silence it so the I/O stays out of the timings.
    logging.getLogger().setLevel(logging.ERROR)
    np.random.seed(SEED)  # every rank builds the same batches

    phases = [
        ("prefill", time_prefill, plan.get("prefill", [])),
        ("decode", time_decode, plan.get("decode", [])),
        (
            "hybrid",
            lambda model_runner, batch: time_hybrid(model_runner, *batch),
            plan.get("hybrid", []),
        ),
    ]
    with open(out_path, "a") if tp_rank == 0 else open(os.devnull, "w") as out:
        for phase, time_fn, batches in phases:
            for i, batch in enumerate(batches):
                if (phase, json.dumps(batch)) in done:
                    continue
                for _ in range(WARMUP_RUNS):
                    time_fn(model_runner, batch)
                for sample in range(SAMPLES):
                    latency_s = time_fn(model_runner, batch)
                    if tp_rank:
                        continue
                    row = {
                        "phase": phase,
                        "batch": batch,
                        "sample": sample,
                        "latency_s": latency_s,
                    }
                    print(json.dumps(row), file=out, flush=True)
                    print(
                        f"{phase} {i + 1}/{len(batches)} sample {sample + 1}/{SAMPLES}: "
                        f"{latency_s * 1e3:.2f} ms"
                    )


def run_ranks(server_args, port_args, plan, out_path):
    """Runs time_plan in one process per TP rank, as bench_one_batch does. Returns the exit code."""
    workers = []
    for tp_rank in range(server_args.tp_size):
        with maybe_reindex_device_id(tp_rank) as gpu_id:
            worker = multiprocessing.Process(
                target=time_plan,
                args=(server_args, port_args, plan, out_path, gpu_id, tp_rank),
                name=f"TP{tp_rank}",
            )
            worker.start()
            workers.append(worker)
    try:
        while workers:
            # A rank that fails leaves the others waiting in a collective, so stop them all
            ready = multiprocessing.connection.wait([worker.sentinel for worker in workers])
            for worker in [worker for worker in workers if worker.sentinel in ready]:
                worker.join()
                workers.remove(worker)
                if worker.exitcode:
                    print(f"{worker.name} exited with code {worker.exitcode}", file=sys.stderr)
                    return 1
        return 0
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.add_argument(
        "--batches",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "batches.json"),
        help='JSON plan with "prefill", "decode" and/or "hybrid" lists of batches',
    )
    parser.add_argument(
        "--out", default="results.jsonl", help="appends one JSON line per sample"
    )
    args = parser.parse_args()
    with open(args.batches) as f:
        plan = json.load(f)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    server_args = ServerArgs.from_cli_args(args)
    if NEW_API:
        server_args.resolve_once()  # as launch_server and one_batch do before starting ranks
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
    if server_args.tp_size == 1:
        time_plan(server_args, port_args, plan, args.out)
    else:
        sys.exit(run_ranks(server_args, port_args, plan, args.out))
