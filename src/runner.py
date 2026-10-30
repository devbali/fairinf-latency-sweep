"""Time single prefill, decode and hybrid forward passes of SGLang on fake batches.

Same low-level path as `python -m sglang.bench_one_batch`: no server,
scheduler or radix cache, just batches built by hand and run through
SGLang's ModelRunner on one GPU. The batches to time come from a JSON plan
(batches.json by default).
"""

import argparse
import json
import logging
import os
import time
import uuid

import numpy as np
import torch

from sglang.bench_one_batch import TreeCacheNamespace
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

WARMUP_RUNS = 1  # per batch, not recorded
SAMPLES = 3  # per batch


def load_model(server_args):
    """Load the weights, allocate the KV cache and capture CUDA graphs on GPU 0."""
    _set_envs_and_config(server_args)
    initialize_moe_config(server_args)  # MoE layers read their kernel choices from it
    return ModelRunner(
        model_config=ModelConfig.from_server_args(server_args),
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=0,
        tp_rank=0,
        tp_size=1,
        moe_ep_rank=0,
        moe_ep_size=1,
        pp_rank=0,
        pp_size=1,
        nccl_port=PortArgs.init_new(server_args).nccl_port,
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
        req = Req(
            rid=uuid.uuid4().hex,
            origin_input_text="",
            origin_input_ids=np.random.randint(0, 10000, n).tolist(),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=16),
        )
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(n)
        reqs.append(req)
    return ScheduleBatch.init_new(
        reqs=reqs,
        req_to_token_pool=model_runner.req_to_token_pool,
        token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        # bench_one_batch's stand-in for the radix cache: allocation only, no prefix reuse
        tree_cache=TreeCacheNamespace(
            page_size=model_runner.server_args.page_size,
            device=model_runner.device,
            token_to_kv_pool_allocator=model_runner.token_to_kv_pool_allocator,
        ),
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
    batch.prepare_for_extend()
    # Stand-in for the token each request's prefill would have sampled,
    # recorded on the requests too, as the scheduler does after a prefill.
    next_token_ids = np.random.randint(0, 10000, len(lens))
    for req, token_id in zip(batch.reqs, next_token_ids.tolist()):
        req.output_ids.append(token_id)
    batch.output_ids = torch.tensor(next_token_ids, device=model_runner.device)
    return batch


@torch.no_grad()
def forward(model_runner, batch):
    """One forward pass over a prepared batch; samples one token per request."""
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
    batch.prepare_for_extend()
    forward(model_runner, batch)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def time_decode(model_runner, lens):
    """Seconds for one decode step; request i has lens[i] tokens of context."""
    clear_kv_cache(model_runner)
    batch = decode_batch(model_runner, lens)
    torch.cuda.synchronize()
    start = time.perf_counter()
    batch.prepare_for_decode()
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
    """
    clear_kv_cache(model_runner)
    running = decode_batch(model_runner, decode_lens)
    batch = fake_batch(model_runner, prefill_lens)
    torch.cuda.synchronize()
    start = time.perf_counter()
    batch.prepare_for_extend()
    running.prepare_for_decode()
    batch.mix_with_running(running)
    forward(model_runner, batch)
    torch.cuda.synchronize()
    return time.perf_counter() - start


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
    phases = [
        ("prefill", time_prefill, plan.get("prefill", [])),
        ("decode", time_decode, plan.get("decode", [])),
        (
            "hybrid",
            lambda model_runner, batch: time_hybrid(model_runner, *batch),
            plan.get("hybrid", []),
        ),
    ]

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model_runner = load_model(ServerArgs.from_cli_args(args))
    # The fork logs a warning on every request-slot allocation (POOL_ALLOC in
    # mem_cache/memory_pool.py). Silence it so the I/O stays out of the timings.
    logging.getLogger().setLevel(logging.ERROR)

    with open(args.out, "a") as out:
        for phase, time_fn, batches in phases:
            for i, batch in enumerate(batches):
                for _ in range(WARMUP_RUNS):
                    time_fn(model_runner, batch)
                for sample in range(SAMPLES):
                    latency_s = time_fn(model_runner, batch)
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
