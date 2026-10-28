"""Time one prefill and one decode forward pass of SGLang on fake batches.

Same low-level path as `python -m sglang.bench_one_batch`: no server,
scheduler or radix cache, just batches built by hand and run through
SGLang's ModelRunner on one GPU. The batches to time come from a separate
file (batches.py by default).
"""

import argparse
import json
import logging
import runpy
import time

import numpy as np
import torch

from sglang.bench_one_batch import TreeCacheNamespace
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.entrypoints.engine import _set_envs_and_config
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


def fake_batch(model_runner, lens):
    """One request per entry of lens, each with that many random prompt tokens.

    Starts from an empty KV cache, so calls don't depend on each other.
    """
    model_runner.req_to_token_pool.clear()
    model_runner.token_to_kv_pool_allocator.clear()
    reqs = []
    for i, n in enumerate(lens):
        req = Req(
            rid=str(i),
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


@torch.no_grad()
def forward(model_runner, batch):
    """One forward pass over a prepared batch; samples one token per request."""
    forward_batch = ForwardBatch.init_new(batch.get_model_worker_batch(), model_runner)
    logits_output = model_runner.forward(forward_batch).logits_output
    return model_runner.sample(logits_output, forward_batch)


def time_batch(model_runner, lens):
    """Prefill a fake batch, then decode one token per request.

    Returns (prefill_s, decode_s): wall time from batch preparation through
    sampling, with the GPU synced on both ends. The decode step attends over
    lens[i] + 1 tokens for request i.
    """
    batch = fake_batch(model_runner, lens)

    torch.cuda.synchronize()
    start = time.perf_counter()
    batch.prepare_for_extend()
    next_token_ids = forward(model_runner, batch)
    torch.cuda.synchronize()
    prefill_s = time.perf_counter() - start

    start = time.perf_counter()
    batch.output_ids = next_token_ids
    batch.prepare_for_decode()
    forward(model_runner, batch)
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - start

    return prefill_s, decode_s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.add_argument(
        "--batches",
        default="batches.py",
        help="Python file defining BATCHES, the fake batches to time",
    )
    parser.add_argument(
        "--out", default="results.jsonl", help="appends one JSON line per sample"
    )
    args = parser.parse_args()
    batches = runpy.run_path(args.batches)["BATCHES"]

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    model_runner = load_model(ServerArgs.from_cli_args(args))
    # The fork logs a warning on every request-slot allocation (POOL_ALLOC in
    # mem_cache/memory_pool.py). Silence it so the I/O stays out of the timings.
    logging.getLogger().setLevel(logging.ERROR)

    with open(args.out, "a") as out:
        for i, lens in enumerate(batches):
            for _ in range(WARMUP_RUNS):
                time_batch(model_runner, lens)
            for sample in range(SAMPLES):
                prefill_s, decode_s = time_batch(model_runner, lens)
                row = {
                    "batch": lens,
                    "sample": sample,
                    "prefill_s": prefill_s,
                    "decode_s": decode_s,
                }
                print(json.dumps(row), file=out, flush=True)
                print(
                    f"batch {i + 1}/{len(batches)} ({len(lens)} requests, "
                    f"{sum(lens)} tokens) sample {sample + 1}/{SAMPLES}: "
                    f"prefill {prefill_s * 1e3:.2f} ms, decode {decode_s * 1e3:.2f} ms"
                )
