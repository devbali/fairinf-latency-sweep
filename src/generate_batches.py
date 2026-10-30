"""Write the batch plans runner.py times, for gpt-oss-20b (BF16) on one A100 80GB.

- batches.json: the full plan, from the designs and seeded random draws below.
- sanity_batches.json: 88 batches of the full plan, for sanity_latency.py.

The plans list every batch as data, so they never change behind anyone's back.
To change one, edit this file and rerun it:

    python src/generate_batches.py

Each batch is a list with one length per request: prompt tokens for prefill,
tokens already in the KV cache for decode. A hybrid batch is a [prefill,
decode] pair of such lists. Every phase of the full plan combines structured
designs, which move S, M and N independently, with a larger seeded random
sample.

The full plan starts with the 1,631 batches of results/gptoss-clean-20261028,
drawn exactly as they were then, so later runs compare with that one batch by
batch. Denser designs and more random batches follow. README.md explains the
design.
"""

import json
import os
import random

BUDGET = 8192  # --chunked-prefill-size: most tokens in one prefill or hybrid step
KV_TOKENS = 300_000  # below the 309,703-token sliding-window KV pool
MAX_DECODE = 256  # largest CUDA-graph decode batch
HYBRID_DECODE_TOKENS = 65_536  # hybrid attention cost grows with M_p * S_d; keep runs short
# Random batches per phase: those of results/gptoss-clean-20261028, then more.
N_RANDOM = {"prefill": 500, "decode": 500, "hybrid": 400}
N_MORE_RANDOM = {"prefill": 1000, "decode": 1000, "hybrid": 600}
HERE = os.path.dirname(os.path.abspath(__file__))
rng = random.Random(0)


def log_uniform(lo, hi):
    return int(round(lo * (hi / lo) ** rng.random()))


def random_lengths(max_n, lo, hi, total_range, log_total=False):
    """A random batch of up to max_n lengths in [lo, hi], summing into total_range.

    The total is drawn first (uniformly, or log-uniformly for ranges spanning
    orders of magnitude) and then split among the requests with lognormal
    weights, from nearly even (sigma 0) to one dominant request (sigma 2).
    """
    while True:
        n = log_uniform(1, max_n)
        t_lo, t_hi = max(total_range[0], lo * n), min(total_range[1], hi * n)
        if t_lo > t_hi:
            continue
        total = log_uniform(t_lo, t_hi) if log_total else rng.randint(t_lo, t_hi)
        sigma = rng.uniform(0, 2)
        weights = [rng.lognormvariate(0, sigma) for _ in range(n)]
        scale = total / sum(weights)
        batch = [min(hi, max(lo, round(w * scale))) for w in weights]
        if total_range[0] <= sum(batch) <= total_range[1]:
            return batch


def random_prefill():
    return random_lengths(64, 16, 8192, (16, BUDGET))


def random_decode():
    return random_lengths(MAX_DECODE, 32, 131072, (32, KV_TOKENS - MAX_DECODE), log_total=True)


def random_hybrid():
    decode = random_lengths(MAX_DECODE, 32, 16384, (32, HYBRID_DECODE_TOKENS), log_total=True)
    prefill = random_lengths(16, 16, 8192, (16, BUDGET - len(decode)))
    return [prefill, decode]


def prefill_fits(lens):
    return sum(lens) <= BUDGET


def decode_fits(lens):
    return sum(lens) + len(lens) <= KV_TOKENS  # each request also gets one new token


def hybrid_fits(prefill, decode):
    return sum(prefill) + len(decode) <= BUDGET  # each decode uses one token of the budget


def uniform(counts, lengths, fits):
    """n requests of one length, for every n in counts and length in lengths that fit."""
    return [[length] * n for n in counts for length in lengths if fits([length] * n)]


def skewed(counts, longs, shorts, fits):
    """One long request and n - 1 short ones."""
    return [
        [long] + [short] * (n - 1)
        for n in counts
        for long in longs
        for short in shorts
        if short < long and fits([long] + [short] * (n - 1))
    ]


def crossed(prefill_parts, decode_parts):
    """Every prefill part with every decode part, within the step's budget."""
    return [[p, d] for p in prefill_parts for d in decode_parts if hybrid_fits(p, d)]


def powers_of_two(lo, hi):
    return [2**k for k in range(lo.bit_length() - 1, hi.bit_length())]


# The plan of results/gptoss-clean-20261028. Only the random batches use rng,
# and they are drawn in the order they were then: prefill, decode, hybrid.
PREFILL = uniform((1, 2, 4, 8, 16, 32, 64), powers_of_two(16, 8192), prefill_fits)
PREFILL += skewed((2, 4, 8, 16, 32), (512, 1024, 2048, 4096, 6144), (16, 64, 256), prefill_fits)
PREFILL += [random_prefill() for _ in range(N_RANDOM["prefill"])]
DECODE = uniform((1, 2, 4, 8, 16, 32, 64, 128, 256), (64, 256, 1024, 4096, 16384, 65536), decode_fits)
DECODE += skewed((8, 32, 128, 256), (8192, 32768, 98304), (128, 1024), decode_fits)
DECODE += [random_decode() for _ in range(N_RANDOM["decode"])]
HYBRID_PARTS = (
    [[256], [1024], [4096], [1024] * 4, [256] * 8, [128] * 32],
    [[length] * n for n in (1, 8, 32, 128) for length in (512, 2048)],
)
HYBRID = crossed(*HYBRID_PARTS)
HYBRID += [random_hybrid() for _ in range(N_RANDOM["hybrid"])]
OLD = {"prefill": len(PREFILL), "decode": len(DECODE), "hybrid": len(HYBRID)}

# Denser designs. Prefill: one prompt every 16 tokens up to 256, every 32 up to
# 1,024 and every 128 up to the budget; more counts and lengths in between the
# powers of two; and more ways to pair one long prompt with short ones.
PREFILL += [[length] for length in [*range(16, 256, 16), *range(256, 1024, 32), *range(1024, BUDGET + 1, 128)]]
PREFILL += uniform(
    (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64),
    (16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192),
    prefill_fits,
)
PREFILL += skewed(
    (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64),
    (256, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144),
    (16, 32, 64, 128),
    prefill_fits,
)
# Decode: every batch size up to 16, where each request still adds a lot, then
# steps of a quarter of a power of two; contexts at every power of two.
DECODE += uniform(
    (*range(1, 17), 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256),
    powers_of_two(64, 131072),
    decode_fits,
)
DECODE += skewed(
    (2, 4, 8, 16, 32, 64, 128, 256),
    (8192, 16384, 32768, 65536, 98304, 131072),
    (64, 128, 512, 1024, 4096),
    decode_fits,
)
# Hybrid: more prefill parts, crossed with decode parts of up to 64k context tokens.
HYBRID += crossed(
    [[16], [64], [256], [1024], [2048], [4096], [6144], [256] * 4, [1024] * 4, [256] * 8, [64] * 16,
     [128] * 32, [32] * 64],
    [
        [length] * n
        for n in (1, 2, 4, 8, 16, 32, 64, 128, 256)
        for length in (128, 512, 2048, 8192)
        if length * n <= HYBRID_DECODE_TOKENS
    ],
)
# More random batches, drawn after the old ones from the same rng.
PREFILL += [random_prefill() for _ in range(N_MORE_RANDOM["prefill"])]
DECODE += [random_decode() for _ in range(N_MORE_RANDOM["decode"])]
HYBRID += [random_hybrid() for _ in range(N_MORE_RANDOM["hybrid"])]

# The sanity plan: a short sweep, about 1.5 minutes of timing after the model
# loads, that covers each regime the models describe:
# - prefill: one prompt from the CPU floor up to the full budget, the budget
#   split into 4 to 64 equal prompts, and one long prompt with short ones;
# - decode: 1 to 256 requests, and contexts from 64 tokens to 98k;
# - hybrid: every prefill part with every decode part of the old plan.
SANITY = {
    "prefill": [[length] for length in powers_of_two(16, 8192)]
    + [[length] * (BUDGET // length) for length in (128, 256, 512, 1024, 2048)]
    + [[6144] + [16] * 31, [4096] + [64] * 31, [2048] + [256] * 15, [1024] + [16] * 15, [512] + [256] * 7],
    "decode": [[1024] * n for n in (1, 2, 4, 8, 16, 32, 64, 128, 256)]
    + [[64], [4096], [65536], [64] * 256, [4096] * 64, [16384] * 16, [65536] * 4]
    + [[98304] + [128] * 7, [98304] + [1024] * 31, [32768] + [1024] * 127, [8192] + [128] * 255],
    "hybrid": crossed(*HYBRID_PARTS),
}


def unique(batches):
    """The batches without repeats, each kept where it first appears."""
    seen = set()
    kept = []
    for batch in batches:
        key = json.dumps(batch)
        if key not in seen:
            seen.add(key)
            kept.append(batch)
    return kept


def write(path, about, phases):
    """A JSON plan: an "about" note, then one list per phase with one batch per line."""
    lists = [
        f'  "{phase}": [\n' + ",\n".join(f"    {json.dumps(batch)}" for batch in batches) + "\n  ]"
        for phase, batches in phases.items()
    ]
    with open(path, "w") as out:
        out.write("{\n" + f'  "about": {json.dumps(about)},\n' + ",\n".join(lists) + "\n}\n")


if __name__ == "__main__":
    plan = {"prefill": unique(PREFILL), "decode": unique(DECODE), "hybrid": unique(HYBRID)}
    for phase, batches in SANITY.items():
        assert all(batch in plan[phase] for batch in batches), f"a sanity {phase} batch is not in the plan"
    write(
        os.path.join(HERE, "batches.json"),
        "The batch plan runner.py times, for gpt-oss-20b (BF16) on one A100 80GB. Written by "
        "generate_batches.py, which explains the design: edit that and rerun it instead of editing this "
        "file. Each batch lists one length per request: prompt tokens for prefill, tokens already in the "
        "KV cache for decode. A hybrid batch is a [prefill, decode] pair of such lists. The first "
        f"{OLD['prefill']} prefill, {OLD['decode']} decode and {OLD['hybrid']} hybrid batches are the plan "
        "of results/gptoss-clean-20261028.",
        plan,
    )
    write(
        os.path.join(HERE, "sanity_batches.json"),
        "The short plan sanity_latency.py times: batches of batches.json that cover each regime the latency "
        "models describe, about 1.5 minutes of timing on one A100. Written by generate_batches.py.",
        SANITY,
    )
    for phase, batches in plan.items():
        print(f"{phase}: {len(batches):,} batches, the first {OLD[phase]} from results/gptoss-clean-20261028; "
              f"{len(SANITY[phase])} in the sanity plan")
