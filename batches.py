"""The batch plan runner.py times, sized for gpt-oss-20b (BF16) on one A100 80GB.

Each batch is a list with one length per request: prompt tokens for prefill,
tokens already in the KV cache for decode. A hybrid batch is a
(prefill, decode) pair of such lists.

Every phase combines a structured design, which moves S, M and N independently,
with a larger seeded random sample. fit.py cross-validates over all of them to
show how well a linear model predicts batches it was not fitted on. README.md
explains the design.
"""

import random

BUDGET = 8192  # --chunked-prefill-size: most tokens in one prefill or hybrid step
KV_TOKENS = 300_000  # below the 309,703-token sliding-window KV pool
MAX_DECODE = 256  # largest CUDA-graph decode batch
HYBRID_DECODE_TOKENS = 65_536  # hybrid attention cost grows with M_p * S_d; keep runs short
N_RANDOM = {"prefill": 500, "decode": 500, "hybrid": 400}
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


# Prefill: S = sum of prompt lengths, M = longest prompt, N = prompts.
PREFILL_BATCHES = [
    [length] * n
    for n in (1, 2, 4, 8, 16, 32, 64)
    for length in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
    if n * length <= BUDGET
]
PREFILL_BATCHES += [
    [long] + [short] * (n - 1)
    for n in (2, 4, 8, 16, 32)
    for long in (512, 1024, 2048, 4096, 6144)
    for short in (16, 64, 256)
    if long + short * (n - 1) <= BUDGET
]
PREFILL_BATCHES += [random_lengths(64, 16, 8192, (16, BUDGET)) for _ in range(N_RANDOM["prefill"])]

# Decode: S = sum of context lengths, M = longest context, N = requests.
DECODE_BATCHES = [
    [length] * n
    for n in (1, 2, 4, 8, 16, 32, 64, 128, 256)
    for length in (64, 256, 1024, 4096, 16384, 65536)
    if n * (length + 1) <= KV_TOKENS
]
DECODE_BATCHES += [
    [long] + [short] * (n - 1)
    for n in (8, 32, 128, 256)
    for long in (8192, 32768, 98304)
    for short in (128, 1024)
    if long + (short + 1) * (n - 1) + 1 <= KV_TOKENS
]
DECODE_BATCHES += [
    random_lengths(MAX_DECODE, 32, 131072, (32, KV_TOKENS - MAX_DECODE), log_total=True)
    for _ in range(N_RANDOM["decode"])
]


# Hybrid: a prefill part and a decode part in one step. First every prefill
# part crossed with every decode part (each part is also timed on its own
# above), then random pairs. Each decode uses one token of the step's budget.
def random_hybrid():
    decode = random_lengths(MAX_DECODE, 32, 16384, (32, HYBRID_DECODE_TOKENS), log_total=True)
    prefill = random_lengths(16, 16, 8192, (16, BUDGET - len(decode)))
    return prefill, decode


PREFILL_PARTS = [[256], [1024], [4096], [1024] * 4, [256] * 8, [128] * 32]
DECODE_PARTS = [[length] * n for n in (1, 8, 32, 128) for length in (512, 2048)]
HYBRID_BATCHES = [
    (prefill, decode)
    for prefill in PREFILL_PARTS
    for decode in DECODE_PARTS
    if sum(prefill) + len(decode) <= BUDGET
]
HYBRID_BATCHES += [random_hybrid() for _ in range(N_RANDOM["hybrid"])]
