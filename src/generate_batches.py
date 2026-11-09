"""Write the batch plans runner.py times, for gpt-oss-20b (BF16) on one A100 80GB.

- batches.json: the full plan, from the designs and seeded random draws below.
- sanity_batches.json: 88 batches of the full plan, for sanity_latency.py.
- pd_batches.json: the plan pd_runner.py sends through a prefill and a decode
  server, to time the KV cache transfer between them.
- pd_sanity_batches.json: 20 batches and 1 burst of the PD plan, for
  sanity_pd.py.
- pd_load.json: probe batches and background loads for pd_load.py, to see
  what a transfer's time depends on besides its batch.
- hybrid_edges_batches.json: hybrid steps with an empty prefill or decode
  part, next to the full ones, to model a hybrid step piecewise.
- pd_load_sweep.json: the PD plan's batches under loads of several shapes, for
  pd_load.py, to see how each part of a transfer's time scales with the load.

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
N_PD_RANDOM = 200  # random batches of the PD plan
BURST_TOKENS = 65_536  # prompt tokens per burst of the PD plan: 8 full prefill steps
MAX_BURST = 512  # requests per burst of the PD plan
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

# The PD plan. A batch is prompts sent together, and its requests transfer
# their KV caches one after another, so the designs vary the number of
# requests N, their total length S and the longest one M independently, as the
# prefill designs do:
# - batches, sent one at a time so each batch's transfers run alone: one prompt
#   at every half octave from 16 to 8,192 tokens, and three longer than one
#   prefill step, which go in chunks; the uniform and skewed designs of the
#   prefill plan; 128 and 256 short prompts, the most requests per token; and
#   random batches drawn like the prefill plan's, after all of its draws.
# - bursts: more prompts of one length than one prefill step holds, so the
#   prefill and the transfer run at once and the slower of the two sets the
#   pace. That pace is how much load saturates the transfer.
PD_BATCHES = [[length] for length in sorted({*powers_of_two(16, BUDGET), *(3 * 2**k for k in range(3, 12))})]
PD_BATCHES += [[12288], [16384], [32768]]
PD_BATCHES += uniform((1, 2, 4, 8, 16, 32, 64), powers_of_two(16, BUDGET), prefill_fits)
PD_BATCHES += uniform((128, 256), (16, 32), prefill_fits)
PD_BATCHES += skewed((2, 4, 8, 16, 32), (512, 1024, 2048, 4096, 6144), (16, 64, 256), prefill_fits)
PD_BATCHES += [random_prefill() for _ in range(N_PD_RANDOM)]
PD_BURSTS = [[length] * min(MAX_BURST, BURST_TOKENS // length) for length in powers_of_two(16, BUDGET)]

# The PD sanity plan: about 2 minutes of sending after the servers start.
# - one prompt alone, 16 to 8,192 tokens: what moving the bytes costs;
# - 2 to 256 short prompts, and the budget split into 2 to 8 prompts: the cost
#   per request of a step;
# - a long prompt with short ones;
# - one burst of 64 prompts of 1,024 tokens, where the transfer has to keep up
#   with prefill.
PD_SANITY = {
    "batches": [[length] for length in (16, 128, 512, 1024, 2048, 4096, 8192)]
    + [[128] * n for n in (2, 4, 8, 16, 32, 64)]
    + [[16] * 128, [16] * 256, [1024] * 8, [2048] * 4, [4096] * 2]
    + [[6144] + [16] * 31, [2048] + [256] * 15],
    "bursts": [[1024] * 64],
}

# The PD load plan, for pd_load.py: what a transfer's time depends on besides
# its batch. The same probe batches go out one at a time in every scenario,
# while background load keeps the prefill server, the decode server, both or
# neither busy. A load is clients that each send one request at a time, again
# and again, until the scenario ends:
# - prefill: two clients with 2,048-token prompts and one output token, so a
#   prefill step is always queued (the prefill server never decodes) and the
#   decode server only receives KV;
# - decode: 32 clients with 256-token prompts and 6,000 output tokens, so the
#   decode server always runs a decode step and its KV cache stays far from
#   full (32 x 6,256 of 309,703 tokens);
# - pressure: 32 clients with 1,024-token prompts and 1,500 output tokens, more
#   than a decode server started with --max-total-tokens 40000 holds, so it has
#   to make requests wait and retract running ones.
# The probes: one prompt of 128, 1,024 and 4,096 tokens, 4 and 16 of 128, and
# 8 of 1,024, each sent 8 times per scenario (4 under pressure, for at most
# 3 minutes).
PD_LOAD = {
    "probes": [[128], [1024], [4096], [128] * 4, [128] * 16, [1024] * 8],
    "loads": [
        {"name": "prefill", "clients": 2, "len": 2048, "new_tokens": 1, "warm_s": 2},
        {"name": "decode", "clients": 32, "len": 256, "new_tokens": 6000, "warm_s": 5},
        {"name": "pressure", "clients": 32, "len": 1024, "new_tokens": 1500, "warm_s": 5},
    ],
    "scenarios": [
        {"name": "idle", "loads": [], "samples": 8},
        {"name": "prefill", "loads": ["prefill"], "samples": 8},
        {"name": "decode", "loads": ["decode"], "samples": 8},
        {"name": "both", "loads": ["prefill", "decode"], "samples": 8},
        {"name": "pressure", "loads": ["pressure"], "samples": 4, "max_s": 180},
    ],
}

# The PD load sweep, for pd_load.py: how each part of a transfer's time scales
# with the work around it. The probes are the PD plan's batches, each sent once
# per scenario. Each load is one shape of one server's work:
# - prefill: two clients with 512- or 2,048-token prompts, or one with 8,192
#   (the whole budget), and one output token, so a prefill step of that size
#   is always queued;
# - decode: 8, 32 or 128 clients with 256-token prompts, or 32 with 4,096, so
#   the decode step runs from about 12 to 30 ms. Their longest contexts, 8 and
#   32 x 6,256, 128 x 1,756 and 32 x 5,596 tokens, stay far below the decode
#   server's KV cache. "spread" shortens each client's first request, so the
#   clients' requests end at spread-out times and the decode batch holds steady.
# The scenarios: idle, each load alone, and three pairs: the earlier runs' pair,
# the shortest prefill steps with the largest decode batch, and the longest with
# the smallest.
PD_LOAD_SWEEP = {
    "probes": PD_BATCHES,
    "loads": [
        {"name": "prefill-512", "clients": 2, "len": 512, "new_tokens": 1, "warm_s": 2},
        {"name": "prefill-2048", "clients": 2, "len": 2048, "new_tokens": 1, "warm_s": 2},
        {"name": "prefill-8192", "clients": 1, "len": 8192, "new_tokens": 1, "warm_s": 2},
        {"name": "decode-8", "clients": 8, "len": 256, "new_tokens": 6000, "warm_s": 5, "spread": True},
        {"name": "decode-32", "clients": 32, "len": 256, "new_tokens": 6000, "warm_s": 5, "spread": True},
        {"name": "decode-128", "clients": 128, "len": 256, "new_tokens": 1500, "warm_s": 10, "spread": True},
        {"name": "decode-32-long", "clients": 32, "len": 4096, "new_tokens": 1500, "warm_s": 10, "spread": True},
    ],
    "scenarios": [{"name": "idle", "loads": [], "samples": 1}]
    + [{"name": load, "loads": [load], "samples": 1}
       for load in ("prefill-512", "prefill-2048", "prefill-8192", "decode-8", "decode-32", "decode-128",
                    "decode-32-long")]
    + [{"name": f"{p}+{d}", "loads": [p, d], "samples": 1}
       for p, d in (("prefill-2048", "decode-32"), ("prefill-512", "decode-128"), ("prefill-8192", "decode-8"))],
}

# The hybrid edges plan: hybrid steps whose prefill or decode part is empty,
# next to the full ones they border, timed in one run, to model a hybrid step
# piecewise. With one part empty the scheduler mixes nothing, and runner.py
# times the step it runs instead: a prefill step, or a decode step as a CUDA
# graph. Every prefill part of the denser hybrid design with every decode part
# (up to 64k context tokens) and with none, and every decode part with none.
HYBRID_EDGE_PREFILL = [[16], [64], [256], [1024], [2048], [4096], [6144], [256] * 4, [1024] * 4, [256] * 8,
                       [64] * 16, [128] * 32, [32] * 64]
HYBRID_EDGE_DECODE = [[length] * n for n in (1, 2, 4, 8, 16, 32, 64, 128, 256) for length in (128, 512, 2048, 8192)
                      if length * n <= HYBRID_DECODE_TOKENS]
HYBRID_EDGES = {
    "hybrid": [[p, d] for p in HYBRID_EDGE_PREFILL + [[]] for d in HYBRID_EDGE_DECODE + [[]]
               if (p or d) and hybrid_fits(p, d)],
}

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
    pd_plan = {"batches": unique(PD_BATCHES), "bursts": unique(PD_BURSTS)}
    for phase, batches in PD_SANITY.items():
        assert all(batch in pd_plan[phase] for batch in batches), f"a PD sanity {phase} batch is not in the PD plan"
    write(
        os.path.join(HERE, "pd_batches.json"),
        "The plan pd_runner.py sends through a prefill and a decode server. Each batch lists one prompt "
        "length per request; its prompts are sent together. Batches go one at a time, so each one's KV "
        "transfers run alone. Bursts hold more prompts of one length than one prefill step, so prefill and "
        "transfer run at once. Written by generate_batches.py, which explains the design.",
        pd_plan,
    )
    write(
        os.path.join(HERE, "pd_sanity_batches.json"),
        "The short plan sanity_pd.py sends through a prefill and a decode server: batches and a burst of "
        "pd_batches.json that cover what the PD transfer model describes, about 2 minutes of sending. Written "
        "by generate_batches.py.",
        PD_SANITY,
    )
    write(
        os.path.join(HERE, "pd_load.json"),
        "The plan pd_load.py runs through a prefill and a decode server: probe batches, sent one at a time in "
        "each scenario while its loads keep the servers busy. A load is clients that each send one request of "
        "len prompt tokens and new_tokens output tokens at a time, until the scenario ends; warm_s is how long "
        "it runs before the first probe. The pressure scenario needs a decode server started with "
        "--max-total-tokens 40000. Written by generate_batches.py, which explains the design.",
        PD_LOAD,
    )
    write(
        os.path.join(HERE, "hybrid_edges_batches.json"),
        "Hybrid steps whose prefill or decode part is empty, with the full ones they border, for runner.py: "
        "every prefill part of the denser hybrid design with every decode part and with none, and every "
        "decode part with none. Each batch is a [prefill, decode] pair; runner.py times an empty part's step "
        "as the prefill or decode step the scheduler runs instead. Written by generate_batches.py.",
        HYBRID_EDGES,
    )
    write(
        os.path.join(HERE, "pd_load_sweep.json"),
        "The load sweep pd_load.py runs through a prefill and a decode server: the PD plan's batches as the "
        "probes, each sent once per scenario, under loads of several shapes, one server's work each. A load is "
        "clients that each send one request of len prompt tokens and new_tokens output tokens at a time, until "
        "the scenario ends; warm_s is how long it runs before the first probe, and spread shortens each "
        "client's first request so their requests end at spread-out times. Written by generate_batches.py, "
        "which explains the design.",
        {**PD_LOAD_SWEEP, "probes": unique(PD_LOAD_SWEEP["probes"])},
    )
    for phase, batches in plan.items():
        print(f"{phase}: {len(batches):,} batches, the first {OLD[phase]} from results/gptoss-clean-20261028; "
              f"{len(SANITY[phase])} in the sanity plan")
    for phase, batches in pd_plan.items():
        print(f"PD {phase}: {len(batches):,}, {sum(map(len, batches)):,} requests; "
              f"{len(PD_SANITY[phase])} in the PD sanity plan")
