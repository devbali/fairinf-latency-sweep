"""Write a latency profile JSON for fairinf-sglang (FAIRINF_LATENCY_PROFILE=path.json).

The profile holds, per step kind, the pooled estimator at a percentile and the isolated estimator of it for n
clients, both made by this package's pooled() and isolated(); the fork only evaluates them. Term names are those
of steps.py, which the fork's latency_profiles.py shares.

Usage (from src/):
    python -m estimators.export_profile gpt-oss-20b-a100 --percentile 95 --n 4 --out profile.json
"""

import argparse
import json

from . import decode, decode_tp4, hybrid, prefill, prefill_tp4
from .estimator import NO_FLOOR, isolated, pooled
from .steps import DECODE_KNOTS_LARGE, step_model

KINDS = ("prefill", "decode", "hybrid")

MODELS = {
    "gpt-oss-20b-a100": lambda: (prefill.MODEL, decode.MODEL, hybrid.MODEL),
    # Hybrid at TP 4 has no module of its own (no experiment runs it); the run measured it and a profile needs one.
    "gpt-oss-120b-tp4": lambda: (prefill_tp4.MODEL, decode_tp4.MODEL,
                                 step_model("hybrid", "gptoss-120b-tp4-20261103", "hybrid, gpt-oss-120b, TP 4")),
    # The same sweep on the FairInference experiments setup (the fork after its upstream sync, torch 2.14.1):
    # decode runs 1.2-1.9x slower there than on the runs above, prefill and hybrid faster.
    # Decode is fitted on that run and the large decode steps after it (up to 512 requests and each model's
    # KV cache), where the main plan has little or no data.
    **{name: (lambda run=run, large=large, what=what: (
                  step_model("prefill", run, f"prefill, {what}, experiments' setup"),
                  step_model("decode", (run, large), f"decode, {what}, experiments' setup", DECODE_KNOTS_LARGE),
                  step_model("hybrid", run, f"hybrid, {what}, experiments' setup")))
       for name, run, large, what in (
           ("gpt-oss-20b-a100-synced", "gptoss-20b-final-setup-20261124",
            "gptoss-20b-decode-large-final-setup-20261124", "gpt-oss-20b, 1 A100"),
           ("gpt-oss-120b-tp4-synced", "gptoss-120b-tp4-final-setup-20261124",
            "gptoss-120b-tp4-decode-large-final-setup-20261124", "gpt-oss-120b, TP 4"))},
}

# The decode batches the largest pooled decode step is taken over: the range the sweep measured decode on
# (src/batches.json, the same for both runs): up to 256 requests, 299,118 tokens in all, a longest request of
# 131,072 tokens. The fits hold there; past it they extrapolate.
DECODE_RANGE = {"max_batch": 256, "max_tokens": 299_118, "max_context": 131_072}
# The -synced models measured decode further: up to 299,912 (20b) and 464,909 (120b) tokens in all.
DECODE_RANGES = {"gpt-oss-20b-a100-synced": {**DECODE_RANGE, "max_tokens": 299_912},
                 "gpt-oss-120b-tp4-synced": {**DECODE_RANGE, "max_tokens": 464_909}}


def _batches(n, s, m):
    """Two batches of n requests, s tokens, longest m: the rest split evenly (least sum l^2) and as many at m as fit
    (most sum l^2). With one of each, any sum l^2 term's extreme over the batches with that (n, s, m) is covered."""
    rest = s - m
    even = [m] + [rest // (n - 1)] * (n - 1) if n > 1 else [m]
    k = min(n - 1, rest // m) if n > 1 else 0
    left = n - 1 - k
    tail = rest - k * m
    packed = [m] * (k + 1) + ([tail // left] * left if left else [])
    return [b for b in (even, packed) if all(x >= 1 for x in b)]


def max_decode_ms(est, max_batch, max_tokens, max_context, steps=48):
    """The largest step time of the decode estimator over the decode range:
    max over N <= max_batch, N <= S <= max_tokens, ceil(S / N) <= M <= min(S, max_context) of T(N, S, M, sum l^2),
    on a grid (S and M geometric, N at every CUDA-graph size, which is what decode pads N to)."""
    import numpy as np
    from .steps import GRAPH_BS
    best, arg = -np.inf, None
    for n in [b for b in GRAPH_BS if b <= max_batch]:
        for s in np.unique(np.geomspace(n, max_tokens, steps).astype(int)):
            lo, hi = -(-s // n), min(s, max_context)
            if lo > hi:
                continue
            for m in np.unique(np.geomspace(lo, hi, steps).astype(int)):
                for b in _batches(n, int(s), int(m)):
                    t = est(b)
                    if t > best:
                        best, arg = t, (n, int(s), int(m))
    return float(best), arg


def model_json(est):
    """T = max(floor_ms, terms) + add_ms, as the fork evaluates every model of a profile."""
    return {"terms": {t: float(c) for t, c in zip(est.model.terms, est.coef)},
            "floor_ms": 0.0 if est.floor == NO_FLOOR else float(est.floor),
            "add_ms": float(est.add),
            "label": est.label}


def profile(name, percentile, n, refit=False, static_gap_ms=10.0):
    ests = [pooled(m, percentile, refit) for m in MODELS[name]()]
    decode_range = DECODE_RANGES.get(name, DECODE_RANGE)
    floor, (n_, s_, m_) = max_decode_ms(ests[1], **decode_range)
    return {
        # 2: every model is max(floor_ms, terms) + add_ms, the isolated ones complete (floor and gap inside).
        "format": 2,
        "source": f"fairinf-latency-sweep estimators: {name}, pooled p{percentile:g}"
                  f"{' refit' if refit else ' shifted'}, isolated n={n}",
        "pooled": {k: model_json(e) for k, e in zip(KINDS, ests)},
        # The isolated models, complete: an isolated step that decodes (decode, hybrid) is floored at D^MT,
        # the largest pooled decode step over the measured decode range (the paper's D^MT_B <= D^ISO_B), and
        # takes the static gap on top: T = max(D^MT, terms) + gap. Hybrid's floor stays at least its fitted one.
        "isolated": {"n": n,
                     "prefill": model_json(isolated(ests[0], n)),
                     "decode": model_json(isolated(ests[1], n, floor=floor, add=static_gap_ms)),
                     "hybrid": model_json(isolated(ests[2], n, floor=floor, add=static_gap_ms))},
        # For reference: the isolated decode and hybrid models above already include both.
        "max_pooled_decode_ms": floor,
        "max_pooled_decode_at": {"N": n_, "S": s_, "M": m_, "range": decode_range},
        "static_gap_ms": static_gap_ms,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", choices=sorted(MODELS))
    ap.add_argument("--percentile", type=float, default=95)
    ap.add_argument("--n", type=int, required=True, help="clients; the isolated model is for 1/n of the GPU")
    ap.add_argument("--refit", action="store_true", help="quantile-regression refit instead of an intercept shift")
    ap.add_argument("--static-gap-ms", type=float, default=10.0, help="fixed time added to every isolated decode step")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    with open(args.out, "w") as f:
        json.dump(profile(args.model, args.percentile, args.n, args.refit, args.static_gap_ms), f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
