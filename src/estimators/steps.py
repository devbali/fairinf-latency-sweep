"""Step-latency models (prefill, decode, hybrid) on runner.py results: degree 2 + sum l^2, as in compare_fits.py.

Terms are the constant, N, S and M (degree 1), every square and product of them, and sum l^2 (degree 2). Decode
also has the bends (N - k)+ at each power of two (degree 1), with N padded to its CUDA graph; prefill and hybrid
are fitted with the floor. A hybrid step uses all six variables of its two parts and needs both parts non-empty
(a step with an empty part is a prefill or decode step: use those models).
"""

import itertools
import json
import os
from functools import lru_cache

import numpy as np

from .estimator import Model

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")
# The decode CUDA-graph sizes SGLang captures (up to 512 on both setups measured); a decode step pads N to the next.
GRAPH_BS = [1, 2, 4, 8, 12, 16, 24] + list(range(32, 257, 8)) + list(range(272, 513, 16))
DECODE_KNOTS = (2, 4, 8, 16, 32, 64, 128)
# The large decode plans reach 512 requests, where decode bends again.
DECODE_KNOTS_LARGE = DECODE_KNOTS + (256, 384)


def _variables(phase, batch):
    if phase == "hybrid":
        p, d = batch
        if not p or not d:
            raise ValueError("a hybrid step needs both a prefill and a decode part")
        return {"N_p": len(p), "S_p": sum(p), "M_p": max(p), "N_d": len(d), "S_d": sum(d), "M_d": max(d)}, \
            sum(x * x for x in p)
    v = {"N": len(batch), "S": sum(batch), "M": max(batch)}
    if phase == "decode":
        v["N"] = next((bs for bs in GRAPH_BS if bs >= v["N"]), v["N"])
    return v, sum(x * x for x in batch)


def _terms(phase, batch, knots=DECODE_KNOTS):
    """{term: (value, degree)}"""
    v, l2 = _variables(phase, batch)
    t = {"const": (1.0, 0)} | {k: (x, 1) for k, x in v.items()}
    for a, b in itertools.combinations_with_replacement(list(v), 2):
        t[f"{a}^2" if a == b else f"{a}*{b}"] = (v[a] * v[b], 2)
    t["sum l_p^2" if phase == "hybrid" else "sum l^2"] = (l2, 2)
    if phase == "decode":
        for k in knots:
            t[f"(N-{k})+"] = (max(0, v["N"] - k), 1)
    return t


@lru_cache(maxsize=None)
def _data(phase, run, knots=DECODE_KNOTS):
    """The samples of one run, or of several (a tuple of runs) together."""
    X, y, group = [], [], []
    runs = run if isinstance(run, tuple) else (run,)
    lines = (line for r in runs for line in open(os.path.join(RESULTS, r, "results.jsonl")))
    for line in lines:
        row = json.loads(line)
        if row["phase"] != phase:
            continue
        if phase == "hybrid" and not (row["batch"][0] and row["batch"][1]):
            continue
        X.append([value for value, _ in _terms(phase, row["batch"], knots).values()])
        y.append(1e3 * row["latency_s"])
        group.append(json.dumps(row["batch"]))
    return np.array(X, dtype=float), np.array(y), np.array(group)


def step_model(phase, run, name, knots=DECODE_KNOTS):
    """The degree 2 + sum l^2 model of one phase ("prefill", "decode" or "hybrid") of a runner.py run (or of
    a tuple of runs). Its input is a batch as runner.py takes it: a list of lengths, or [prefill lengths,
    decode lengths]. knots are decode's bends in N."""
    knots = tuple(knots)
    example = [[16], [16]] if phase == "hybrid" else [16]
    t = _terms(phase, example, knots)
    return Model(
        name=name,
        terms=tuple(t),
        degrees=tuple(d for _, d in t.values()),
        features=lambda batch: np.array([v for v, _ in _terms(phase, batch, knots).values()], dtype=float),
        data=lambda: _data(phase, run, knots),
        floor=phase in ("prefill", "hybrid"),
    )
