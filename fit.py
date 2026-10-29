"""Fit piecewise-linear latency models to runner.py results, and report how well they fit.

Each batch's latency is the median of its samples. Prefill and hybrid steps are
predicted as max(floor, linear): a CPU-bound floor for small steps, linear GPU
time above it. Decode is piecewise linear in its padded batch size. README.md
explains the models.

For every phase this prints the coefficients with bootstrap 95% intervals, then
R², RMSE and relative error in-sample and on held-out batches from 5-fold
cross-validation, overall and by batch size.

Usage: python fit.py RESULTS.jsonl [MORE.jsonl ...]
"""

import json
import random
import sys
from collections import defaultdict

import numpy as np

# Decode batches run as CUDA graphs, padded up to these sizes.
GRAPH_BS = [1, 2, 4, 8, 12, 16, 24] + list(range(32, 257, 8))
# Decode time is piecewise linear in the padded batch size, bending at each power of two.
DECODE_KNOTS = [2, 4, 8, 16, 32, 64, 128]
FOLDS = 5
BOOTSTRAP = 200


def stats(lens):
    return {"N": len(lens), "S": sum(lens), "M": max(lens)}


def features(phase, batch):
    if phase == "prefill":
        f = stats(batch)
        # Triton extend attention runs ceil(M / BLOCK_M) query blocks for every
        # request, each looping over up to that request's whole prompt (see README.md).
        f["N*M"] = f["N"] * f["M"]
        f["S*M"] = f["S"] * f["M"]
        f["sum l^2"] = sum(length * length for length in batch)
        return f
    if phase == "decode":
        f = stats(batch)
        f["N"] = next((bs for bs in GRAPH_BS if bs >= f["N"]), f["N"])  # padded batch
        for knot in DECODE_KNOTS:
            f[f"(N-{knot})+"] = max(0, f["N"] - knot)
        return f
    p, d = stats(batch[0]), stats(batch[1])
    f = {f"{k}_p": v for k, v in p.items()} | {f"{k}_d": v for k, v in d.items()}
    # The prefill part pays the same attention terms as a prefill step, and every
    # decode also gets ceil(M_p / BLOCK_M) blocks, each looping over its context.
    f["N_p*M_p"] = p["N"] * p["M"]
    f["S_p*M_p"] = p["S"] * p["M"]
    f["sum l_p^2"] = sum(length * length for length in batch[0])
    f["N_d*M_p"] = d["N"] * p["M"]
    f["S_d*M_p"] = d["S"] * p["M"]
    return f


def step_size(phase, batch):
    """Tokens a prefill or hybrid step computes, or requests in a decode step."""
    if phase == "hybrid":
        return sum(batch[0]) + len(batch[1])
    return sum(batch) if phase == "prefill" else len(batch)


def lstsq(X, y):
    scale = np.abs(X).max(axis=0)
    scale[scale == 0] = 1
    return np.linalg.lstsq(X / scale, y, rcond=None)[0] / scale


def fit(phase, X, y):
    """Least squares for T = max(floor, X @ w). Returns (floor, w).

    Decode runs as CUDA graphs and has no floor. For prefill and hybrid, fit w
    to the steps above the floor and the floor to the rest, then move each step
    to the side the model now puts it on, until none moves.
    """
    if phase == "decode":
        return 0.0, lstsq(X, y)
    above = y > 1.2 * y.min()
    for _ in range(50):
        w = lstsq(X[above], y[above])
        floor = y[~above].mean() if not above.all() else 0.0
        if ((X @ w > floor) == above).all():
            break
        above = X @ w > floor
    return floor, w


def predict(model, X):
    floor, w = model
    return np.maximum(floor, X @ w)


def r2(y, pred):
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def relative(y, pred):
    err = np.abs(pred - y) / y
    return (
        f"error mean {100 * err.mean():.1f}%, median {100 * np.median(err):.1f}%, "
        f"p90 {100 * np.percentile(err, 90):.1f}%, max {100 * err.max():.1f}%"
    )


def describe(y, pred):
    rmse = np.sqrt(((pred - y) ** 2).mean())
    return f"RMSE {rmse:.2f} ms, MAE {np.abs(pred - y).mean():.2f} ms, {relative(y, pred)}"


def load(paths):
    """Every sample's latency in seconds, by phase and batch (as JSON)."""
    samples = defaultdict(list)
    for path in paths:
        for line in open(path):
            row = json.loads(line)
            samples[row["phase"], json.dumps(row["batch"])].append(row["latency_s"])
    return samples


def main(paths):
    samples = load(paths)
    for phase in ("prefill", "decode", "hybrid"):
        keys = sorted(key for key in samples if key[0] == phase)
        if not keys:
            continue
        batches = [json.loads(batch) for _, batch in keys]
        rows = [features(phase, batch) for batch in batches]
        X = np.array([[1.0] + list(r.values()) for r in rows])
        y = np.array([np.median(samples[key]) for key in keys]) * 1e3  # ms

        model = fit(phase, X, y)
        rng = np.random.default_rng(0)
        boot = []
        for _ in range(BOOTSTRAP):
            i = rng.integers(0, len(y), len(y))
            floor, w = fit(phase, X[i], y[i])
            boot.append(np.r_[floor, w])
        low, high = np.percentile(boot, [2.5, 97.5], axis=0)

        names = ["floor", "const"] + list(rows[0])
        coef = np.r_[model[0], model[1]]
        skip = 1 if phase == "decode" else 0  # decode has no floor
        n, p = len(y), len(coef) - skip
        print(f"\n{phase}: {n} batches, {p} parameters (ms per unit, bootstrap 95% interval)")
        for name, c, lo, hi in list(zip(names, coef, low, high))[skip:]:
            print(f"  {name:>10}  {c:10.4g}   [{lo:.4g}, {hi:.4g}]")

        order = list(range(n))
        random.Random(0).shuffle(order)
        pred = np.empty_like(y)
        for fold in range(FOLDS):
            test = order[fold::FOLDS]
            held_out = set(test)
            train = [i for i in order if i not in held_out]
            pred[test] = predict(fit(phase, X[train], y[train]), X[test])

        fitted = predict(model, X)
        adjusted = 1 - (1 - r2(y, fitted)) * (n - 1) / (n - p)
        spread = np.array([np.ptp(samples[key]) / np.median(samples[key]) for key in keys])
        print(f"  in-sample  R² {r2(y, fitted):.4f} (adjusted {adjusted:.4f}), {describe(y, fitted)}")
        print(f"  held-out   R² {r2(y, pred):.4f}, {describe(y, pred)}")
        print(
            f"  noise      a batch's samples spread by {100 * np.median(spread):.1f}% "
            f"(median), {100 * np.percentile(spread, 90):.1f}% (p90)"
        )

        sizes = np.array([step_size(phase, batch) for batch in batches])
        unit = "requests" if phase == "decode" else "tokens"
        edges = np.unique(np.percentile(sizes, [0, 25, 50, 75, 100]).round())
        bucket = np.clip(np.searchsorted(edges, sizes, side="right") - 1, 0, len(edges) - 2)
        for b in range(len(edges) - 1):
            sel = bucket == b
            if sel.any():
                label = f"{edges[b]:.0f}-{edges[b + 1]:.0f} {unit}"
                print(f"    {label:>22} ({sel.sum():>3}): held-out {relative(y[sel], pred[sel])}")


if __name__ == "__main__":
    main(sys.argv[1:])
