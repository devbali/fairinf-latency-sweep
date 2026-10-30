"""Helpers the scripts share: loading runner.py results, least squares with a
floor, and the cross-validation and bootstrap the models are judged by.

A model is linear in its terms X. Prefill and hybrid steps are fitted with a
floor, T = max(floor, X @ w), because the CPU launching kernels sets the time
of small steps whatever their size. Decode steps run as CUDA graphs and are
fitted without one, T = X @ w.
"""

import json
import random
from collections import defaultdict

import numpy as np

FOLDS = 5
BOOTSTRAP = 200


def load(paths):
    """Every sample's latency in seconds, by phase and batch (as JSON)."""
    samples = defaultdict(list)
    for path in paths:
        for line in open(path):
            row = json.loads(line)
            samples[row["phase"], json.dumps(row["batch"])].append(row["latency_s"])
    return samples


def lstsq(X, y):
    scale = np.abs(X).max(axis=0)
    scale[scale == 0] = 1
    return np.linalg.lstsq(X / scale, y, rcond=None)[0] / scale


def fit(X, y, floor):
    """Least squares for T = max(floor, X @ w), or T = X @ w without a floor. Returns (floor, w).

    With a floor, fit w to the steps above the floor and the floor to the rest,
    then move each step to the side the model now puts it on, until none moves.
    """
    if not floor:
        return 0.0, lstsq(X, y)
    above = y > 1.2 * y.min()
    for _ in range(50):
        w = lstsq(X[above], y[above])
        f = y[~above].mean() if not above.all() else 0.0
        if ((X @ w > f) == above).all():
            break
        above = X @ w > f
    return f, w


def predict(model, X):
    floor, w = model
    return np.maximum(floor, X @ w)


def r2(y, pred):
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def errors(pred, y):
    """Relative error of each prediction, in percent."""
    return 100 * np.abs(pred - y) / y


def folds(n):
    """A seeded 5-fold split, as (training, held-out) index lists."""
    order = list(range(n))
    random.Random(0).shuffle(order)
    for fold in range(FOLDS):
        test = order[fold::FOLDS]
        held = set(test)
        yield [i for i in order if i not in held], test


def held_out(X, y, floor):
    """Each batch's prediction by a model fitted without its fold."""
    pred = np.empty_like(y)
    for train, test in folds(len(y)):
        pred[test] = predict(fit(X[train], y[train], floor), X[test])
    return pred


def bootstrap(X, y, floor):
    """Floor and coefficients refitted on resamples of the batches, one row per resample."""
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(BOOTSTRAP):
        i = rng.integers(0, len(y), len(y))
        f, w = fit(X[i], y[i], floor)
        boot.append(np.r_[f, w])
    return np.array(boot)
