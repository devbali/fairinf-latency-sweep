"""The two functions every latency estimator shares, pooled() and isolated(), and the types they work on.

A latency model is linear in its terms, T = max(F, sum_j w_j * term_j), where F is a floor (none for models
without one). Each module in this directory (prefill.py, decode.py, hybrid.py, pd_kv_transfer.py, prefill_tp4.py,
decode_tp4.py) defines one Model: its terms, the degree of each term, how to compute the terms from an input,
and the samples it is fitted on. Every model goes through the same two functions:

    pooled(model, percentile=None, refit=False) -> Estimator
        Fits the model on all its samples. With a percentile p, returns an estimator that p% of the samples
        fall below: by default the central fit with its intercept (and floor) shifted up by the p-th
        percentile of the residuals; with refit=True, a quantile regression at p refitted from scratch.
    isolated(estimator, n) -> Estimator
        The hypothetical latency with 1/n of the resources: the constant stays the same, every linear term's
        coefficient is multiplied by n and every quadratic term's by n^2.

An Estimator is called on one input and returns milliseconds:

    from estimators import prefill
    from estimators.estimator import pooled, isolated
    p90 = pooled(prefill.MODEL, 90)       # 90% of the measured steps take at most this
    p90([512, 1024])                      # ms for a prefill step of two prompts
    isolated(p90, 4)([512, 1024])         # the same step with a quarter of the GPU
"""

import os
import sys
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fit import fit as fit_floor  # noqa: E402  least squares with the CPU floor, as compare_fits.py fits

NO_FLOOR = -np.inf


@dataclass(frozen=True)
class Model:
    """A latency model before fitting.

    terms:    term names; the first is "const".
    degrees:  each term's degree in the input: 0 for the constant, 1 for linear, 2 for quadratic.
    features: one input -> its term values, in the order of terms.
    data:     () -> (X, y, group): one row of term values and one latency in ms per sample, and which batch each
              sample measured. Samples of one batch share their terms.
    floor:    fit T = max(F, X @ w) instead of T = X @ w.
    loss:     "squares" fits least squares on each batch's median sample (the latency sweep's models);
              "huber" fits a robust Huber regression on every sample (the PD transfer model).
    """
    name: str
    terms: tuple
    degrees: tuple
    features: Callable
    data: Callable
    floor: bool = False
    loss: str = "squares"


@dataclass(frozen=True)
class Estimator:
    """A fitted model: T = max(floor, terms @ coef) + add, in ms. add is 0 for a fitted model; an isolated
    decode model adds the static gap (isolated())."""
    model: Model
    coef: np.ndarray
    floor: float = NO_FLOOR
    label: str = "central"
    add: float = 0.0

    def __call__(self, x):
        return max(self.floor, float(np.dot(self.model.features(x), self.coef))) + self.add

    def predict(self, X):
        """Latencies of many rows of term values at once."""
        return np.maximum(self.floor, X @ self.coef) + self.add

    def formula(self):
        parts = [f"{self.coef[0]:.6g}"] + [f"{w:+.6g} {t}" for t, w in zip(self.model.terms[1:], self.coef[1:])]
        body = " ".join(parts)
        t = f"max({self.floor:.6g}, {body})" if self.floor > NO_FLOOR else body
        return f"T = {t} + {self.add:.6g}" if self.add else f"T = {t}"


def _medians(X, y, group):
    """One row per batch: its terms and its median sample."""
    keys, first, inverse = np.unique(group, return_index=True, return_inverse=True)
    med = np.array([np.median(y[inverse == k]) for k in range(len(keys))])
    return X[first], med


def _huber(X, y):
    from sklearn.linear_model import HuberRegressor
    A = X[:, 1:]
    mu, sd = A.mean(0), A.std(0) + 1e-12
    m = HuberRegressor(max_iter=5000, alpha=1e-6).fit((A - mu) / sd, y)
    w = m.coef_ / sd
    return np.r_[m.intercept_ - (w * mu).sum(), w]


def _quantile(X, y, q):
    """Linear quantile regression (pinball loss) at quantile q, exact, by linear programming."""
    from sklearn.linear_model import QuantileRegressor
    scale = np.abs(X).max(axis=0)
    scale[scale == 0] = 1
    m = QuantileRegressor(quantile=q, alpha=0, fit_intercept=False, solver="highs").fit(X / scale, y)
    return m.coef_ / scale


def _central(model, X, y, group):
    if model.loss == "huber":
        return Estimator(model, _huber(X, y))
    Xb, yb = _medians(X, y, group)
    f, w = fit_floor(Xb, yb, model.floor)
    return Estimator(model, w, f if model.floor else NO_FLOOR)


def _refit(model, X, y, q, start):
    """Quantile regression at q on every sample. With a floor, alternate as fit.fit does: the coefficients on the
    samples above the floor, the floor as the q-th quantile of the rest, until no sample changes side."""
    if not model.floor:
        return Estimator(model, _quantile(X, y, q))
    above = X @ start.coef > start.floor
    for _ in range(50):
        w = _quantile(X[above], y[above], q)
        f = float(np.quantile(y[~above], q)) if (~above).any() else 0.0
        now = X @ w > f
        if (now == above).all():
            break
        above = now
    return Estimator(model, w, f)


def pooled(model, percentile=None, refit=False):
    """The model fitted on all its samples. With a percentile p (0-100), the estimator that p% of the samples fall
    below: the central fit shifted by the p-th percentile of its residuals, or with refit=True, a quantile
    regression at p."""
    X, y, group = model.data()
    central = _central(model, X, y, group)
    if percentile is None:
        return central
    if refit:
        return replace(_refit(model, X, y, percentile / 100, central), label=f"p{percentile:g} refit")
    shift = float(np.percentile(y - central.predict(X), percentile))
    coef = central.coef.copy()
    coef[0] += shift
    return replace(central, coef=coef, floor=central.floor + shift, label=f"p{percentile:g} shifted")


def isolated(estimator, n, floor=None, add=0.0):
    """The estimator with 1/n of the resources: the constant unchanged, linear terms x n, quadratic terms
    x n^2. floor, if given, raises the estimator's floor to it (an isolated step is never faster than the
    slowest pooled one, D^MT <= D^ISO); add is a time added to every step after the floor (the static gap)."""
    scale = float(n) ** np.array(estimator.model.degrees, dtype=float)
    lowest = estimator.floor if floor is None else max(estimator.floor, floor)
    return replace(estimator, coef=estimator.coef * scale, floor=lowest, add=add,
                   label=f"{estimator.label}, isolated n={n:g}")


def report(model, percentiles=(50, 90, 99)):
    """Print a model's central fit and how many samples fall below each percentile estimator, both ways."""
    X, y, _ = model.data()
    central = pooled(model)
    print(f"{model.name}: {len(y)} samples\ncentral: {central.formula()}")
    for p in percentiles:
        for refit in (False, True):
            est = pooled(model, p, refit)
            below = 100 * np.mean(y <= est.predict(X) + 1e-9)
            print(f"{est.label:12s} {below:5.1f}% of samples below   {est.formula()}")
