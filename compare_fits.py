"""Compare four regressions for step latency on runner.py results: linear, a full
degree-2 polynomial, the degree-2 polynomial plus the attention term sum l^2,
and the terms of that last model that matter most.

All four are polynomials in N, S and M as in fit.py (hybrid: the _p and _d
versions of each). Prefill and hybrid are fitted as max(floor, polynomial) and
decode keeps fit.py's bends in its padded batch size, so the models differ only
in their polynomial terms. fit.py's own model is included for reference, and
the first three are also fitted alone, with no floor and no bends.

The fourth model comes from backward elimination. Starting from degree 2 +
sum l^2, it repeatedly drops the term whose removal leaves the lowest held-out
error, then keeps the smallest model on that path whose error is within one
standard error of the path's best (the one-standard-error rule). Repeating the
whole selection inside each training fold (nested cross-validation) checks
that choosing the terms didn't overfit.

Prints markdown: accuracy in-sample and on held-out batches (5-fold
cross-validation with fit.py's folds), the elimination path, and every
coefficient in ms per unit, with bootstrap 95% intervals for the selected model.

Usage: python compare_fits.py RESULTS.jsonl [MORE.jsonl ...]
"""

import itertools
import json
import random
import sys
from collections import defaultdict

import numpy as np

GRAPH_BS = [1, 2, 4, 8, 12, 16, 24] + list(range(32, 257, 8))
DECODE_KNOTS = [2, 4, 8, 16, 32, 64, 128]
FOLDS = 5
BOOTSTRAP = 200
MODELS = ["linear", "degree 2", "degree 2 + sum l^2"]
SELECTED = "selected"
REFERENCE = "fit.py's model"


def variables(phase, batch):
    """N, S and M of a step (decode N padded to its CUDA graph), and sum l^2 of its prompts."""
    if phase == "hybrid":
        p, d = batch
        v = {"N_p": len(p), "S_p": sum(p), "M_p": max(p), "N_d": len(d), "S_d": sum(d), "M_d": max(d)}
        return v, sum(length * length for length in p)
    v = {"N": len(batch), "S": sum(batch), "M": max(batch)}
    if phase == "decode":
        v["N"] = next((bs for bs in GRAPH_BS if bs >= v["N"]), v["N"])
    return v, sum(length * length for length in batch)


def terms(phase, batch, model, bends):
    v, l2 = variables(phase, batch)
    t = {"const": 1.0} | v
    l2_name = "sum l_p^2" if phase == "hybrid" else "sum l^2"
    if model == REFERENCE and phase == "prefill":
        t |= {"N*M": v["N"] * v["M"], "S*M": v["S"] * v["M"], l2_name: l2}
    elif model == REFERENCE and phase == "hybrid":
        t |= {"N_p*M_p": v["N_p"] * v["M_p"], "S_p*M_p": v["S_p"] * v["M_p"], l2_name: l2,
              "N_d*M_p": v["N_d"] * v["M_p"], "S_d*M_p": v["S_d"] * v["M_p"]}
    elif model in MODELS[1:]:
        # The same terms, in the same order, as scikit-learn's PolynomialFeatures(degree=2).
        for a, b in itertools.combinations_with_replacement(list(v), 2):
            t[f"{a}^2" if a == b else f"{a}*{b}"] = v[a] * v[b]
        if model == MODELS[2]:
            t[l2_name] = l2
    if bends:
        for knot in DECODE_KNOTS:
            t[f"(N-{knot})+"] = max(0, v["N"] - knot)
    return t


def lstsq(X, y):
    scale = np.abs(X).max(axis=0)
    scale[scale == 0] = 1
    return np.linalg.lstsq(X / scale, y, rcond=None)[0] / scale


def fit(X, y, floor):
    """Least squares for T = max(floor, X @ w), fitted as in fit.py. Returns (floor, w)."""
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
    """fit.py's 5-fold split, as (training, held-out) index lists."""
    order = list(range(n))
    random.Random(0).shuffle(order)
    for fold in range(FOLDS):
        test = order[fold::FOLDS]
        held = set(test)
        yield [i for i in order if i not in held], test


def held_out(X, y, floor):
    pred = np.empty_like(y)
    for train, test in folds(len(y)):
        pred[test] = predict(fit(X[train], y[train], floor), X[test])
    return pred


def columns(names, kept):
    return [names.index(name) for name in ["const"] + kept]


def eliminate(names, X, y, floor):
    """Backward elimination over every term but the constant.

    Returns the path as (dropped term, kept terms, held-out errors) steps, the
    kept terms of the smallest model whose mean error is within one standard
    error of the path's best, and that error limit.
    """
    kept = names[1:]
    path = [(None, kept, errors(held_out(X, y, floor), y))]
    while kept:
        trials = []
        for term in kept:
            rest = [k for k in kept if k != term]
            trials.append((term, rest, errors(held_out(X[:, columns(names, rest)], y, floor), y)))
        path.append(min(trials, key=lambda trial: trial[2].mean()))
        kept = path[-1][1]
    best = min((err for _, _, err in path), key=np.mean)
    limit = best.mean() + best.std(ddof=1) / np.sqrt(len(best))
    return path, min((kept for _, kept, err in path if err.mean() <= limit), key=len), limit


def nested(names, X, y, floor):
    """Held-out errors when the terms, too, are chosen without the held-out batches,
    and the terms each training fold chose."""
    pred = np.empty_like(y)
    chosen = []
    for train, test in folds(len(y)):
        _, kept, _ = eliminate(names, X[train], y[train], floor)
        cols = columns(names, kept)
        pred[test] = predict(fit(X[train][:, cols], y[train], floor), X[test][:, cols])
        chosen.append(kept)
    return errors(pred, y), chosen


def bootstrap(X, y, floor):
    """Floor and coefficients refitted on resamples of the batches, one row per resample."""
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(BOOTSTRAP):
        i = rng.integers(0, len(y), len(y))
        f, w = fit(X[i], y[i], floor)
        boot.append(np.r_[f, w])
    return np.array(boot)


samples = defaultdict(list)
for path in sys.argv[1:]:
    for line in open(path):
        row = json.loads(line)
        samples[row["phase"], json.dumps(row["batch"])].append(row["latency_s"])

header = "| model | params | R^2 | held-out R^2 | held-out RMSE | mean error | median | p90 | max |\n" + "|---" * 9 + "|"
summary, plain = defaultdict(dict), defaultdict(dict)  # (phase, statistic) -> {model: value}
for phase in ("prefill", "decode", "hybrid"):
    keys = sorted(key for key in samples if key[0] == phase)
    if not keys:
        continue
    batches = [json.loads(batch) for _, batch in keys]
    y = np.array([np.median(samples[key]) for key in keys]) * 1e3  # ms
    spread = np.array([np.ptp(samples[key]) / np.median(samples[key]) for key in keys])
    floor, bends = phase != "decode", phase == "decode"
    print(f"\n## {phase}: {len(y)} batches, samples spread by {100 * np.median(spread):.1f}% (median)")

    order = MODELS + [SELECTED] + ([REFERENCE] if phase != "decode" else [])  # fit.py's decode model is the linear one
    models = {}  # name -> (term names, X)
    for name in order:
        if name != SELECTED:
            rows = [terms(phase, batch, name, bends) for batch in batches]
            models[name] = (list(rows[0]), np.array([list(r.values()) for r in rows], dtype=float))
    names, X = models[MODELS[2]]
    path, kept, limit = eliminate(names, X, y, floor)
    models[SELECTED] = (["const"] + kept, X[:, columns(names, kept)])
    fitted = {name: fit(models[name][1], y, floor) for name in order}
    pred = {name: held_out(models[name][1], y, floor) for name in order}

    print(f"\nWith the {'floor' if floor else 'bends'}; errors are on held-out batches.\n\n{header}")
    for name in order:
        X_model = models[name][1]
        err = errors(pred[name], y)
        params = X_model.shape[1] + (1 if floor else 0)
        rmse = np.sqrt(((pred[name] - y) ** 2).mean())
        print(f"| {name} | {params} | {r2(y, predict(fitted[name], X_model)):.4f} | {r2(y, pred[name]):.4f} "
              f"| {rmse:.1f} ms | {err.mean():.1f}% | {np.median(err):.1f}% | {np.percentile(err, 90):.1f}% "
              f"| {err.max():.1f}% |")
        summary[phase, "parameters"][name] = f"{params}"
        summary[phase, "held-out R^2"][name] = f"{r2(y, pred[name]):.4f}"
        summary[phase, "mean error"][name] = f"{err.mean():.1f}%"
        summary[phase, "max error"][name] = f"{err.max():.1f}%"

    err, chosen = nested(names, X, y, floor)
    print(f"\nWith the terms also chosen inside each training fold (nested cross-validation), the "
          f"{SELECTED} model's held-out error is mean {err.mean():.1f}%, median {np.median(err):.1f}%, "
          f"p90 {np.percentile(err, 90):.1f}%, max {err.max():.1f}%.")
    for fold, fold_kept in enumerate(chosen):
        changes = [f"+{t}" for t in fold_kept if t not in kept] + [f"-{t}" for t in kept if t not in fold_kept]
        print(f"- fold {fold} chose {len(fold_kept)} terms: {', '.join(changes) or 'the same ones'}")

    if phase != "decode":
        # With one prompt, sum l^2 = S^2 = M^2; with equal prompts it is S*M.
        prompts = [batch if phase == "prefill" else batch[0] for batch in batches]
        kinds = np.array(["one prompt" if len(p) == 1 else "equal prompts" if len(set(p)) == 1
                          else "unequal prompts" for p in prompts])
        groups = ["one prompt", "equal prompts", "unequal prompts"]
        print(f"\nHeld-out error by the step's prompts: mean, max.\n\n| model | "
              + " | ".join(f"{k} ({(kinds == k).sum()})" for k in groups) + " |\n" + "|---" * 4 + "|")
        for name in order:
            err = errors(pred[name], y)
            print(f"| {name} | " + " | ".join(f"{err[kinds == k].mean():.1f}%, {err[kinds == k].max():.1f}%"
                                              for k in groups) + " |")

    print(f"\nThe polynomial alone, without the {'floor' if floor else 'bends'}.\n\n{header}")
    for name in MODELS:
        X_plain = np.array([list(terms(phase, batch, name, False).values()) for batch in batches], dtype=float)
        p = held_out(X_plain, y, False)
        err = errors(p, y)
        print(f"| {name} | {X_plain.shape[1]} | {r2(y, predict(fit(X_plain, y, False), X_plain)):.4f} "
              f"| {r2(y, p):.4f} | {np.sqrt(((p - y) ** 2).mean()):.1f} ms | {err.mean():.1f}% "
              f"| {np.median(err):.1f}% | {np.percentile(err, 90):.1f}% | {err.max():.1f}% |")
        plain[phase, "held-out R^2"][name] = f"{r2(y, p):.4f}"
        plain[phase, "mean error"][name] = f"{err.mean():.1f}%"
        plain[phase, "max error"][name] = f"{err.max():.1f}%"

    best = min(err.mean() for _, _, err in path)
    print(f"\nBackward elimination from {MODELS[2]}: held-out error after each drop. The {SELECTED} model "
          f"is the smallest with mean error at most {limit:.2f}%: the best, {best:.2f}%, plus one standard "
          f"error.\n\n| params | dropped | mean error | max |\n" + "|---" * 4 + "|")
    for term, step_kept, err in path:
        params = len(step_kept) + 1 + (1 if floor else 0)
        print(f"| {params}{' (selected)' if step_kept == kept else ''} | {term or 'none'} | {err.mean():.2f}% "
              f"| {err.max():.1f}% |")

    selected_names, X_selected = models[SELECTED]
    boot = bootstrap(X_selected, y, floor)
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    f, w = fitted[SELECTED]
    print(f"\nThe {SELECTED} model, in ms per unit.\n\n"
          "| term | coefficient | 95% interval | held-out mean error without it |\n" + "|---" * 4 + "|")
    if floor:
        print(f"| floor F | {f:.4g} | [{low[0]:.3g}, {high[0]:.3g}] | |")
    for i, term in enumerate(selected_names):
        without = "" if term == "const" else \
            f"{errors(held_out(np.delete(X_selected, i, axis=1), y, floor), y).mean():.2f}%"
        print(f"| {term} | {w[i]:.4g} | [{low[1 + i]:.3g}, {high[1 + i]:.3g}] | {without} |")

    coef = {name: ({"floor F": fitted[name][0]} if floor else {}) | dict(zip(models[name][0], fitted[name][1]))
            for name in order}
    shown = MODELS + [SELECTED]
    print(f"\nCoefficients in ms per unit.\n\n| term | {' | '.join(shown)} |\n" + "|---" * (len(shown) + 1) + "|")
    for term in (["floor F"] if floor else []) + names:
        print(f"| {term} | " + " | ".join(f"{coef[name][term]:.4g}" if term in coef[name] else ""
                                          for name in shown) + " |")

    if phase != "decode":
        sm, l2 = ("S_p*M_p", "sum l_p^2") if phase == "hybrid" else ("S*M", "sum l^2")
        ratio = boot[:, 1 + selected_names.index(l2)] / boot[:, 1 + selected_names.index(sm)]
        print(f"\n{l2} / {sm}: {coef[SELECTED][l2] / coef[SELECTED][sm]:.3f} in the {SELECTED} model "
              f"(95% interval [{np.percentile(ratio, 2.5):.3f}, {np.percentile(ratio, 97.5):.3f}]), "
              f"{coef[MODELS[2]][l2] / coef[MODELS[2]][sm]:.3f} in {MODELS[2]}, "
              f"{coef[REFERENCE][l2] / coef[REFERENCE][sm]:.3f} in {REFERENCE}")

shown = MODELS + [SELECTED, REFERENCE]
print("\n## Summary, held out, with the floor and bends\n\n| | " + " | ".join(shown) + " |\n"
      + "|---" * (len(shown) + 1) + "|")
for (phase, stat), values in summary.items():
    print(f"| {phase} {stat} | " + " | ".join(values.get(name, values[MODELS[0]]) for name in shown) + " |")
print("\n## Summary, held out, the polynomial alone\n\n| | " + " | ".join(MODELS) + " |\n"
      + "|---" * (len(MODELS) + 1) + "|")
for (phase, stat), values in plain.items():
    print(f"| {phase} {stat} | " + " | ".join(values[name] for name in MODELS) + " |")
