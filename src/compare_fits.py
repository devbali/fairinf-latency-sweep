"""Compare four regressions for step latency on runner.py results: linear, a full
degree-2 polynomial, the degree-2 polynomial plus the attention term sum l^2,
and the terms of that last model that matter most.

All four are polynomials in N, S and M: a step's requests, its total tokens
(prompt tokens for prefill, context tokens for decode) and its longest request.
A hybrid step has a _p and a _d version of each, for its prefill and decode
parts. Each batch's latency is the median of its samples. Prefill and hybrid
steps are fitted as max(floor, polynomial), because the CPU launching kernels
sets the time of small ones. Decode steps run as CUDA graphs, so their N is
padded to the graph's batch size, and their time bends at each power of two in
N. The models differ only in their polynomial terms. The first three are also
fitted alone, with no floor and no bends.

The fourth model comes from backward elimination. Starting from degree 2 +
sum l^2, it repeatedly drops the term whose removal leaves the lowest held-out
error, then keeps the smallest model on that path whose error is within one
standard error of the path's best (the one-standard-error rule). Repeating the
whole selection inside each training fold (nested cross-validation) checks
that choosing the terms didn't overfit.

Prints markdown: accuracy in-sample and on held-out batches (5-fold
cross-validation), the elimination path, and every coefficient in ms per unit,
with bootstrap 95% intervals for the selected model.

Usage: python src/compare_fits.py RESULTS.jsonl [MORE.jsonl ...]
"""

import itertools
import json
import sys
from collections import defaultdict

import numpy as np

from fit import bootstrap, errors, fit, folds, held_out, load, predict, r2

# Decode batches run as CUDA graphs, padded up to these sizes.
GRAPH_BS = [1, 2, 4, 8, 12, 16, 24] + list(range(32, 257, 8))
# Decode time is piecewise linear in the padded batch size, bending at each power of two.
DECODE_KNOTS = [2, 4, 8, 16, 32, 64, 128]
MODELS = ["linear", "degree 2", "degree 2 + sum l^2"]
SELECTED = "selected"


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
    if model in MODELS[1:]:
        # The same terms, in the same order, as scikit-learn's PolynomialFeatures(degree=2).
        for a, b in itertools.combinations_with_replacement(list(v), 2):
            t[f"{a}^2" if a == b else f"{a}*{b}"] = v[a] * v[b]
        if model == MODELS[2]:
            t["sum l_p^2" if phase == "hybrid" else "sum l^2"] = l2
    if bends:
        for knot in DECODE_KNOTS:
            t[f"(N-{knot})+"] = max(0, v["N"] - knot)
    return t


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


samples = load(sys.argv[1:])
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

    order = MODELS + [SELECTED]
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
    print(f"\nCoefficients in ms per unit.\n\n| term | {' | '.join(order)} |\n" + "|---" * (len(order) + 1) + "|")
    for term in (["floor F"] if floor else []) + names:
        print(f"| {term} | " + " | ".join(f"{coef[name][term]:.4g}" if term in coef[name] else ""
                                          for name in order) + " |")

    if phase != "decode":
        sm, l2 = ("S_p*M_p", "sum l_p^2") if phase == "hybrid" else ("S*M", "sum l^2")
        ratio = boot[:, 1 + selected_names.index(l2)] / boot[:, 1 + selected_names.index(sm)]
        print(f"\n{l2} / {sm}: {coef[SELECTED][l2] / coef[SELECTED][sm]:.3f} in the {SELECTED} model "
              f"(95% interval [{np.percentile(ratio, 2.5):.3f}, {np.percentile(ratio, 97.5):.3f}]), "
              f"{coef[MODELS[2]][l2] / coef[MODELS[2]][sm]:.3f} in {MODELS[2]}")

shown = MODELS + [SELECTED]
print("\n## Summary, held out, with the floor and bends\n\n| | " + " | ".join(shown) + " |\n"
      + "|---" * (len(shown) + 1) + "|")
for (phase, stat), values in summary.items():
    print(f"| {phase} {stat} | " + " | ".join(values[name] for name in shown) + " |")
print("\n## Summary, held out, the polynomial alone\n\n| | " + " | ".join(MODELS) + " |\n"
      + "|---" * (len(MODELS) + 1) + "|")
for (phase, stat), values in plain.items():
    print(f"| {phase} {stat} | " + " | ".join(values[name] for name in MODELS) + " |")
