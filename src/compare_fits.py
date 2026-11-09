"""Compare four regressions for step latency on runner.py results, and for the
KV transfer time of PD disaggregation on pd_runner.py results: linear, a full
degree-2 polynomial, the degree-2 polynomial plus the attention term sum l^2,
and the terms of that last model that matter most.

All four are polynomials in N, S and M: a step's requests, its total tokens
(prompt tokens for prefill and transfer, context tokens for decode) and its
longest request. A hybrid step has a _p and a _d version of each, for its
prefill and decode parts. Each batch's latency is the median of its samples.
Prefill and hybrid steps are fitted as max(floor, polynomial), because the CPU
launching kernels sets the time of small ones. Decode steps run as CUDA graphs,
so their N is padded to the graph's batch size, and their time bends at each
power of two in N. The models differ only in their polynomial terms. The first
three are also fitted alone, with no floor and no bends.

A transfer "batch" is one prefill step of a PD run: its time runs from the
step's first result to the last of its requests' KV caches reaching the decode
server (see pd_fits.py), and every step of the same prompt lengths counts as a
sample of one batch. It is fitted as the polynomial alone.

The fourth model comes from backward elimination. Starting from degree 2 +
sum l^2, it repeatedly drops the term whose removal leaves the lowest held-out
error, then keeps the smallest model on that path whose error is within one
standard error of the path's best (the one-standard-error rule). Repeating the
whole selection inside each training fold (nested cross-validation) checks
that choosing the terms didn't overfit.

Two scikit-learn models are compared alongside: a random forest and gradient
boosted trees (HistGradientBoostingRegressor), each on the step's variables
(N, S, M and sum l^2, both parts' for a hybrid step), with scikit-learn's
default settings and no floor or bends, on the same held-out folds. They have
no coefficients; trees also cannot predict beyond the times they were fitted on.

Prints markdown: accuracy in-sample and on held-out batches (5-fold
cross-validation), the elimination path, and every coefficient in ms per unit,
with bootstrap 95% intervals for the selected model.

Usage: python src/compare_fits.py RESULTS.jsonl [MORE.jsonl ...] [--pd-phase NAME]

--pd-phase picks which phase of PD results to fit: pd_runner.py's "batches"
(the default), or one scenario of a pd_load.py run, e.g. "idle".
"""

import argparse
import itertools
import json
from collections import defaultdict

import numpy as np

from fit import bootstrap, errors, fit, folds, held_out, load, predict, r2
from pd_fits import is_pd, transfer_samples

# Decode batches run as CUDA graphs, padded up to these sizes.
GRAPH_BS = [1, 2, 4, 8, 12, 16, 24] + list(range(32, 257, 8))
# Decode time is piecewise linear in the padded batch size, bending at each power of two.
DECODE_KNOTS = [2, 4, 8, 16, 32, 64, 128]
MODELS = ["linear", "degree 2", "degree 2 + sum l^2"]
# A hybrid step with an empty part is the step the scheduler runs instead (see runner.py): each such piece is
# fitted as that kind of step, so a hybrid step's time is piecewise in which parts it has.
EDGES = {"hybrid, no decode": "prefill", "hybrid, no prefill": "decode"}
SELECTED = "selected"
TREES = ["random forest", "gradient boosting"]


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


def tree_model(name):
    """A scikit-learn model with its default settings, seeded."""
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    if name == TREES[0]:
        return RandomForestRegressor(random_state=0, n_jobs=-1)
    return HistGradientBoostingRegressor(random_state=0)


def tree_held_out(name, X, y):
    """Each batch's prediction by a tree model fitted without its fold."""
    pred = np.empty_like(y)
    for train, test in folds(len(y)):
        pred[test] = tree_model(name).fit(X[train], y[train]).predict(X[test])
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


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("results", nargs="+", help="runner.py or pd_runner.py results")
parser.add_argument("--pd-phase", default="batches", help="the phase of PD results to fit")
args = parser.parse_args()
samples = load([path for path in args.results if not is_pd(path)])
for path in filter(is_pd, args.results):
    for key, values in transfer_samples(path, args.pd_phase).items():
        samples[key] += values
for key in [k for k in samples if k[0] == "hybrid"]:
    p, d = json.loads(key[1])
    if not d or not p:
        samples["hybrid, no decode" if not d else "hybrid, no prefill", json.dumps(p or d)] += samples.pop(key)
header = "| model | params | R^2 | held-out R^2 | held-out RMSE | mean error | median | p90 | max |\n" + "|---" * 9 + "|"
summary, plain = defaultdict(dict), defaultdict(dict)  # (phase, statistic) -> {model: value}
for phase in ("prefill", "decode", "hybrid", *EDGES, "transfer"):
    kind = EDGES.get(phase, phase)
    keys = sorted(key for key in samples if key[0] == phase)
    if not keys:
        continue
    batches = [json.loads(batch) for _, batch in keys]
    y = np.array([np.median(samples[key]) for key in keys]) * 1e3  # ms
    # A transfer batch can have one sample: every step of the same prompt lengths is one
    spread = np.array([np.ptp(samples[key]) / np.median(samples[key]) for key in keys if len(samples[key]) > 1])
    floor, bends = kind in ("prefill", "hybrid"), kind == "decode"
    wrapper = "the floor" if floor else "the bends" if bends else "neither floor nor bends"
    print(f"\n## {phase}: {len(y)} batches, samples spread by {100 * np.median(spread):.1f}% (median)")

    order = MODELS + [SELECTED]
    models = {}  # name -> (term names, X)
    for name in order:
        if name != SELECTED:
            rows = [terms(kind, batch, name, bends) for batch in batches]
            models[name] = (list(rows[0]), np.array([list(r.values()) for r in rows], dtype=float))
    names, X = models[MODELS[2]]
    path, kept, limit = eliminate(names, X, y, floor)
    models[SELECTED] = (["const"] + kept, X[:, columns(names, kept)])
    fitted = {name: fit(models[name][1], y, floor) for name in order}
    pred = {name: held_out(models[name][1], y, floor) for name in order}

    print(f"\nWith {wrapper}; errors are on held-out batches.\n\n{header}")
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
    X_raw = np.array([list(variables(kind, batch)[0].values()) + [variables(kind, batch)[1]] for batch in batches],
                     dtype=float)
    for name in TREES:
        pred[name] = tree_held_out(name, X_raw, y)
        err = errors(pred[name], y)
        rmse = np.sqrt(((pred[name] - y) ** 2).mean())
        in_sample = tree_model(name).fit(X_raw, y).predict(X_raw)
        print(f"| {name} | - | {r2(y, in_sample):.4f} | {r2(y, pred[name]):.4f} | {rmse:.1f} ms | {err.mean():.1f}% "
              f"| {np.median(err):.1f}% | {np.percentile(err, 90):.1f}% | {err.max():.1f}% |")
        summary[phase, "parameters"][name] = "-"
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

    if kind != "decode":
        # With one prompt, sum l^2 = S^2 = M^2; with equal prompts it is S*M.
        prompts = [batch[0] if kind == "hybrid" else batch for batch in batches]
        kinds = np.array(["one prompt" if len(p) == 1 else "equal prompts" if len(set(p)) == 1
                          else "unequal prompts" for p in prompts])
        groups = ["one prompt", "equal prompts", "unequal prompts"]
        print(f"\nHeld-out error by the step's prompts: mean, max.\n\n| model | "
              + " | ".join(f"{k} ({(kinds == k).sum()})" for k in groups) + " |\n" + "|---" * 4 + "|")
        for name in order + TREES:
            err = errors(pred[name], y)
            print(f"| {name} | " + " | ".join(f"{err[kinds == k].mean():.1f}%, {err[kinds == k].max():.1f}%"
                                              if (kinds == k).any() else "-" for k in groups) + " |")

    if floor or bends:
        print(f"\nThe polynomial alone, without {wrapper}.\n\n{header}")
        for name in MODELS:
            X_plain = np.array([list(terms(kind, batch, name, False).values()) for batch in batches], dtype=float)
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

    sm, l2 = ("S_p*M_p", "sum l_p^2") if kind == "hybrid" else ("S*M", "sum l^2")
    if floor and sm in selected_names and l2 in selected_names:
        ratio = boot[:, 1 + selected_names.index(l2)] / boot[:, 1 + selected_names.index(sm)]
        print(f"\n{l2} / {sm}: {coef[SELECTED][l2] / coef[SELECTED][sm]:.3f} in the {SELECTED} model "
              f"(95% interval [{np.percentile(ratio, 2.5):.3f}, {np.percentile(ratio, 97.5):.3f}]), "
              f"{coef[MODELS[2]][l2] / coef[MODELS[2]][sm]:.3f} in {MODELS[2]}")

edge = {k: np.median(v) * 1e3 for k, v in samples.items() if k[0] in EDGES}
full = {tuple(map(json.dumps, json.loads(k[1]))): np.median(v) * 1e3 for k, v in samples.items() if k[0] == "hybrid"}
if edge and full:
    print("\n## Hybrid at its edges: what the first request of the other part adds (ms, medians)\n")
    for phase, other in (("hybrid, no prefill", 0), ("hybrid, no decode", 1)):
        rows = []
        for (_, part), t in sorted(edge.items()):
            if _ != phase:
                continue
            pairs = [(json.loads(k[other]), v) for k, v in full.items() if k[1 - other] == part]
            if pairs:
                smallest = min(pairs, key=lambda kv: (len(kv[0]), sum(kv[0])))
                rows.append((json.loads(part), t, smallest[0], smallest[1]))
        label = "decode part, alone as a CUDA graph" if other == 0 else "prefill part, alone"
        print(f"| {label} | alone | with the smallest other part | that part | added |\n|---|---|---|---|---|")
        for part, t, small, t2 in rows:
            show = lambda b: f"{len(b)} x {b[0]}" if len(set(b)) == 1 else f"{len(b)} requests, {sum(b)} tokens"
            print(f"| {show(part)} | {t:.2f} | {t2:.2f} | {show(small)} | {t2 - t:+.2f} |")
        print()

shown = MODELS + [SELECTED] + TREES
print("\n## Summary, held out, with the floor and bends (transfer has neither; nor have the trees)\n\n| | " + " | ".join(shown) + " |\n"
      + "|---" * (len(shown) + 1) + "|")
for (phase, stat), values in summary.items():
    print(f"| {phase} {stat} | " + " | ".join(values[name] for name in shown) + " |")
if plain:
    print("\n## Summary, held out, the polynomial alone\n\n| | " + " | ".join(MODELS) + " |\n"
          + "|---" * (len(MODELS) + 1) + "|")
    for (phase, stat), values in plain.items():
        print(f"| {phase} {stat} | " + " | ".join(values[name] for name in MODELS) + " |")
