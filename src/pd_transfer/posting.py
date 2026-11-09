"""The paper's four regressions (linear, degree 2, degree 2 + sum l^2, backward-selected) for the posting span E of a
prefill step's transfers (first end to last end, steps of 2+ requests), on the migrating group's N, S, M, the prefill
step running after it (N_p, S_p, M_p; 0 if none) and the decode batch running when posting starts (N_d, S_d context
tokens, M_d; 0 if empty). Held out by set of prompt lengths. Writes each model's held-out E for every step.
Usage: python3 posting.py STEPS.pkl OUT.pkl"""
import importlib.util, itertools, os, pickle, random, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # src/
from fit import lstsq

out = sys.argv[2]
sys.argv = ["component.py", sys.argv[1]]
spec = importlib.util.spec_from_file_location("F", os.path.join(HERE, "component.py")); F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
rows = F.rows
MODELS = ["linear", "degree 2", "degree 2 + sum l^2"]


def variables(r):
    ctx = F.running_at(r, -r["E"])  # decode batch's context lengths when posting starts
    b = r["behind"]
    v = {"N": r["N"], "S": r["S"], "M": r["M"],
         "N_p": len(b), "S_p": sum(b), "M_p": max(b) if b else 0,
         "N_d": len(ctx), "S_d": sum(ctx), "M_d": max(ctx) if ctx else 0}
    return v, sum(l * l for l in r["lens"]), sum(l * l for l in b)


def terms(r, model):
    v, l2, l2p = variables(r)
    t = {"const": 1.0} | v
    if model != "linear":
        for a, b in itertools.combinations_with_replacement(list(v), 2):
            t[f"{a}^2" if a == b else f"{a}*{b}"] = v[a] * v[b]
        if model == MODELS[2]:
            t["sum l^2"], t["sum l_p^2"] = l2, l2p
    return t


keys = sorted({r["key"] for r in rows}); kidx = {k: i for i, k in enumerate(keys)}
order = list(range(len(keys))); random.Random(0).shuffle(order)
FOLD = {k: order.index(i) % 5 for i, k in enumerate(keys)}  # the same seeded 5-fold split over prompt-length sets
multi = [r for r in rows if r["N"] > 1 and np.diff(r["ends"]).max() * 1000 <= 50]  # no cut prompt
# the component posting model (component.py's E), held out the same way, as a baseline
F.REL = True
for f in range(5):
    tr = [r for r in rows if FOLD[r["key"]] != f and not (r["N"] > 1 and np.diff(r["ends"]).max() * 1000 > 50)]
    w = F.fit_E(tr)
    for r in rows:
        if FOLD[r["key"]] == f:
            r["E_comp"] = float(np.dot(w, F.E_feats(r)))
y = np.array([r["E"] for r in multi])
fold = np.array([FOLD[r["key"]] for r in multi])


def held_out(X, y, fold, sub=None):
    pred = np.empty_like(y)
    for f in range(5):
        tr, te = fold != f, fold == f
        pred[te] = X[te] @ lstsq(X[tr], y[tr])
    return pred


def rel(pred, y):
    return 100 * np.abs(pred - y) / y


def eliminate(names, X, y, fold):
    kept = names[1:]
    cols = lambda k: [names.index(n) for n in ["const"] + k]
    path = [(None, kept, rel(held_out(X, y, fold), y))]
    while kept:
        trials = [(t, [k for k in kept if k != t]) for t in kept]
        trials = [(t, rest, rel(held_out(X[:, cols(rest)], y, fold), y)) for t, rest in trials]
        path.append(min(trials, key=lambda x: x[2].mean()))
        kept = path[-1][1]
    best = min((e for _, _, e in path), key=np.mean)
    limit = best.mean() + best.std(ddof=1) / np.sqrt(len(best))
    return min((k for _, k, e in path if e.mean() <= limit), key=len)


results = {}
for model in MODELS:
    names = list(terms(multi[0], model))
    X = np.array([[terms(r, model)[n] for n in names] for r in multi])
    results[model] = (names, X, held_out(X, y, fold))
# backward-selected from degree 2 + sum l^2, the terms chosen within each training fold (nested)
names, X, _ = results[MODELS[2]]
pred = np.empty_like(y); chosen = []
for f in range(5):
    tr, te = fold != f, fold == f
    kept = eliminate(names, X[tr], y[tr], fold[tr])
    cols = [names.index(n) for n in ["const"] + kept]
    pred[te] = X[te][:, cols] @ lstsq(X[tr][:, cols], y[tr]); chosen.append(kept)
results["selected"] = (None, None, pred)
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
X_raw = np.array([list(variables(r)[0].values()) + list(variables(r)[1:]) for r in multi], dtype=float)
for name, make in (("random forest", lambda: RandomForestRegressor(random_state=0, n_jobs=-1)),
                   ("gradient boosting", lambda: HistGradientBoostingRegressor(random_state=0))):
    p = np.empty_like(y)
    for f in range(5):
        tr, te = fold != f, fold == f
        p[te] = make().fit(X_raw[tr], y[tr]).predict(X_raw[te])
    results[name] = (["-"] * X_raw.shape[1], None, p)
results["component model"] = (["-"]*6, None, np.array([r["E_comp"] for r in multi]))
kept_all = eliminate(names, X, y, fold)
cols = [names.index(n) for n in ["const"] + kept_all]
w_all = lstsq(X[:, cols], y)

print(f"posting span E, {len(multi)} steps of 2+ requests, held out by prompt-length set")
print("| model | terms | median abs err ms | p90 ms | within 1 ms | within 2 ms | mean rel err |")
print("|---|---|---|---|---|---|---|")
base = np.array([r["E_pred"] for r in multi]) if "E_pred" in multi[0] else None
for model, (n, Xm, p) in results.items():
    e = np.abs(p - y)
    print(f"| {model} | {len(n) if n else len(cols)} | {np.median(e):.2f} | {np.percentile(e, 90):.1f} | {100 * np.mean(e <= 1):.0f}% | {100 * np.mean(e <= 2):.0f}% | {rel(p, y).mean():.1f}% |")
print("selected terms (all data):", ", ".join(f"{n} {w:+.4g}" for n, w in zip(["const"] + kept_all, w_all)))
print("terms chosen per fold:", [len(k) for k in chosen])
for lo, hi in ((2, 9), (9, 33), (33, 10**9)):
    m = np.array([lo <= r["N"] < hi for r in multi])
    print(f"  N {lo}-{hi - 1 if hi < 10**9 else '+'}: " + "; ".join(f"{k} median {np.median(np.abs(p - y)[m]):.2f} bias {np.median((p - y)[m]):+.2f}" for k, (_, _, p) in results.items()))
pickle.dump({"rids": [(r["scenario"], r["index"], r["sample"], r["k"]) for r in multi],
             "pred": {k: v[2] for k, v in results.items()}}, open(out, "wb"))
