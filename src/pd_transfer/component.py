"""The transfer time T = E + D of a PD run with millisecond logs, from its parts: one model for every load.

E, the posting span: a cost per request after the first and per thousand tokens they carry, more per request
while a prefill step runs behind, and the wait for the rest of a cut prompt (steps.py's replay).
D, the tail: the last KV lands c0 ms after its post plus its MB over bw GB/s. If the decode server is running a
step then (a request in its batch), it is found c1 ms after that step ends (its log line); else c_idle ms later.
Without the decode server's timing, D is c_idle plus a share of the predicted decode step when it is running.

Usage: python component.py STEPS.pkl"""
import os, pickle, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))  # src/
sys.path.insert(0, HERE)
from fit import lstsq, folds
from stepmodels import step_ms

SLOW_POST_MS = 3.5
rows = [r for r in pickle.load(open(sys.argv[1], "rb"))["rows"] if not r["scenario"].endswith("-torch")]  # the profiled pass is apart
models = pickle.load(open("stepmodels.pkl", "rb"))  # written by stepmodels.py in the working directory
pm, dm = models["prefill"], models["decode"]
SCEN = list(dict.fromkeys(r["scenario"] for r in rows))

def post_ms(P):
    """ms per post while a prefill step of P ms runs behind (fitted on the sweep's first scenarios)."""
    return 1.6 + 0.0126 * P


for r in rows:
    r["P_behind"] = step_ms(pm, r["behind"])
    n = min(r["N"] - 1, r["P_behind"] / post_ms(r["P_behind"])) if r["behind_running"] else 0.0
    r["slow_posts"], r["slow_posts_P"] = n, n * r["P_behind"] / 100
    r["chunk_wait"] = r["own_cut"] * max(0.0, r["P_behind"] - (r["N"] - 2) * post_ms(r["P_behind"]))
    r["posted_k"] = sum(r["posted_lens"][:-1]) / 1000  # tokens posted before the last request, thousands
    r["steps_rel"] = 1000 * (np.array(r["decode_steps"]) - r["last_end"])  # decode steps' log lines, ms after the last post
    r["mb_last"] = r["mbs"][-1] or 0.0
    r["spans_rel"] = [(1000 * (j0 - r["last_end"]), 1000 * (j1 - r["last_end"]), own, l) for j0, j1, own, l in r["decode_spans"]]


def running_at(r, L):
    """The context lengths of the decode server's batch L ms after the last post (one token per logged step since join)."""
    st = r["steps_rel"]
    return [l + 1 + int(((st > j0) & (st < L)).sum()) for j0, j1, own, l in r["spans_rel"] if j0 <= L < j1]


for r in rows:
    ctx = running_at(r, 3.0)
    r["busy3"] = len(ctx) > 0
    r["D_step"] = step_ms(dm, ctx) if ctx else 0.0
    r["busy_post"] = len(running_at(r, -r["E"] / 2)) > 0 if r["N"] > 1 else False  # the decode server ran during posting
    # the slow-copy risk: the copy may wait for the prefill step behind, for what is left of it after the last post
    r["left_behind"] = max(0.0, r["P_behind"] - r["E"]) if r["behind_running"] else 0.0
    mb = r["mbs"][-1] or 0.0
    r["mb_bin"] = int(np.searchsorted([5, 25, 100], mb, side="right"))


def E_feats(r):
    return [r["N"] - 1, r["posted_k"], r["slow_posts"], r["slow_posts_P"], (r["N"] - 1) * r["busy_post"], r["chunk_wait"]]


REL = False  # fit to relative error of T (weights 1/T) instead of squared ms


def wls(X, y, T):
    w = 1 / np.maximum(T, 1.0) if REL else np.ones(len(y))
    return lstsq(X * w[:, None], y * w)


def fit_E(train):
    return wls(np.array([E_feats(r) for r in train]), np.array([r["E"] for r in train]), np.array([r["T"] for r in train]))


OTHERS_ONLY = False  # judge whether the decode server is running a step by other requests only


def D_parts(r, L):
    """(the next decode step's log line after L, or None if the decode server is not running a step at L)."""
    if not any(j0 <= L < j1 for j0, j1, own, l in r["spans_rel"] if not (OTHERS_ONLY and own)):
        return None
    st = r["steps_rel"]
    i = int(np.searchsorted(st, L))
    return st[i] if i < len(st) and st[i] - L < 100 else None


N_GROUPS = (1, 2, 3, 9, 33)  # the step's requests: 1, 2, 3-8, 9-32, 33 or more


def ngroup(r):
    return int(np.searchsorted(N_GROUPS, r["N"], side="right")) - 1


def fit_D_timeline(train):
    """Grid over the landing delay c0 and bandwidth bw; then, by the step's requests, the median delay c1 after the
    decode step the KV lands in, and ci when none is running."""
    y = np.array([r["D"] for r in train])
    g = np.array([ngroup(r) for r in train])
    best = None
    for bw in (0, 25, 50, 100, 200):
        for c0 in np.arange(0, 10.1, 0.5):
            L = np.array([c0 + (r["mb_last"] / bw if bw else 0.0) for r in train])
            nxt = [D_parts(r, l) for r, l in zip(train, L)]
            busy = np.array([n is not None for n in nxt])
            base = np.array([n if n is not None else l for n, l in zip(nxt, L)])
            res = y - base
            c1 = {k: float(np.median(res[busy & (g == k)])) for k in set(g[busy])}
            ci = {k: float(np.median(res[~busy & (g == k)])) for k in set(g[~busy])}
            pred = base + np.array([c1.get(k, 2.0) if bz else ci.get(k, 5.0) for k, bz in zip(g, busy)])
            err = np.mean(np.abs(pred - y) / np.maximum(y, 1))
            if best is None or err < best[0]:
                best = (err, (c0, bw, c1, ci))
    return best[1]


def D_timeline(r, E_pred, c):
    c0, bw, c1, ci = c
    shift = E_pred - r["E"]  # the predicted last post, relative to the measured one
    L = shift + c0 + (r["mb_last"] / bw if bw else 0.0)
    n = D_parts(r, L)
    k = ngroup(r)
    return (n + c1.get(k, 2.0) if n is not None else L + ci.get(k, 5.0)) - shift


def steady(r, L):
    """Whether the decode server has been running steps for the 200 ms before L: a load of its own, not a probe's."""
    st = r["steps_rel"]
    return int(((st > L - 200) & (st <= L)).sum()) >= 6


def D_mean_feats(r):
    return [1.0, float(r["behind_running"]), r["D_step"] * r["busy3"], np.log(r["N"])]


def fit_D_mean(train):
    return wls(np.array([D_mean_feats(r) for r in train]), np.array([r["D"] for r in train]), np.array([r["T"] for r in train]))


def slow_rates(train):
    """The share of slow tails (D > 50 ms) among steps with a step behind, by last KV size and decode activity."""
    n, k = defaultdict(int), defaultdict(int)
    for r in train:
        if r["behind_running"]:
            key = (r["mb_bin"], r["busy3"])
            n[key] += 1
            k[key] += r["D"] > 50
    return {key: k[key] / n[key] for key in n}


def scores(sub, pred):
    out = {}
    for sc in SCEN:
        g = defaultdict(list)
        for r, p in zip(sub, pred):
            if r["scenario"] == sc:
                g[r["key"]].append((r["T"], p))
        if g:
            y = np.array([np.median([t for t, _ in v]) for v in g.values()])
            p = np.array([np.median([q for _, q in v]) for v in g.values()])
            out[sc] = (np.mean(100 * np.abs(p - y) / y), np.sqrt(np.mean((p - y) ** 2)))
    return out


VARIANTS = ("parts, mean decode wait", "parts, decode timing if steady or one prompt",
            "parts, mean decode wait, relative", "parts, decode timing if steady or one prompt, relative",
            "measured E, decode timing", "parts E, measured D, relative")
N_BINS = (1, 2, 9, 33, 10**9)  # E's residuals by the step's requests: 1, 2-8, 9-32, 33 or more
QUANTILES = np.linspace(0.05, 0.95, 10)


def slow_rates(train):
    """The share of slow tails (D > 50 ms) among steps with a step behind, by last KV size and decode activity."""
    n, k = defaultdict(int), defaultdict(int)
    for r in train:
        if r["behind_running"]:
            key = (r["mb_bin"], r["busy3"])
            n[key] += 1
            k[key] += r["D"] > 50
    return {key: k[key] / n[key] for key in n}


def predict(variant, train, test):
    global OTHERS_ONLY, REL
    OTHERS_ONLY = "others" in variant
    REL = "relative" in variant
    if variant == "batch shape only":
        X = np.array([[1.0, r["N"], r["S"], r["M"]] for r in train])
        w = lstsq(X, np.array([r["T"] for r in train]))
        return [float(np.dot(w, [1.0, r["N"], r["S"], r["M"]])) for r in test]
    wE = fit_E(train)
    normal = [r for r in train if r["D"] <= 50]  # slow copies are their own term
    rates = slow_rates(train)
    slow = lambda r: rates.get((r["mb_bin"], r["busy3"]), 0.0) * r["left_behind"] if ("slow" in variant and r["behind_running"]) else 0.0
    cm = fit_D_mean(normal) if ("mean" in variant or "steady" in variant) else None
    ct = fit_D_timeline(normal[::3]) if "timing" in variant else None
    out = []
    for r in test:
        E = r["E"] if variant.startswith("measured E") else float(np.dot(wE, E_feats(r)))
        if "measured D" in variant:
            D = r["D"]
        elif cm is not None and not ("steady" in variant and (r["N"] == 1 or steady(r, E - r["E"]))):
            D = float(np.dot(cm, D_mean_feats(r))) + slow(r)
        else:
            D = D_timeline(r, E, ct) + slow(r)
        out.append(E + D)
    return out


def evaluate(variant, held_out_scenario=None):
    pred = np.empty(len(rows))
    if held_out_scenario:
        splits = [([i for i, r in enumerate(rows) if r["scenario"] != held_out_scenario],
                   [i for i, r in enumerate(rows) if r["scenario"] == held_out_scenario])]
    else:
        keys = sorted({r["key"] for r in rows}); idx = {k: i for i, k in enumerate(keys)}
        splits = [([i for i, r in enumerate(rows) if idx[r["key"]] in set(tr)], [i for i, r in enumerate(rows) if idx[r["key"]] in set(te)])
                  for tr, te in folds(len(keys))]
    for tr, te in splits:
        pred[te] = predict(variant, [rows[i] for i in tr], [rows[i] for i in te])
    return scores(rows, pred)


if __name__ == "__main__":
    print(f"{len(rows)} probe steps in {len(SCEN)} scenarios; {sum(r['own_cut'] for r in rows)} with their own prompt cut")
    wE = fit_E(rows)
    print("E = " + " + ".join(f"{c:.3g} x {n}" for c, n in zip(wE, ["(N-1)", "thousand tokens posted", "posts behind a step",
          "posts behind x its 100 ms", "posts while decoding", "chunk wait"])))
    c = fit_D_timeline([r for r in rows if r["D"] <= 50][::2])
    print(f"D: the last KV lands {c[0]:.1f} ms after its post" + (f" + MB / {c[1]} GB/s" if c[1] else "")
          + "; found, by the step's requests " + str(N_GROUPS) + ": after the decode step it lands in "
          + str({k: round(v, 1) for k, v in sorted(c[2].items())}) + ", if none is running " + str({k: round(v, 1) for k, v in sorted(c[3].items())}))
    print("D without decode timing: %.2f + %.2f with a step behind + %.3f x predicted decode step + %.2f x ln N" % tuple(fit_D_mean([r for r in rows if r["D"] <= 50])))
    yD = np.array([r["D"] for r in rows]); pD = np.array([D_timeline(r, r["E"], c) for r in rows])
    print(f"D with the decode timing, in-sample: |error| median {np.median(np.abs(pD - yD)):.2f} ms, p90 {np.percentile(np.abs(pD - yD), 90):.2f}")
    print("\nHeld-out mean error (RMSE ms), sets of prompt lengths held out in every scenario at once\n")
    print("| variant | " + " | ".join(SCEN) + " |\n|---|" + "---|" * len(SCEN))
    for v in VARIANTS:
        s = evaluate(v)
        print(f"| {v} | " + " | ".join(f"{s[sc][0]:.1f}% ({s[sc][1]:.1f})" for sc in SCEN) + " |")
