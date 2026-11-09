"""Transfer time to the decode batch: from a prefill step's first end until its last request's decode batch starts
(the decode server's forward_entry_time), TJ. Held out by set of prompt lengths, as component.py.

Prediction: the last post ends at first_end + E (the posting model, held out); its KV is ready c0 + k * MB later;
the decode server picks it up at its next loop boundary (the next "Decode batch" log line) if it is running steps
then, else at once; the request's batch starts cj later. The decode server's step times come from its log, as a
simulator of the decode server would know them.
Usage: REG=posting.pkl MODEL=selected OUT=scored.pkl python3 score.py STEPS.pkl"""
import importlib.util, pickle, sys
import numpy as np
import os
path, nooverlap = sys.argv[1], False
REG = pickle.load(open(os.environ["REG"], "rb")) if os.environ.get("REG") else None
OVERRIDE = dict(zip(REG["rids"], REG["pred"][os.environ["MODEL"]])) if REG else {}
if nooverlap:  # no prefill step runs behind a posting one
    d = pickle.load(open(path, "rb"))
    for r in d["rows"]:
        r["behind_running"], r["behind"], r["own_cut"] = False, [], False
    path = path.replace(".pkl", "_noov.pkl"); pickle.dump(d, open(path, "wb"))
sys.argv = ["component.py", path]
spec = importlib.util.spec_from_file_location("F", os.path.join(os.path.dirname(os.path.abspath(__file__)), "component.py")); F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
F.REL = True
rows = F.rows
keys = sorted({r["key"] for r in rows}); idx = {k: i for i, k in enumerate(keys)}
GAP = 0.08  # s: decode log lines closer than this around the ready time mean the decode server is running steps


def pickup(r, ready):
    """The decode server checks for arrived KVs between steps. It is running a step at the ready time if any request
    is in its batch then (the step's own earlier requests too, which start steps the moment they join); the KV then
    waits for what is left of that step, until its log line."""
    lines = np.array(r["dlog_before"] + r["decode_steps"])
    nxt = lines[lines > ready]
    running = any(j0 <= ready < j1 for j0, j1, own, l in r["decode_spans"])
    if running and len(nxt) and nxt[0] - ready < GAP:
        return nxt[0]
    return ready


def predict(r, E, c0, k, cj, kn=0.0):
    """kn: ms per request of the step, the decode server's work on the step's earlier arrivals before the last."""
    ready = r["first_end"] + (E + c0 + k * (r["mbs"][-1] or 0.0) + kn * (r["N"] - 1)) / 1000
    return 1000 * (pickup(r, ready) - r["first_end"]) + cj


def two_options(r, E, c0, k, cj, kn=0.0):
    """The KV lands before the end of the decode step running at the predicted ready time (picked up then), or after
    it (picked up at the end of the next step). With the decode batch empty: picked up at once, or, if a step starts
    first, at that step's end."""
    ready = r["first_end"] + (E + c0 + k * (r["mbs"][-1] or 0.0) + kn * (r["N"] - 1)) / 1000
    lines = np.array(r["dlog_before"] + r["decode_steps"])
    nxt = lines[lines > ready]
    running = any(j0 <= ready < j1 for j0, j1, own, l in r["decode_spans"])
    if running and len(nxt) and nxt[0] - ready < GAP:
        opts = [nxt[0]] + ([nxt[1]] if len(nxt) > 1 and nxt[1] - nxt[0] < GAP else [])
        opts = [1000 * (o - r["first_end"]) + cj for o in opts]
    else:
        opts = [1000 * (ready - r["first_end"]) + cj] + ([1000 * (nxt[0] - r["first_end"]) + cj] if len(nxt) and nxt[0] - ready < GAP else [])
    return opts


GRID = [(c0, k, cj, kn) for c0 in np.arange(0, 6.01, 0.5) for k in (0.0, 0.02) for cj in np.arange(0, 3.01, 0.25)
        for kn in (0.0, 0.05, 0.1, 0.15, 0.2, 0.3)]


def fit(train, Ekey):
    best = None
    sub = train[::2]
    for c in GRID:
        e = np.median(np.abs([predict(r, r[Ekey], *c) - r["T"] for r in sub]))
        if best is None or e < best[0]:
            best = (e, c)
    return best[1]


for tr, te in F.folds(len(keys)):
    tr, te = set(tr), set(te)
    a = [r for r in rows if idx[r["key"]] in tr]
    wE = F.fit_E(a)
    for r in rows:
        r["E_fit"] = float(np.dot(wE, F.E_feats(r)))
        rid = (r["scenario"], r["index"], r["sample"], r["k"])
        if r["N"] == 1:
            r["E_fit"] = 0.0
        elif rid in OVERRIDE:
            r["E_fit"] = OVERRIDE[rid]
    c_model, c_known = fit(a, "E_fit"), fit(a, "E")
    for r in rows:
        if idx[r["key"]] in te:
            r["two"] = two_options(r, r["E_fit"], *c_model)
            r["two_Eknown"] = two_options(r, r["E"], *c_known)
            r["TJ_pred"] = predict(r, r["E_fit"], *c_model)
            r["TJ_pred_Eknown"] = predict(r, r["E"], *c_known)
            r["E_pred"] = r["E_fit"]

TJ = np.array([r["T"] for r in rows])
slow = np.array([r["D"] > 50 for r in rows])


def line(name, m, key):
    e = np.abs(np.array([r[key] for r in rows]) - TJ)[m]
    print(f"| {name} | {m.sum()} | {np.median(TJ[m]):.1f} | {np.median(e):.2f} | {np.percentile(e, 90):.1f} | {100 * np.mean(e <= 1):.0f}% | {100 * np.mean(e <= 2):.0f}% |")


print(f"fit (last fold): c0 {c_model[0]} ms, {c_model[1]} ms/MB, join {c_model[2]} ms, {c_model[3]} ms per request")
print("| subset | steps | median TJ ms | median abs err ms | p90 ms | within 1 ms | within 2 ms |")
print("|---|---|---|---|---|---|---|")
allm = np.ones(len(rows), bool)
line("all", allm, "TJ_pred")
line("all, posting span known", allm, "TJ_pred_Eknown")
line("no slow copies", ~slow, "TJ_pred")
for sc in F.SCEN:
    line(sc, np.array([r["scenario"] == sc for r in rows]) & ~slow, "TJ_pred")
for lo, hi in ((1, 2), (2, 9), (9, 33), (33, 10**9)):
    line(f"N {lo}-{hi - 1 if hi < 10**9 else '+'}", np.array([lo <= r["N"] < hi for r in rows]) & ~slow, "TJ_pred")
for lo, hi in ((1, 2), (2, 9), (9, 33), (33, 10**9)):
    line(f"N {lo}-{hi - 1 if hi < 10**9 else '+'}, posting known", np.array([lo <= r["N"] < hi for r in rows]) & ~slow, "TJ_pred_Eknown")
pickle.dump(rows, open(os.environ.get("OUT", path.replace(".pkl", "_two.pkl")), "wb"))
