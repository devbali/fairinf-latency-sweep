"""The clean run's selected prefill and decode models (no piecewise CUDA graphs, as in PD mode), fitted as compare_fits.py fits them."""
import json, os, pickle, sys
import numpy as np
SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
src = open(f"{SRC}/compare_fits.py").read()
ns = {}
exec(src[:src.index("parser = argparse.ArgumentParser")], ns)  # compare_fits.py's helpers, without running it
from fit import load, fit, predict

RUN = os.path.join(SRC, "..", "results", "gptoss-clean-20261028", "results.jsonl")

def selected(phase):
    samples = load([RUN])
    keys = sorted(k for k in samples if k[0] == phase)
    batches = [json.loads(b) for _, b in keys]
    y = np.array([np.median(samples[k]) for k in keys]) * 1e3
    floor, bends = phase == "prefill", phase == "decode"
    rows = [ns["terms"](phase, b, ns["MODELS"][2], bends) for b in batches]
    names = list(rows[0]); X = np.array([list(r.values()) for r in rows], dtype=float)
    _, kept, _ = ns["eliminate"](names, X, y, floor)
    cols = ns["columns"](names, kept)
    return {"phase": phase, "names": names, "cols": cols, "kept": kept, "bends": bends, "model": fit(X[:, cols], y, floor)}

def step_ms(m, lens):
    if not lens:
        return 0.0
    x = np.array(list(ns["terms"](m["phase"], lens, ns["MODELS"][2], m["bends"]).values()), dtype=float)[m["cols"]]
    return float(predict(m["model"], x[None, :])[0])

if __name__ == "__main__":
    models = {p: selected(p) for p in ("prefill", "decode")}
    for p, m in models.items():
        f, w = m["model"]
        print(p, "floor", round(f, 2), dict(zip(["const"] + m["kept"], np.round(w, 7))))
    pickle.dump(models, open("stepmodels.pkl", "wb"))
    pm, dm = models["prefill"], models["decode"]
    for lens in ([2048], [4096], [2048, 2048], [8192], [313, 453, 652, 622], [128]):
        print("prefill", lens, round(step_ms(pm, lens), 1), "ms")
    for n, ctx in ((32, 3256), (32, 900), (32, 5500), (2, 1000)):
        print("decode", n, "x", ctx, round(step_ms(dm, [ctx] * n), 1), "ms")
