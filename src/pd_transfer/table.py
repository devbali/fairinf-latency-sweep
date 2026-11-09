"""The decode-batch barometer for every posting model: which decode batch a prefill step's last KV joins, against the
predicted one, overall, without slow copies, and by group size. Usage: python3 table.py [WORK_DIR]"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figure import offset

MODELS = ["component model", "linear", "degree 2", "degree 2 + sum l^2", "selected", "random forest", "gradient boosting"]
CATS = (-2, -1, 0, 1, 2)
work = sys.argv[1] if len(sys.argv) > 1 else "."


def shares(off):
    return [round(100 * np.mean(off == c), 1) for c in CATS]


out = {}
print("| posting model | steps | 2+ earlier | 1 earlier | predicted | next | 2+ later | predicted or next | without slow copies: predicted / predicted or next |")
print("|---|---|---|---|---|---|---|---|---|")
for m in MODELS:
    rows = pickle.load(open(os.path.join(work, f"scored_{m.replace(' ', '_').replace('+', 'p').replace('^', '')}.pkl"), "rb"))
    off = np.clip([offset(r) for r in rows], -2, 2)
    N = np.array([r["N"] for r in rows]); slow = np.array([r["D"] > 50 for r in rows])
    p, q = shares(off), shares(off[~slow])
    d = {"steps": len(rows), "all": p, "no_slow": q, "slow_share": round(100 * slow.mean(), 1)}
    for lo, hi, lab in ((1, 2, "1"), (2, 9, "2-8"), (9, 33, "9-32"), (33, 10**9, "33+")):
        s = (N >= lo) & (N < hi); d[lab] = shares(off[s]); d[lab + "_n"] = int(s.sum())
    out[m] = d
    print(f"| {m} | {len(rows)} | " + " | ".join(f"{x}%" for x in p) + f" | {p[2] + p[3]:.1f}% | {q[2]:.1f}% / {q[2] + q[3]:.1f}% |")
json.dump(out, open(os.path.join(work, "barometer.json"), "w"), indent=1)
