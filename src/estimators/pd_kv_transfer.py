"""PD disaggregation KV migration time of one request: gpt-oss-20b, NIXL over NVLink, 1 prefill + 1 decode A100.

From the moment SGLang starts sending the request's KV cache (its prefill_finished_time) until the KV is usable on
the decode server (NIXL's last notification for it). The prefill scheduler sends a step's requests one at a time in
batch order, and the whole batch becomes ready together after its last send, so a request waits for every send
still ahead of it in its batch, including its own. Sends that run while the next prefill step is on the GPU are
slowed to about D_next / 50 each.

    T = c0 + c1 reqs_left + c2 pieces_left + c3 MB_left + c4 n_ov + c5 n_ov * D_next

reqs_left    requests of the batch still to send, this one included                         degree 1
pieces_left  sum over them of min(KV pieces, 32) + state pieces                               degree 1
MB_left      their KV bytes, MB                                                              degree 1
n_ov         sends that overlap the next prefill step: min(reqs_left, max(0, 50 (D_next - 82) / D_next - place))
                                                                                              degree 1
n_ov*D_next                                                                                   degree 2

A piece is one contiguous run of destination slots in the decode server's KV pool (a NIXL write descriptor).
D_next is the next prefill step's duration in ms (at least 89.4, the eager launch floor); the fit used the latency
sweep's eager prefill model of results/gptoss-clean-20261028 for it. place is the request's 0-based position in
its batch.

Fitted (Huber) on results/gptoss-pd-kv-migration-20261111/requests.csv.gz: 58,058 probe requests of the
pd_load.py runs gptoss-pd-xfer-instr-{nopoll,rest}-nixl-20261111. Held out by probe batch (5 folds), median / p90 /
p99 absolute error 1.48 / 6.07 / 21.6 ms.

Input: a dict {"sends": [{"kv_pieces": .., "state_pieces": .., "MB": ..}, ...], "place": k, "d_next_ms": D}, where
sends lists the requests still to send, this one first. Output: ms.

Usage: python -m estimators.pd_kv_transfer   (from src/) prints the fit and the percentile estimators.
"""

import gzip
import os
from functools import lru_cache

import numpy as np

from .estimator import Model, report
from .steps import RESULTS

DATA = os.path.join(RESULTS, "gptoss-pd-kv-migration-20261111", "requests.csv.gz")
LAUNCH_MS = 82.0      # the next step's launch: sends start about this long after it began on the GPU
SENDS_PER_STEP = 50   # sends that fit in one step's GPU time (about 2 per layer)
PIECE_CAP = 32        # beyond 32 pieces a KV write costs about the same


def n_overlap(reqs_left, place, d_next_ms):
    return min(reqs_left, max(0.0, SENDS_PER_STEP * max(0.0, d_next_ms - LAUNCH_MS) / d_next_ms - place))


def _row(reqs_left, pieces_left, mb_left, n_ov, d_next_ms):
    return [1.0, reqs_left, pieces_left, mb_left, n_ov, n_ov * d_next_ms]


def features(x):
    sends, d = x["sends"], x["d_next_ms"]
    pieces = sum(min(s["kv_pieces"], PIECE_CAP) + s["state_pieces"] for s in sends)
    mb = sum(s["MB"] for s in sends)
    return np.array(_row(len(sends), pieces, mb, n_overlap(len(sends), x["place"], d), d), dtype=float)


@lru_cache(maxsize=None)
def _data():
    X, y = [], []
    with gzip.open(DATA, "rt") as f:
        cols = f.readline().strip().split(",")
        for line in f:
            r = dict(zip(cols, line.strip().split(",")))
            X.append(_row(float(r["reqs_left"]), float(r["pieces_left"]), float(r["MB_left"]), float(r["n_ov"]),
                          float(r["d_next_ms"])))
            y.append(float(r["migration_ms"]))
    y = np.array(y)
    return np.array(X), y, np.arange(len(y))  # every request is its own sample


MODEL = Model(
    name="PD KV migration, gpt-oss-20b, NIXL over NVLink",
    terms=("const", "reqs_left", "pieces_left", "MB_left", "n_ov", "n_ov*D_next"),
    degrees=(0, 1, 1, 1, 1, 2),
    features=features,
    data=_data,
    loss="huber",
)

if __name__ == "__main__":
    report(MODEL)
