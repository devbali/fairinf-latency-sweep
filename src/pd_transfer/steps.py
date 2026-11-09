"""Every probe prefill step of a pd_load.py run with millisecond logs, with the prefill steps around it from the
prefill server's log and the decode server's steps from its log. Writes a pickle of one dict per probe step.

The prefill server logs one line per step, in launch order, after processing its results. A step first runs the
rest of a prompt cut in the step before, then the requests that started in it, in queue order, up to its logged
tokens; the last one that does not fit is cut, and its rest runs first in the next step. Replaying the lines in
order with the steps in start order assigns each step its line and finds every cut.

Cut prompts are grouped with the step that ran their last chunk.

Usage: python steps.py RUN_DIR OUT.pkl"""
import json, os, pickle, sys
from collections import Counter, defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pd_load_report import read, busy_spans, found
from pd_fits import steps, SAME_STEP
import logs

BUDGET = 8192
CACHE_HIT = 2  # tokens per request a random prompt can match in the radix cache, and so not run
run, out = sys.argv[1], sys.argv[2]
records = read(run)
plog = logs.read(found(os.path.join(run, "prefill.log")), logs.PREFILL)
dlog = logs.read(found(os.path.join(run, "decode.log")), logs.DECODE)
dlog_t = np.array([d["t"] for d in dlog])

timed = sorted((r for r in records if r["start"] is not None and r["end"] is not None), key=lambda r: r["start"])
groups = [[timed[0]]]
for r in timed[1:]:
    if r["start"] - groups[-1][-1]["start"] > SAME_STEP:
        groups.append([])
    groups[-1].append(r)


def entered(r):
    """When the request joined the waiting queue, which the scheduler fills a step from in order (FCFS)."""
    log = r["prefill"].get("log") or {}
    return r["start"] - log.get("queue_duration", 0.0) / 1000, r["prefill"].get("request_received_ts", 0.0)


allsteps = []
for g in groups:
    g = sorted(g, key=entered)
    allsteps.append({"start": g[0]["start"], "first_end": min(r["end"] for r in g), "lens": [r["len"] for r in g],
                     "rids": [r["rid"] for r in g], "line": None, "cut": None, "cut_rest": 0})
group_of = {rid: i for i, s in enumerate(allsteps) for rid in s["rids"]}

# Replay the prefill log
lines = [{"ran": [], "group": None} for _ in plog]
carry, gi = 0, 0
for j, p in enumerate(plog):
    cont = min(carry, BUDGET, p["tokens"])
    carry -= cont
    new = p["tokens"] - cont
    lines[j]["ran"] = [cont] if cont else []
    if new > 0 and gi < len(allsteps) and allsteps[gi]["start"] <= p["t"] + 0.001:
        s = allsteps[gi]
        s["line"], lines[j]["group"] = j, gi
        # #cached-token counts the new requests' radix-cache hits, which they did not run
        if cont and p["tokens"] + p["cached"] == sum(s["lens"]):
            carry, cont, new = 0, 0, p["tokens"]  # it ran exactly its new requests: no rest was left to run
            lines[j]["ran"] = []
        short = sum(s["lens"]) - new - p["cached"]
        if short > 0:
            s["cut"], s["cut_rest"] = s["rids"][-1], short
            carry += short
            lines[j]["ran"] += s["lens"][:-1] + [s["lens"][-1] - short]
        else:
            lines[j]["ran"] += s["lens"]
        gi += 1
seqs_ok = Counter(plog[s["line"]]["seqs"] - len(s["lens"]) for s in allsteps if s["line"] is not None)
print(f"replay: {sum(s['line'] is not None for s in allsteps)} of {len(allsteps)} steps got a line; "
      f"{sum(s['cut'] is not None for s in allsteps)} cut a prompt; line #new-seq minus the step's requests: "
      f"{dict(seqs_ok.most_common(4))}")

busy = busy_spans(records)  # (join, end, rid): each request's time in the decode server's running batch
busy_start = np.array([b[0] for b in busy])
by_rid = {r["rid"]: r for r in records}

batches = defaultdict(list)
for r in records:
    if r["kind"] == "probe":
        batches[r["tag"], r["index"], r["sample"]].append(r)
rows = []
for (tag, index, sample), batch in sorted(batches.items()):
    if not all(r["ok"] for r in batch):
        continue
    # A cut prompt belongs to the step that ran its last chunk: move it from the step that started it to the next
    # one (whose first work is its rest), or to a group of its own if the batch has no next step.
    regrouped = [list(st) for st in steps(batch)]
    moved = set()
    for k in range(len(regrouped)):
        cut = allsteps[group_of[regrouped[k][0]["rid"]]]["cut"]
        mine = [r for r in regrouped[k] if r["rid"] == cut and r["rid"] not in moved]
        if mine and len(regrouped[k]) > 1:
            regrouped[k].remove(mine[0]); moved.add(cut)
            if k + 1 < len(regrouped):
                regrouped[k + 1].insert(0, mine[0])
            else:
                regrouped.append(mine)
    for k, step in enumerate(regrouped):
        step = sorted(step, key=lambda r: r["end"])
        ends, kvs = [r["end"] for r in step], sorted(r["kv"] for r in step)
        joins = sorted(r["join"] for r in step)
        lens = sorted((r["len"] for r in step), reverse=True)
        own = {r["rid"] for r in step}
        carried = any(r["rid"] in moved for r in step)
        native = [r for r in step if r["rid"] not in moved]
        i = group_of[(native or step)[0]["rid"]]
        s = allsteps[i]
        j = s["line"] if not (carried and len(step) == 1) else (s["line"] + 1 if s["line"] is not None else None)
        nxt = lines[j + 1] if j is not None and j + 1 < len(lines) else None
        nxt_group = allsteps[nxt["group"]] if nxt and nxt["group"] is not None else None
        # the step behind is running while this one posts if it runs the rest of a cut prompt, or its requests
        # started before this step's results were processed
        behind_running = bool(nxt) and (nxt["ran"][:1] != [] and (nxt_group is None or nxt_group["start"] < ends[0]))
        a = int(np.searchsorted(dlog_t, ends[0] - 0.5))
        b = int(np.searchsorted(dlog_t, max(kvs[-1], joins[-1]) + 0.5))
        spans = [(busy[x][0], busy[x][1], busy[x][2] in own, by_rid[busy[x][2]]["len"])
                 for x in range(int(np.searchsorted(busy_start, kvs[-1] + 0.5))) if busy[x][1] > ends[0] - 0.5]
        found_t = [r["end"] + r["prefill"]["log"]["transfer_total"] / r["prefill"]["log"]["transfer_speed"] / 1000
                   for r in step if (r["prefill"].get("log") or {}).get("transfer_speed")]
        rows.append({
            "scenario": tag, "index": index, "sample": sample, "k": k, "lens": lens, "key": json.dumps(lens),
            "N": len(step), "S": sum(lens), "M": max(lens), "posted_lens": [r["len"] for r in step],
            "first_end": ends[0], "last_end": ends[-1], "ends": ends, "kvs": kvs, "last_kv": kvs[-1],
            "T": 1000 * (kvs[-1] - ends[0]), "joins": joins, "TJ": 1000 * (joins[-1] - ends[0]), "dlog_before": dlog_t[max(0, a - 3):a].tolist(), "E": 1000 * (ends[-1] - ends[0]), "D": 1000 * (kvs[-1] - ends[-1]),
            "found_last": max(found_t) if len(found_t) == len(step) else None,
            "mbs": [(r["prefill"].get("log") or {}).get("transfer_total") for r in step],
            "gi": i, "line": j, "own_cut": s["cut"] in own and len(step) > 1 and s["cut"] not in moved, "carried_cut": carried, "cut_rest": s["cut_rest"],
            "group_lens": s["lens"], "behind": nxt["ran"] if behind_running else [], "behind_running": behind_running,
            "decode_steps": dlog_t[a:b].tolist(), "decode_running": [dlog[x]["running"] for x in range(a, b)],
            "decode_spans": spans,
        })
with open(out, "wb") as f:
    pickle.dump({"rows": rows}, f)
print(f"{len(rows)} probe steps; {sum(r['own_cut'] for r in rows)} had their own last prompt cut; "
      f"{sum(r['behind_running'] for r in rows)} had a step running behind")
