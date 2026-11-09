"""Report on a pd_load.py run: what a PD KV transfer's time depends on besides its batch.

Usage: python src/pd_load_report.py RUN_DIR

RUN_DIR is pd_load.py's --out directory: results.jsonl, decode.log and, from the
scenarios run with --profile, profiles/ (each file may be gzipped). Requests'
times are pd_fits.py's. Each prefill step of a probe batch (pd_fits.steps)
gets:
- T = E + D, ms: E is the posting span, from the step's first end to its last,
  while the prefill scheduler posts one transfer after another; D is the tail,
  from its last end to the decode server finding its last KV;
- pred, the selected model of results/gptoss-pd-nixl-20261103 (SELECTED);
- next_prefill: another prefill step, a probe's or a load's, launched before the
  step's last end. With overlap scheduling the step's results were then
  processed while that step ran;
- decode_busy: another request was in the decode server's running batch while
  the step's KV landed.

It prints, as markdown:
- by scenario: the probes' steps against the selected model, and how often each
  state came up;
- T by probe and scenario;
- E and D by state, and the tail of one-request steps;
- for a pressure scenario: the waits before the transfer, the retractions and
  the decode server's queues, from its "Decode batch" log lines;
- from py-spy's raw profiles: what each scheduler's event loop ran, its hottest
  lines, and what posting a transfer cost the prefill server.
"""

import calendar
import glob
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np

from pd_fits import SAME_STEP, open_results, steps

SELECTED = (7.919, 2.111, 0.0007932, -0.001005)  # T = c + a N + b S + d M, ms
DECODING_LOADS = ("decode", "pressure")  # name prefixes of loads whose requests stay in the decode server's running batch
ORDER = ("idle", "prefill", "decode", "both", "pressure")
SAMPLE_MS = 4  # one py-spy sample at --rate 250
D_BINS = [0, 4, 6, 8, 10, 12, 15, 20, 50, 1e9]  # ms: the one-request tail's histogram
MAX_PROBE_ROWS = 20  # more probe batches than this: no table by probe

DECODE_LINE = re.compile(
    r"^\[(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\]]*\] Decode batch[^,]*, #running-req: (?P<running>\d+), "
    r".*?full token usage: (?P<usage>[\d.]+).*?#prealloc-req: (?P<prealloc>\d+), #transfer-req: (?P<transfer>\d+), "
    r"#retracted-req: (?P<retracted>\d+).*gen throughput \(token/s\): (?P<throughput>[\d.]+)")
RETRACT_LINE = re.compile(r"^\[(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)[^\]]*\].*KV cache pool is full\. Retract requests\. "
                          r"#retracted_reqs: (?P<n>\d+)")
FRAME = re.compile(r"^(?P<name>.+?) \((?P<file>[^()]*?)(?::(?P<line>\d+))?\)$")


def found(path):
    """path or path.gz, whichever exists, or None."""
    return next((p for p in (path, path + ".gz") if os.path.exists(p)), None)


def read(run):
    """Every request of the run, with its tag (scenario, -profile when profiled), load and times."""
    records = [json.loads(line) for line in open_results(found(os.path.join(run, "results.jsonl")))]
    for r in records:
        p, d = r["prefill"], r["decode"]
        r["tag"], _, r["load"] = r["phase"].partition(":")
        r["start"], r["end"] = p.get("forward_entry_time"), p.get("prefill_finished_time")
        r["join"] = d.get("forward_entry_time")
        r["kv"] = r["join"] - d["queue_time"] if r["join"] is not None and "queue_time" in d else None
        r["ok"] = "error" not in p and "error" not in d and None not in (r["start"], r["end"], r["kv"])
    return records


def busy_spans(records):
    """(start, end, rid) of each request's time in the decode server's running batch."""
    spans = []
    for r in records:
        if not (r["kind"] in ("probe", "warmup") or r["load"].startswith(DECODING_LOADS)) or r["end"] is None:
            continue
        start = r["join"] if r["join"] is not None else r["end"]  # an aborted request: from its prefill's end
        end = r["decode"].get("request_finished_ts") or r["client"].get("decode")
        if end and end > start:
            spans.append((start, end, r["rid"]))
    return sorted(spans)


def logged(requests, side, field):
    """The median of a ReqTimeStats field over the requests that logged it, or nan."""
    values = [(r[side].get("log") or {}).get(field) for r in requests]
    values = [v for v in values if v is not None]
    return np.median(values) if values else np.nan


def probe_steps(records):
    """One row per prefill step of each probe batch whose requests all got both answers, and the broken batches."""
    starts = np.array(sorted(r["start"] for r in records if r["start"] is not None))
    busy = busy_spans(records)
    busy_start = np.array([b[0] for b in busy])
    batches = defaultdict(list)
    for r in records:
        if r["kind"] == "probe":
            batches[r["tag"], r["index"], r["sample"]].append(r)
    rows, broken = [], Counter()
    for (tag, index, sample), batch in sorted(batches.items()):
        if not all(r["ok"] for r in batch):
            broken[tag] += 1
            continue
        split = steps(batch)
        for step in split:
            ends, kvs = sorted(r["end"] for r in step), sorted(r["kv"] for r in step)
            lens = [r["len"] for r in step]
            n, s, m = len(step), sum(lens), max(lens)
            own = {r["rid"] for r in step}
            i = np.searchsorted(busy_start, kvs[-1])
            rows.append({
                "tag": tag, "scenario": tag.removesuffix("-profile"), "profiled": tag.endswith("-profile"),
                "index": index, "sample": sample, "batch": tuple(r["len"] for r in batch), "K": len(split),
                "N": n, "T": 1000 * (kvs[-1] - ends[0]), "E": 1000 * (ends[-1] - ends[0]),
                "D": 1000 * (kvs[-1] - ends[-1]), "spread": 1000 * (kvs[-1] - kvs[0]),
                "pred": SELECTED[0] + SELECTED[1] * n + SELECTED[2] * s + SELECTED[3] * m,
                "next_prefill": bool(((starts > step[0]["start"] + SAME_STEP) & (starts < ends[-1])).any()),
                "decode_busy": any(e > ends[-1] and rid not in own for _, e, rid in busy[:i]),
                "wait": 1000 * (step[0]["start"] - step[0]["client"]["sent"]),
                "bootstrap": logged(step, "prefill", "bootstrap_duration"),
                "alloc_wait": logged(step, "decode", "alloc_wait_duration"),
            })
    return rows, broken


def spread(values, fmt="{:.1f}"):
    """median [p10-p90] (count)."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return "-"
    return (fmt.format(np.median(values)) + f" [{np.percentile(values, 10):.1f}-{np.percentile(values, 90):.1f}] "
            f"({len(values)})")


def ordered(names):
    return [s for s in ORDER if s in names] + sorted(set(names) - set(ORDER))


def label(batch):
    return f"{len(batch)} x {batch[0]}" if len(set(batch)) == 1 else str(list(batch))


def report_transfer(records, rows, broken):
    main = [r for r in rows if not r["profiled"]]
    names = ordered({r["scenario"] for r in main})
    print("## The probes' prefill steps by scenario (passes without the profiler)\n")
    print("| scenario | batches | broken | split into 2+ steps | steps | T median (ms) | T p90 | T / selected model | "
          "next_prefill | decode_busy | posting per extra request (ms) | tail D (ms) | wait to launch (ms) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name in names:
        b = [r for r in main if r["scenario"] == name]
        sent = {(r["index"], r["sample"]) for r in b}
        split = {(r["index"], r["sample"]) for r in b if r["K"] > 1}
        T = [r["T"] for r in b]
        per = [r["E"] / (r["N"] - 1) for r in b if r["N"] > 1]
        print(f"| {name} | {len(sent)} | {broken.get(name, 0)} | {100 * len(split) / len(sent):.0f}% | {len(b)} | "
              f"{np.median(T):.1f} | {np.percentile(T, 90):.1f} | {np.median([r['T'] / r['pred'] for r in b]):.2f} | "
              f"{100 * np.mean([r['next_prefill'] for r in b]):.0f}% | {100 * np.mean([r['decode_busy'] for r in b]):.0f}% | "
              f"{np.median(per) if per else np.nan:.2f} | {np.median([r['D'] for r in b]):.1f} | "
              f"{np.median([r['wait'] for r in b]):,.0f} |")

    probes = sorted({(r["index"], r["batch"]) for r in main})
    if len(probes) <= MAX_PROBE_ROWS:
        print("\n## T (ms) by probe: median [p10-p90] (steps)\n")
        print("| probe | " + " | ".join(names) + " |")
        print("|---|" + "---|" * len(names))
        for index, batch in probes:
            cells = [spread([r["T"] for r in main if r["scenario"] == name and r["index"] == index]) for name in names]
            print(f"| {label(batch)} | " + " | ".join(cells) + " |")
    else:
        print(f"\n{len(probes)} probe batches: `python src/compare_fits.py RESULTS --pd-phase SCENARIO` fits each "
              "scenario's transfer time.")

    states = [(nf, db) for nf in (False, True) for db in (False, True)]
    print("\n## By state: posting per extra request E/(N-1) and tail D (ms), median [p10-p90] (steps)\n")
    print("| next_prefill | decode_busy | from scenarios | posting per extra request | tail D | D under 6 ms |")
    print("|---|---|---|---|---|---|")
    for nf, db in states:
        b = [r for r in main if r["next_prefill"] == nf and r["decode_busy"] == db]
        if b:
            D = np.array([r["D"] for r in b])
            where = Counter(r["scenario"] for r in b)
            print(f"| {nf} | {db} | {', '.join(f'{k} {v}' for k, v in where.most_common())} | "
                  f"{spread([r['E'] / (r['N'] - 1) for r in b if r['N'] > 1], '{:.2f}')} | {spread(D)} | "
                  f"{100 * (D < 6).mean():.0f}% |")
    print("\nTail D by scenario and state:\n")
    print("| scenario | " + " | ".join(f"next_prefill {nf}, decode_busy {db}" for nf, db in states) + " |")
    print("|---|" + "---|" * len(states))
    for name in names:
        cells = [spread([r["D"] for r in main if r["scenario"] == name and (r["next_prefill"], r["decode_busy"]) == s])
                 for s in states]
        print(f"| {name} | " + " | ".join(cells) + " |")

    print("\n## When the KVs land (ms), median [p10-p90] (count)\n")
    print("The prefill server logs each transfer's latency (transfer_total / transfer_speed): from the start of "
          "posting it to the prefill scheduler's first poll that found it done.\n")
    print("| scenario | steps of 4+ requests: posting span E | their KVs' spread, last kv - first kv | "
          "probe requests: kv - end | the prefill's transfer latency | kv after the prefill found it done |")
    print("|---|---|---|---|---|---|")
    for name in names:
        multi = [r for r in main if r["scenario"] == name and r["N"] >= 4]
        delay, latency = [], []
        for r in records:
            log = r["prefill"].get("log") or {}
            if r["tag"] == name and r["kind"] == "probe" and r["ok"] and log.get("transfer_speed"):
                delay.append(1000 * (r["kv"] - r["end"]))
                latency.append(log["transfer_total"] / log["transfer_speed"])  # MB / (GB/s) = ms
        print(f"| {name} | {spread([r['E'] for r in multi])} | {spread([r['spread'] for r in multi])} | "
              f"{spread(delay)} | {spread(latency)} | {spread(np.subtract(delay, latency))} |")

    print("\n## The tail D of one-request steps (ms): how many steps in each range\n")
    bins = [f"{lo:g}-{hi:g}" if hi < 1e9 else f"{lo:g}+" for lo, hi in zip(D_BINS, D_BINS[1:])]
    print("| scenario | median [p10-p90] | " + " | ".join(bins) + " |")
    print("|---|---|" + "---|" * len(bins))
    for name in names:
        D = [r["D"] for r in main if r["scenario"] == name and r["N"] == 1]
        if D:
            counts, _ = np.histogram(D, bins=D_BINS)
            print(f"| {name} | {spread(D)} | " + " | ".join(str(c) for c in counts) + " |")

    profiled = [r for r in rows if r["profiled"]]
    if profiled:
        names = ordered({r["scenario"] for r in profiled})
        print("\n## T median (ms) by probe, without / with py-spy on both schedulers\n")
        print("| probe | " + " | ".join(names) + " |")
        print("|---|" + "---|" * len(names))
        for index, batch in sorted({(r["index"], r["batch"]) for r in main}):
            cells = []
            for name in names:
                a = [r["T"] for r in main if r["scenario"] == name and r["index"] == index]
                b = [r["T"] for r in profiled if r["scenario"] == name and r["index"] == index]
                cells.append(f"{np.median(a) if a else np.nan:.1f} / {np.median(b) if b else np.nan:.1f}")
            print(f"| {label(batch)} | " + " | ".join(cells) + " |")


def epoch(ts):
    return calendar.timegm(time.strptime(ts, "%Y-%m-%d %H:%M:%S"))  # SGLang logs local time; the VM runs on UTC


def windows(records):
    """Each tag's first send and last answer, in time order."""
    span = {}
    for r in records:
        if r["kind"] == "warmup":
            continue
        c = r["client"]
        lo, hi = c["sent"], max(v for v in (c.get("prefill"), c.get("decode"), c["sent"]) if v)
        a, b = span.get(r["tag"], (lo, hi))
        span[r["tag"]] = (min(a, lo), max(b, hi))
    return dict(sorted(span.items(), key=lambda kv: kv[1][0]))


def report_decode_log(run, records):
    """The decode server's 'Decode batch' lines (one per 40 decode steps) and retractions, by tag."""
    path = found(os.path.join(run, "decode.log"))
    if not path:
        return
    lines, retracts = [], []
    for line in open_results(path):
        if m := DECODE_LINE.match(line):
            lines.append({k: epoch(v) if k == "ts" else float(v) for k, v in m.groupdict().items()})
        elif m := RETRACT_LINE.match(line):
            retracts.append((epoch(m["ts"]), int(m["n"])))
    print("\n## The decode server's log over each scenario\n")
    print("| tag | log lines | running requests, median | decode step (ms), median | full token usage, max | "
          "prealloc queue, mean / max | transfer queue, max | retracted queue, mean / max | retractions | requests retracted |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for tag, (lo, hi) in windows(records).items():
        b = [x for x in lines if lo - 1 <= x["ts"] <= hi + 1]
        ev = [n for t, n in retracts if lo - 1 <= t <= hi + 1]
        if not b:
            continue
        g = lambda k: np.array([x[k] for x in b])
        step = [1000 * x["running"] / x["throughput"] for x in b if x["throughput"] > 0 and x["running"] >= 8]
        steady = len(b) >= 10 and np.median(g("running")) >= 8  # else mostly idle: a line spans idle time
        step = f"{np.median(step):.1f}" if steady else "-"
        print(f"| {tag} | {len(b)} | {np.median(g('running')):.0f} | {step} | "
              f"{g('usage').max():.2f} | {g('prealloc').mean():.1f} / {g('prealloc').max():.0f} | "
              f"{g('transfer').max():.0f} | {g('retracted').mean():.1f} / {g('retracted').max():.0f} | "
              f"{len(ev)} | {sum(ev)} |")
    print("\nThe decode step is #running-req / gen throughput, from lines with 8 or more running requests.")


def report_pressure(records, rows):
    tags = [t for t in windows(records) if t.startswith("pressure")]
    if not tags:
        return
    print("\n## Memory pressure: the waits before a transfer (ms), median / p90 / max\n")
    print("| tag | kind | requests | aborted or failed | prefill bootstrap | decode alloc wait | decode transfer | "
          "retracted at least once |")
    print("|---|---|---|---|---|---|---|---|")
    for tag in tags:
        for kind in ("probe", "load"):
            b = [r for r in records if r["tag"] == tag and r["kind"] == kind]
            if not b:
                continue

            def stat(side, field):
                v = [r[side]["log"][field] for r in b if r[side].get("log") and field in r[side]["log"]]
                return f"{np.percentile(v, 50):,.0f} / {np.percentile(v, 90):,.0f} / {max(v):,.0f}" if v else "-"

            print(f"| {tag} | {kind} | {len(b)} | {sum(not r['ok'] for r in b)} | "
                  f"{stat('prefill', 'bootstrap_duration')} | {stat('decode', 'alloc_wait_duration')} | "
                  f"{stat('decode', 'transfer_duration')} | {sum((r['decode'].get('num_retractions') or 0) > 0 for r in b)} |")
    p = [r for r in rows if r["scenario"] == "pressure"]
    print(f"\nThe probes' steps, both passes: T {spread([r['T'] for r in p])}, tail D {spread([r['D'] for r in p])}, "
          f"wait to launch {spread([r['wait'] for r in p], '{:,.0f}')} ms.")


def function(frame, line=False):
    m = FRAME.match(frame)
    if not m:
        return frame
    where = os.path.basename(m["file"]) + (f":{m['line']}" if line and m["line"] else "")
    return f"{m['name']} ({where})"


def loop_stacks(path):
    """The scheduler loop thread's stacks (frames, root first) and sample counts, from a py-spy raw profile."""
    threads = defaultdict(Counter)
    for line in open_results(path):
        stack, _, count = line.rstrip("\n").rpartition(" ")
        if stack and count.isdigit():
            frames = stack.split(";")
            threads[frames[0]][tuple(frames[1:])] += int(count)
    reach = lambda t: sum(c for s, c in threads[t].items() if any(f.startswith("event_loop") for f in s))
    return threads[max(threads, key=reach)] if threads else Counter()


def under(stacks, prefix):
    """Samples by the frame right under the first one starting with prefix."""
    out = Counter()
    for s, c in stacks.items():
        for i, f in enumerate(s):
            if f.startswith(prefix):
                out[function(s[i + 1]) if i + 1 < len(s) else "(itself)"] += c
                break
    return out


def posting_part(rest):
    """What a sample under process_batch_result_disagg_prefill was doing."""
    if rest.startswith("synchronize"):
        return "wait for the step's results (copy_done.synchronize)"
    if "add_transfer_request" in rest:
        return "NIXL: build and post the transfers (add_transfer_request)"
    if "send_kv_chunk (prefill.py:770)" in rest or "send_kv_chunk (prefill.py:806)" in rest:
        return "copy the KV indices to the CPU (.cpu())"
    if "cache_unfinished_req" in rest:
        return "radix cache insert"
    if "report_prefill_stats" in rest:
        return "stats"
    return "other"


def table(title, columns, top):
    """columns: tag -> (Counter, total). The top keys over all columns, as shares of each total."""
    keys = Counter()
    for counts, total in columns.values():
        for k, v in counts.items():
            keys[k] += v / max(total, 1)
    print(f"\n{title}\n")
    print("| | " + " | ".join(f"{tag} ({total:,} samples)" for tag, (_, total) in columns.items()) + " |")
    print("|---|" + "---|" * len(columns))
    for k, _ in keys.most_common(top):
        print(f"| {k} | " + " | ".join(f"{100 * c[k] / max(t, 1):.1f}%" for c, t in columns.values()) + " |")


def report_profiles(run, records):
    files = sorted(glob.glob(os.path.join(run, "profiles", "*.txt")) + glob.glob(os.path.join(run, "profiles", "*.txt.gz")))
    if not files:
        return
    by_server = defaultdict(dict)
    for path in files:
        tag, _, server = re.sub(r"\.txt(\.gz)?$", "", os.path.basename(path)).rpartition("-")
        by_server[server][tag.removesuffix("-profile")] = loop_stacks(path)
    print("\n## py-spy: the scheduler loop threads (passes with --profile)")
    for server, tags in by_server.items():
        tags = {t: tags[t] for t in ordered(tags)}
        total = {t: sum(s.values()) for t, s in tags.items()}
        print(f"\n### {server} server")
        table("What the event loop ran, share of the loop thread's samples:",
              {t: (under(s, "event_loop"), total[t]) for t, s in tags.items()}, 8)
        leaves = {t: Counter() for t in tags}
        for t, s in tags.items():
            for stack, c in s.items():
                leaves[t][function(stack[-1], line=True) if stack else "(empty)"] += c
        table("Its hottest lines (the innermost frame):", {t: (leaves[t], total[t]) for t, s in tags.items()}, 6)
        if server != "prefill":
            continue
        posted = Counter(r["tag"].removesuffix("-profile") for r in records
                         if r["tag"].endswith("-profile") and r["end"] is not None)
        parts = {t: Counter() for t in tags}
        for t, s in tags.items():
            for stack, c in s.items():
                i = next((i for i, f in enumerate(stack) if f.startswith("process_batch_result_disagg_prefill")), None)
                if i is not None:
                    parts[t][posting_part(";".join(stack[i + 1:]))] += c
        print("\nProcessing a step's results, ms of the loop thread per request posted:\n")
        print("| | " + " | ".join(f"{t} ({posted[t]} requests)" for t in tags) + " |")
        print("|---|" + "---|" * len(tags))
        for k in sorted({k for p in parts.values() for k in p}, key=lambda k: -sum(p[k] for p in parts.values())):
            print(f"| {k} | " + " | ".join(f"{SAMPLE_MS * parts[t][k] / max(posted[t], 1):.2f}" for t in tags) + " |")


def main(run):
    records = read(run)
    rows, broken = probe_steps(records)
    print(f"# {os.path.basename(os.path.normpath(run))}\n")
    print("Requests by tag and kind: " + ", ".join(f"{t} {k} {n}" for (t, k), n in
                                                 Counter((r["tag"], r["kind"]) for r in records).items()) + "\n")
    report_transfer(records, rows, broken)
    report_pressure(records, rows)
    report_decode_log(run, records)
    report_profiles(run, records)


if __name__ == "__main__":
    main(sys.argv[1])
