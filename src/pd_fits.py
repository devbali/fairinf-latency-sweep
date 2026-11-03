"""Describe the KV cache transfer of SGLang's PD disaggregation, from pd_runner.py results.

Each request's times come from the two servers' answers (see pd_runner.py):
- start: its prefill step was scheduled (the prefill server's
  forward_entry_time), the same moment for every request of the step;
- end: its prefill forward's result was processed and its KV transfer queued
  (prefill_finished_time). With overlap scheduling that happens after the next
  step is launched;
- kv: the decode server found its KV cache in place (decode forward_entry_time
  minus queue_time), at its first poll after the transfer finished;
- join: it joined a decode step (decode forward_entry_time).
Its delay is kv - end. Requests with the same start ran in one prefill step,
which SGLang may form differently from the batch that was sent. A request's
transfer is queued (mooncake) or posted (nixl) at its end, so a step's
transfers start in the order of their ends.

The transfer time T of a prefill step runs from its first end to the last of
its requests' kv. src/compare_fits.py models it with the same four polynomial
regressions as a step's latency, in the step's N (requests), S (prompt tokens)
and M (longest prompt), with neither a floor nor bends: pass it this
results.jsonl. It reads the steps of the plan's batches through
transfer_samples(), with every step of the same prompt lengths as one batch.

This script prints, as markdown, what goes around that model:
- how SGLang split the sent batches into prefill steps, and the bootstrap
  before prefill;
- what one request's transfer carries, and how long it takes alone;
- the transfer time of the same S split among more requests;
- the decode server's pickup;
- for each burst, whether prefill or transfer set its pace.

Usage: python src/pd_fits.py RESULTS.jsonl[.gz]
"""

import gzip
import json
import sys
from collections import defaultdict

import numpy as np

from fit import lstsq

SAME_STEP = 0.001  # s: requests whose steps were scheduled this close together ran in one step
BUDGET = 8192  # --chunked-prefill-size: longer prompts are prefilled, and sent, in chunks
STALL_MS = 100  # a transfer this slow waited for something
SIZES = [(0, 30), (30, 150), (150, 1000)]  # MB: buckets of a request's transfer size


def open_results(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def is_pd(path):
    """Whether path holds pd_runner.py results, rather than runner.py's."""
    with open_results(path) as f:
        first = f.readline()
    return bool(first) and isinstance(json.loads(first).get("prefill"), dict)


def load(path):
    """Every request with both answers, with its end, kv and join (s) and transfer bytes (MB)."""
    requests, failed = [], 0
    for line in open_results(path):
        r = json.loads(line)
        p, d = r["prefill"], r["decode"]
        if "error" in p or "error" in d:
            failed += 1
            continue
        try:
            r["start"] = p["forward_entry_time"]
            r["end"] = p["prefill_finished_time"]
            r["kv"] = d["forward_entry_time"] - d["queue_time"]
            r["join"] = d["forward_entry_time"]
            r["mb"] = p["log"]["transfer_total"]
            r["bootstrap_ms"] = p["log"]["bootstrap_duration"]
        except (KeyError, TypeError):
            failed += 1
            continue
        requests.append(r)
    return requests, failed


def sent_batches(requests):
    """The requests of each sent batch, by (phase, index, sample), in transfer order."""
    sent = defaultdict(list)
    for r in requests:
        sent[r["phase"], r["index"], r["sample"]].append(r)
    for batch in sent.values():
        batch.sort(key=lambda r: r["end"])
    return sent


def steps(batch):
    """The prefill steps of a sent batch, in the order they ran, each in transfer order."""
    by_start = sorted(batch, key=lambda r: r["start"])
    groups = [[by_start[0]]]
    for r in by_start[1:]:
        if r["start"] - groups[-1][-1]["start"] > SAME_STEP:
            groups.append([])
        groups[-1].append(r)
    return [sorted(group, key=lambda r: r["end"]) for group in groups]


def transfer_time(step):
    """A step's transfer time T, ms: from its first end to the last of its requests' kv."""
    return 1000 * (max(r["kv"] for r in step) - min(r["end"] for r in step))


def transfer_samples(path):
    """The transfer time of every prefill step of the plan's batches, in s, by its prompt lengths.

    Keyed as runner.py results load, ("transfer", lengths as JSON), so compare_fits.py fits them
    like step latencies: every step of the same prompt lengths is one batch, and its times are
    that batch's samples.
    """
    requests, _ = load(path)
    samples = defaultdict(list)
    for key, batch in sent_batches(requests).items():
        if key[0] == "batches":
            for step in steps(batch):
                lens = sorted((r["len"] for r in step), reverse=True)
                samples["transfer", json.dumps(lens)].append(transfer_time(step) / 1000)
    return samples


def percentiles(values, ps=(50, 90, 99)):
    return " / ".join(f"{np.percentile(values, p):.1f}" for p in ps)


def main(path):
    requests, failed = load(path)
    sent = sent_batches(requests)
    split = {key: steps(batch) for key, batch in sent.items()}
    print(f"# KV transfer in SGLang PD disaggregation: {path}\n")
    print(f"{len(requests):,} requests with both answers, {failed} without.\n")

    # How SGLang ran the sent batches
    print("## How the sent batches ran\n")
    print("| phase | sent batches | ran as one prefill step | prefill steps per sent batch (max) |")
    print("|---|---|---|---|")
    for phase in ("warmup", "batches", "bursts"):
        keys = [k for k in split if k[0] == phase]
        if keys:
            one = sum(len(split[k]) == 1 for k in keys)
            print(f"| {phase} | {len(keys)} | {one} ({100 * one / len(keys):.0f}%) | "
                  f"{max(len(split[k]) for k in keys)} |")
    several = [k for k in split if k[0] == "batches" and len(sent[k]) > 1]
    if several:
        alone = sum(len(split[k][0]) == 1 for k in several)
        print(f"\nOf the {len(several)} sent batches of more than one request, {alone} "
              f"({100 * alone / len(several):.0f}%) ran their first request in a step of its own.")
        gaps = [1000 * (split[k][1][0]["start"] - split[k][0][0]["start"]) for k in several if len(split[k]) > 1]
        if gaps:
            print(f"The second step started {percentiles(gaps)} ms (p50 / p90 / p99) after the first.")
    boot = [r["bootstrap_ms"] for r in requests if r["phase"] == "batches"]
    print(f"Bootstrap before prefill, ms (p50 / p90 / p99): {percentiles(boot)}")
    by_n = defaultdict(list)
    for r in requests:
        if r["phase"] == "batches":
            by_n[len(r["batch"])].append(r["bootstrap_ms"])
    ns = [n for n in sorted(by_n) if n & (n - 1) == 0]
    print("\n| requests sent together | " + " | ".join(map(str, ns)) + " |\n" + "|---" * (len(ns) + 1) + "|")
    for p in (50, 90):
        print(f"| bootstrap, p{p} (ms) | " + " | ".join(f"{np.percentile(by_n[n], p):,.0f}" for n in ns) + " |")
    print()
    # The prefill server processes a step's results one request at a time, queueing (mooncake)
    # or posting (nixl) each transfer as it goes, so a step's end times are spread out
    spread = [1000 * (step[-1]["end"] - step[0]["end"]) / (len(step) - 1)
              for k in split if k[0] == "batches" for step in split[k] if len(step) > 1]
    if spread:
        print(f"Within a prefill step, consecutive requests' ends are {percentiles(spread)} ms apart "
              "(p50 / p90 / p99 over steps).")
    print()

    # What one request's transfer carries
    print("## What one request's transfer carries\n")
    by_len = defaultdict(list)
    for r in requests:
        by_len[r["len"]].append(r["mb"])
    lens = sorted(by_len)
    X = np.array([[1.0, n, min(n, 128)] for n in lens])
    mb = np.array([np.mean(by_len[n]) for n in lens])
    w = lstsq(X, mb)
    print(f"transfer_total = {w[0]:.4f} MB + {1024 * w[1]:.3f} KB * l + {1024 * w[2]:.3f} KB * min(l, 128), "
          f"max residual {np.abs(X @ w - mb).max():.3f} MB over {len(lens)} prompt lengths\n")

    keys = sorted(k for k in sent if k[0] == "batches")

    # One request alone: what a transfer costs with nothing queued ahead of it
    print("## One request alone\n")
    alone = defaultdict(list)
    for k in keys:
        if len(sent[k]) == 1:
            alone[sent[k][0]["len"]].append(1000 * (sent[k][0]["kv"] - sent[k][0]["end"]))
    whole = [n for n in sorted(alone) if n <= BUDGET]
    powers = [n for n in whole if n & (n - 1) == 0]
    print(f"Delay, kv - end, median over the samples (ms), for prompts of up to {BUDGET} tokens:\n")
    print("| prompt tokens | " + " | ".join(f"{n:,}" for n in powers) + " |\n" + "|---" * (len(powers) + 1) + "|")
    print("| delay (ms) | " + " | ".join(f"{np.median(alone[n]):.1f}" for n in powers) + " |\n")
    others = [np.median(alone[n]) for n in whole if n not in powers]
    if others:
        print(f"The {len(others)} other lengths up to {BUDGET} tokens: {min(others):.1f} to {max(others):.1f} ms, "
              f"median {np.median(others):.1f}.\n")
    chunked = [n for n in sorted(alone) if n > BUDGET]
    if chunked:
        print("Longer prompts, prefilled in chunks of up to the budget, earlier chunks sent while later ones "
              "compute (median delay after the last chunk): "
              + ", ".join(f"{n}: {np.median(alone[n]):.1f} ms" for n in chunked) + "\n")

    # Same S, more requests
    print("## The same S split among more requests (uniform sent batches, median ms)\n")
    print("From a sent batch's first end to the last of its kv, so this holds the prefill of its later steps.\n")
    uniform = defaultdict(lambda: defaultdict(list))
    for k in keys:
        lens = [r["len"] for r in sent[k]]
        if len(set(lens)) == 1:
            uniform[sum(lens)][len(lens)].append(transfer_time(sent[k]))
    for total in sorted(uniform):
        if len(uniform[total]) >= 3:
            print(f"- S = {total}: " + ", ".join(f"N={n}: {np.median(t):,.1f}"
                                                for n, t in sorted(uniform[total].items())))
    print()

    pickup = [1000 * (r["join"] - r["kv"]) for r in requests if r["phase"] == "batches"]
    print(f"\nDecode pickup, kv to join, ms (p50 / p90 / p99): {percentiles(pickup)}\n")

    # Bursts: which of prefill and transfer set the pace
    bursts = sorted(k for k in sent if k[0] == "bursts")
    if bursts:
        print("## Bursts: prefill and transfer at once\n")
        print("Rates over a whole burst: prefill from its first start to its last end, transfer from its first "
              "end to its last kv. Medians over the samples, and the largest delay of any request in any "
              "sample.\n")
        print("| prompt | requests | prefill steps | prefill (tok/s) | transfer (tok/s) | transfer (req/s) | "
              "median delay (ms) | max delay (ms) |")
        print("|---|---|---|---|---|---|---|---|")
        by_len = defaultdict(list)
        for k in bursts:
            by_len[sent[k][0]["len"]].append(k)
        for n in sorted(by_len):
            cells = defaultdict(list)
            for k in by_len[n]:
                batch = sent[k]
                tokens = sum(r["len"] for r in batch)
                start = min(r["start"] for r in batch)
                first, last_end, last_kv = batch[0]["end"], max(r["end"] for r in batch), max(r["kv"] for r in batch)
                delays = [1000 * (r["kv"] - r["end"]) for r in batch]
                cells["steps"].append(len(split[k]))
                cells["prefill"].append(tokens / (last_end - start))
                cells["transfer"].append(tokens / (last_kv - first))
                cells["req/s"].append(len(batch) / (last_kv - first))
                cells["delay"].append(np.median(delays))
                cells["max"].append(max(delays))
            m = {key: np.median(v) for key, v in cells.items()}
            print(f"| {n} | {len(sent[by_len[n][0]])} | {m['steps']:.0f} | {m['prefill']:,.0f} | "
                  f"{m['transfer']:,.0f} | {m['req/s']:,.1f} | {m['delay']:,.1f} | {max(cells['max']):,.0f} |")

        # With overlap scheduling, a step's transfers are posted once the next step has started
        print(f"\nTransfers posted while the burst's next prefill step computed (end between that step's start "
              f"and its first end), by size, and how many took over {STALL_MS} ms:\n")
        print("| transfer (MB) | posted during the next step | over 100 ms | median delay (ms) |\n|---|---|---|---|")
        during = defaultdict(list)
        for k in bursts:
            for step, after in zip(split[k], split[k][1:]):
                for r in step:
                    if after[0]["start"] < r["end"] < min(q["end"] for q in after):
                        size = next(f"{lo}-{hi}" for lo, hi in SIZES if lo <= r["mb"] < hi)
                        during[size].append(1000 * (r["kv"] - r["end"]))
        for lo, hi in SIZES:
            delays = np.array(during.get(f"{lo}-{hi}", []))
            if len(delays):
                over = (delays > STALL_MS).sum()
                print(f"| {lo}-{hi} | {len(delays)} | {over} ({100 * over / len(delays):.0f}%) | "
                      f"{np.median(delays):.1f} |")


if __name__ == "__main__":
    main(sys.argv[1])
