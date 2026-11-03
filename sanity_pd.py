r"""Check that a system runs SGLang's PD disaggregation with the setup the PD transfer model was fitted on.

Pass the commands that start the prefill server and the decode server on that
system, one string each, as src/pd_runner.py takes them:

    python sanity_pd.py "PREFILL SERVER COMMAND" "DECODE SERVER COMMAND"

Each starts SGLang with --disaggregation-mode prefill or decode. This runs
pd_runner.py with both on the short plan src/pd_sanity_batches.json: 20
batches and a burst, each sent 3 times. Then it prints how the run differs from
the reference, the same plan with the setup the model was fitted on:
gpt-oss-20b on two A100s of one VM joined by NVLink, with NIXL moving the KV
cache (results/gptoss-pd-sanity-nixl-20261103, see its README):
- the setup lines that differ: package versions, the transfer packages and
  backend, the GPUs and how they are linked, RDMA devices, and both servers' KV
  cache and CUDA-graph sizes;
- the KV cache one prompt of each length carries, which depends only on the
  model and how its KV cache is laid out, so it has to be the same;
- each batch's transfer delay in both runs: from a request's prefill result to
  its KV cache reaching the decode server, the mean over the batch's requests,
  median of its samples. A batch is marked with ! when the two differ by more
  than 20% and by more than twice the larger of the two runs' sample spreads;
- for the burst, the rates of prefill and of the transfer.

The run matches if the setup lines and the KV sizes do, every request got both
answers, the batches' delays are no more than 15% slower or faster on average
with at most 20% of them marked, and the transfer keeps up with prefill in each
burst where it did in the reference. SGLang splits most batches into two
prefill steps at a point that varies, so these delays are noisier than step
times: one batch's samples spread by 6% (median). Three runs of the reference's
setup on the same VM differ by up to 8% on average: the reference itself, the
first after another job, came out 7% slower than the other two, which agree to
1.3%. The fork's default setup on the same VM, mooncake over TCP, is 1.2 to 30
times slower on these batches.

Also printed, with no bearing on the verdict: the bytes that went over NVLink,
the bootstrap before prefill, and how soon the decode server picks a request up.

The GPUs have to be free. The run's files go to --out, by default a new
directory under results/. Exits with 0 if the run matches the reference, 1 if it
differs, and 2 if it could not run.

This script needs numpy. SGLang only has to be in the commands' environment.
"""

import argparse
import difflib
import gzip
import json
import os
import re
import socket
import subprocess
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
sys.path.insert(0, SRC)

from compare_runs import SETUP, VARIES, describe  # noqa: E402
from pd_fits import load, sent_batches  # noqa: E402

# The run every check compares with: src/pd_sanity_batches.json on the 4xA100 VM, NIXL over NVLink
REFERENCE = os.path.join(HERE, "results", "gptoss-pd-sanity-nixl-20261103")

# The batches' delays differ from the reference's if they are this much slower or faster on
# average (%), or if more than this share of them are marked: off by more than BEYOND % and
# twice the noise. 15% is about twice the largest shift between runs of the same setup (7.6%).
# A prompt's KV cache size has to match within MAX_KV_DIFF %.
MAX_MEAN_SHIFT = 15.0
MAX_SHARE_BEYOND = 0.20
BEYOND = 20.0
MAX_KV_DIFF = 1.0

# Server log lines that name the transfer backend, besides the ones compare_runs.py reads
TRANSFER = re.compile(r"KVManager initialized with backend|Backend \w+ was instantiated")
TIMESTAMP = re.compile(r"^\[?\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\]? ")
# A server's KV cache can come out a token larger or smaller from one start to the next
KV_TOKENS = re.compile(r"(#tokens: |_tokens=)(\d+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
NVLINK_TX = re.compile(r"Link \d+: Data Tx: (\d+) KiB")


def fail(message):
    print(message, file=sys.stderr)
    sys.exit(2)


def open_log(run, name):
    """A server log of a run, plain or gzipped."""
    path = os.path.join(run, f"{name}.log")
    return open(path, errors="replace") if os.path.exists(path) else gzip.open(path + ".gz", "rt", errors="replace")


def results_path(run):
    """A run's results.jsonl, or its gzipped copy."""
    path = os.path.join(run, "results.jsonl")
    return path if os.path.exists(path) or not os.path.exists(path + ".gz") else path + ".gz"


def blocks(run):
    """run.log's "== title" blocks, as {title: lines}."""
    found, title = defaultdict(list), None
    for line in open(os.path.join(run, "run.log"), errors="replace"):
        line = ANSI.sub("", line.rstrip())
        if line.startswith("== "):
            title = line[3:]
        elif title:
            found[title].append(line)
    return found


def setup(run):
    """The setup lines of a pd_runner.py run: from run.log, then from both servers' logs."""
    b = blocks(run)
    lines = b["versions"] + b["transfer packages"]
    lines.append("rdma devices: " + (" ".join(line.strip() for line in b["rdma devices"] if line.strip()) or "none"))
    for line in b["gpus"][1:]:  # without the PCI bus id, which says where a GPU sits, not what it is
        fields = [field.strip() for field in line.split(",")]
        lines.append("gpu " + ", ".join(fields[:2] + fields[3:]))
    links = [line.split() for line in b["gpu links"] if line.startswith("GPU")]
    lines += ["link " + " ".join(row[:len(links) + 1]) for row in links]  # GPU to GPU, not CPU affinity
    for name in ("prefill", "decode"):
        with open_log(run, name) as log:
            for line in log:
                if SETUP.search(line) or TRANSFER.search(line):
                    line = VARIES.sub("", TIMESTAMP.sub("", line.rstrip()))
                    line = KV_TOKENS.sub(lambda m: f"{m[1]}about {round(int(m[2]), -3):,}", line)
                    lines.append(f"{name}: {line}")
    return lines


def nvlink_gib(run):
    """GiB the GPUs sent over NVLink during the run, from run.log's counters; None without them."""
    b = blocks(run)
    before, after = ([int(kib) for line in b[title] for kib in NVLINK_TX.findall(line)]
                     for title in ("nvlink data counters before the run", "nvlink data counters after the run"))
    if not before or len(before) != len(after):
        return None
    return (sum(after) - sum(before)) / 1024**2


def summarize(run):
    """What the check compares, from a run's results.jsonl."""
    requests, failed = load(results_path(run))
    sent = sent_batches(requests)
    delays, kv_mb, bursts = defaultdict(list), defaultdict(list), defaultdict(list)
    for (phase, _, _), batch in sent.items():
        key = json.dumps(batch[0]["batch"])
        if phase == "batches":
            delays[key].append(np.mean([1000 * (r["kv"] - r["end"]) for r in batch]))
            for r in batch:
                kv_mb[r["len"]].append(r["mb"])
        elif phase == "bursts":
            tokens = sum(r["len"] for r in batch)
            prefill = tokens / (max(r["end"] for r in batch) - min(r["start"] for r in batch))
            transfer = tokens / (max(r["kv"] for r in batch) - min(r["end"] for r in batch))
            bursts[key].append((prefill, transfer))
    batches = [r for r in requests if r["phase"] == "batches"]
    return {
        "requests": len(requests),
        "failed": failed,
        "delays": delays,
        "kv_mb": {n: np.median(mb) for n, mb in kv_mb.items()},
        "bursts": {key: np.median(rates, axis=0) for key, rates in bursts.items()},
        "kv_gib": sum(r["mb"] for r in requests) / 1024,
        "nvlink_gib": nvlink_gib(run),
        "bootstrap_ms": np.median([r["bootstrap_ms"] for r in batches]) if batches else None,
        "pickup_ms": np.median([1000 * (r["join"] - r["kv"]) for r in batches]) if batches else None,
    }


def short(batch):
    name = describe(json.loads(batch))
    return name if len(name) <= 26 else name[:23] + "..."


def report(reference, run, planned):
    """Prints how the run differs from the reference. Returns 0 if it matches, else 1."""
    print(f"\nCompared with {os.path.relpath(reference, HERE)}:")
    problems = []
    try:
        changed = [line for line in difflib.ndiff(setup(reference), setup(run)) if line[:2] in ("- ", "+ ")]
    except OSError as error:
        changed = None
        print(f"\nSetup: not compared, as a log is missing ({error}).")
    if changed:
        print("\nSetup lines that differ (- reference, + this run):")
        print("\n".join(f"  {line}" for line in changed))
        problems.append("setup lines differ")
    elif changed is not None:
        print(f"\nSetup: all {len(setup(run))} setup lines match.")

    then, now = summarize(reference), summarize(run)
    if now["failed"]:
        problems.append(f"{now['failed']} of {now['failed'] + now['requests']} requests did not get both answers")

    lengths = sorted(set(then["kv_mb"]) & set(now["kv_mb"]))
    kv_diff = {n: 100 * (now["kv_mb"][n] / then["kv_mb"][n] - 1) for n in lengths}
    print("\nKV cache one prompt carries, MB:\n\n"
          f"  {'prompt tokens':<16}{'reference':>12}{'this run':>12}{'diff':>9}")
    for n in lengths:
        mark = "  !" if abs(kv_diff[n]) > MAX_KV_DIFF else ""
        print(f"  {n:<16,}{then['kv_mb'][n]:>12.3f}{now['kv_mb'][n]:>12.3f}{kv_diff[n]:>+8.1f}%{mark}")
    if any(abs(d) > MAX_KV_DIFF for d in kv_diff.values()):
        problems.append("a prompt's KV cache is a different size")

    keys = [key for key in now["delays"] if key in then["delays"]]
    if not keys:
        print("\nNone of this run's batches are in the reference, so there is nothing to compare.")
        return 1
    t0 = np.array([np.median(then["delays"][key]) for key in keys])
    t1 = np.array([np.median(now["delays"][key]) for key in keys])
    diff = 100 * (t1 / t0 - 1)
    noise = np.maximum(*[[100 * np.ptp(run_delays[key]) / np.median(run_delays[key]) for key in keys]
                         for run_delays in (then["delays"], now["delays"])])
    beyond = np.abs(diff) > np.maximum(BEYOND, 2 * noise)
    print(f"\nTransfer delay: from a request's prefill result to its KV cache on the decode server, the mean\n"
          f"over a batch's requests, median of its samples, in ms. Noise is the larger of the two runs' sample\n"
          f"spreads, and ! marks a difference of more than both {BEYOND:.0f}% and twice the noise.\n\n"
          f"  {'batch (prompt tokens)':<26}{'reference':>10}{'this run':>10}{'diff':>9}{'noise':>8}")
    for key, a, b, d, n, mark in zip(keys, t0, t1, diff, noise, beyond):
        print(f"  {short(key):<26}{a:>10.1f}{b:>10.1f}{d:>+8.1f}%{n:>7.1f}%{'  !' if mark else ''}")
    differs = abs(diff.mean()) > MAX_MEAN_SHIFT or beyond.mean() > MAX_SHARE_BEYOND
    print(f"\n  {len(keys)} batches: mean diff {diff.mean():+.1f}%, mean abs {np.abs(diff).mean():.1f}%, "
          f"max abs {np.abs(diff).max():.1f}%, {beyond.sum()} beyond noise  {'DIFFERS' if differs else 'ok'}")
    if differs:
        problems.append(f"transfer delays differ (mean {diff.mean():+.1f}%, {beyond.sum()} of {len(keys)} "
                        f"batches beyond noise)")
    if len(keys) < planned:
        problems.append(f"{planned - len(keys)} of {planned} batches did not run")

    for key in sorted(set(then["bursts"]) - set(now["bursts"])):
        problems.append(f"the burst of {describe(json.loads(key))} did not run")
    for key in sorted(set(then["bursts"]) & set(now["bursts"])):
        print(f"\nBurst of {describe(json.loads(key))}, median of its samples, in tokens/s:\n\n"
              f"  {'':<12}{'prefill':>10}{'transfer':>10}  transfer kept up")
        kept = {}
        for name, summary in (("reference", then), ("this run", now)):
            prefill, transfer = summary["bursts"][key]
            kept[name] = transfer >= prefill
            print(f"  {name:<12}{prefill:>10,.0f}{transfer:>10,.0f}  {'yes' if kept[name] else 'NO'}")
        if kept["reference"] and not kept["this run"]:
            problems.append(f"the transfer did not keep up with prefill in the burst of {describe(json.loads(key))}")

    print("\nNot part of the verdict:\n\n"
          f"  {'':<34}{'reference':>10}{'this run':>10}")
    for title, field, unit in (("KV cache sent, GiB", "kv_gib", ".2f"), ("sent over NVLink, GiB", "nvlink_gib", ".2f"),
                               ("bootstrap before prefill, ms", "bootstrap_ms", ".1f"),
                               ("decode pickup, ms", "pickup_ms", ".2f")):
        cells = [f"{s[field]:>10{unit}}" if s[field] is not None else f"{'-':>10}" for s in (then, now)]
        print(f"  {title:<34}{''.join(cells)}")

    if problems:
        print(f"\nDIFFERS from the reference: {'; '.join(problems)}.")
        return 1
    if changed is None:
        print("\nMATCHES the reference in KV sizes and transfer times. The setup was not compared.")
    else:
        print("\nMATCHES the reference: the setup lines, the KV cache sizes and the transfer times.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prefill", help="the command that starts the prefill server on this system, in quotes")
    parser.add_argument("decode", help="the command that starts the decode server on this system, in quotes")
    parser.add_argument("--plan", default=os.path.join(SRC, "pd_sanity_batches.json"), help="the plan to send")
    parser.add_argument("--out", help="a new directory for this run's results and logs")
    parser.add_argument("--force", action="store_true", help="run even if other processes are using a GPU")
    parser.add_argument("--dry-run", action="store_true", help="print the commands that would run, and stop")
    args = parser.parse_args()

    host, now = socket.gethostname(), time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out = os.path.abspath(args.out or os.path.join(HERE, "results", f"sanity-pd-{host}-{now}"))
    runner = [sys.executable, os.path.join(SRC, "pd_runner.py"), "--prefill", args.prefill, "--decode", args.decode,
              "--plan", os.path.abspath(args.plan), "--out", out]
    runner += ["--force"] * args.force + ["--dry-run"] * args.dry_run
    if not args.dry_run and not os.path.exists(results_path(REFERENCE)):
        fail(f"The reference run, {os.path.relpath(REFERENCE, HERE)}, is missing.")
    with open(args.plan) as f:
        planned = len(json.load(f).get("batches", []))

    start = time.time()
    code = subprocess.run(runner).returncode
    if args.dry_run:
        return code
    minutes, seconds = divmod(round(time.time() - start), 60)
    print(f"\npd_runner.py exited with code {code} after {minutes}m{seconds:02d}s.")
    if not os.path.exists(os.path.join(out, "results.jsonl")):
        fail(f"No requests to compare; see {out}.")
    return max(report(REFERENCE, out, planned), 1 if code else 0)


if __name__ == "__main__":
    sys.exit(main())
