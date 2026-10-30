r"""Check that a system runs SGLang with the setup the latency models were fitted on.

Pass the command that starts the SGLang server on that system, as one string:

    python sanity_latency.py "TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
        PYTHONPATH=/path/to/fairinf-sglang/python:/path/to/fairinf-latency-sweep/vm \
        python3 -m sglang.launch_server --model-path lmsys/gpt-oss-20b-bf16 --dtype bfloat16 \
        --mem-fraction-static 0.7 --sampling-backend pytorch --piecewise-cuda-graph-compiler inductor \
        --scheduling-policy-path noop_cache_policy.NoOpWithCachePolicy --port 30000"

The command then starts src/runner.py instead of the server, with the same
environment, Python and SGLang flags, and times the batches of
src/sanity_batches.json. Then this prints how the run differs from the
reference: the same batches timed on eu1 with that command's setup, in
results/gptoss-sanity-cuda-graphs-20261029 (see its README):
- the setup lines of the two run logs that differ: package versions, GPU, MoE
  kernel, KV cache and CUDA-graph sizes;
- each batch's median time in both runs. A batch is marked with ! when the two
  differ by more than 5% and by more than twice the larger of the two runs'
  sample spreads.

The run matches if the setup lines do and no phase's times differ. A phase
differs if its batches are more than 5% slower or faster on average, or if more
than 10% of them are marked: a setup difference moves many batches, while a
single small step can be off by chance (one 10 ms decode step was 9% slow in
an identical rerun). Runs of the same setup on eu1 differ by at most 1.2% on
average in any phase, while prefill without CUDA graphs is 53% slower.

About the command:
- It may start with environment assignments or nohup, and end in a redirect
  or &. The nohup, the redirect and the & are dropped. Shell variables in it
  must be set.
- It starts SGLang with `python -m sglang.launch_server` or `sglang serve`.
- Flags that only matter to a server, such as --port or the scheduling
  policy, are accepted and change nothing: the check covers what sets the
  time of one forward pass.
- It can't use --config, which only the server reads, or tensor parallelism:
  the reference ran on one GPU.

The GPU has to be free. The run's results.jsonl and run.log go to --out, by
default a new directory under results/. Exits with 0 if the run matches the
reference, 1 if it differs, and 2 if it could not run.

This script needs numpy. SGLang only has to be in the command's environment.
"""

import argparse
import collections
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src")
sys.path.insert(0, SRC)

from compare_runs import SETUP, compare, label, setup, setup_changes  # noqa: E402
from server_command import LAUNCHER, foreground, gpu_processes, swap  # noqa: E402

# The run every check compares with: src/sanity_batches.json on eu1, with piecewise CUDA graphs
REFERENCE = os.path.join(HERE, "results", "gptoss-sanity-cuda-graphs-20261029", "results.jsonl")
PHASES = ("prefill", "decode", "hybrid")

# A phase differs from the reference if its batches are this much slower or faster on
# average (%), or if more than this share of them differ by more than noise. 5% leaves
# room for drift: same-setup runs on eu1 differ by up to 1.2%, and a 2-3% shift passes.
MAX_MEAN_SHIFT = 5.0
MAX_SHARE_BEYOND = 0.10

# Parts of a command that runner.py can't take
CONFIG = re.compile(r"--config(?:=|\s|$)")
TP_SIZE = re.compile(r"--(?:tp|tp-size|tensor-parallel-size)(?:=|\s+)(\d+)")

# runner.py's progress lines, and other lines worth showing as they come
SAMPLE = re.compile(r"^(prefill|decode|hybrid) (\d+)/(\d+) sample (\d+)/(\d+): ")
ERROR = re.compile(r"Traceback|Error|Killed")


def fail(message):
    print(message, file=sys.stderr)
    sys.exit(2)


class Console:
    """Shows a run's setup lines and errors as they come, and its timing progress on one line."""

    def __init__(self):
        self.progress = False  # a progress line is waiting for its newline
        self.traceback = False
        self.tail = collections.deque(maxlen=15)

    def __call__(self, line):
        self.tail.append(line)
        sample = SAMPLE.match(line)
        if sample:
            phase, i, n = sample.group(1, 2, 3)
            if sys.stdout.isatty():
                print(f"\r  timing {phase} batch {i}/{n}".ljust(40), end="", flush=True)
                self.progress = True
            elif i == n and sample[4] == sample[5]:
                print(f"  timed {n} {phase} batches")
            return
        self.traceback = self.traceback or line.startswith("Traceback")
        if self.traceback or SETUP.search(line) or ERROR.search(line):
            self.end_progress()
            print(f"  {line}")

    def end_progress(self):
        if self.progress:
            print()
            self.progress = False


def run(command, log, show):
    """Runs command in bash, writing its output to log and passing each line to show.

    Returns its exit code.
    """
    env = dict(os.environ, PYTHONUNBUFFERED="1")  # progress lines as they happen, not in blocks
    with subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, env=env) as proc:
        for raw in proc.stdout:
            log.write(raw)
            log.flush()
            show(raw.decode(errors="replace").rstrip("\r\n").rsplit("\r", 1)[-1])
        return proc.wait()


def report(reference, results, planned):
    """Prints how the run in results differs from the reference. Returns 0 if it matches, else 1."""
    print(f"\nCompared with {os.path.relpath(reference, HERE)}:")
    changed = setup_changes(reference, results)
    if changed is None:
        print("\nSetup: not compared, as one of the runs has no run.log next to its results.")
    elif changed:
        print("\nSetup lines that differ (- reference, + this run):")
        print("\n".join(f"  {line}" for line in changed))
    else:
        print(f"\nSetup: all {len(setup(results))} setup lines match.")

    phases, unmatched = compare(reference, results)
    if not phases:
        print("\nNone of this run's batches are in the reference, so there is nothing to compare.")
        return 1
    print("\nStep times: the median of each batch's samples, in ms. Noise is the larger of the two\n"
          "runs' sample spreads, and ! marks a difference of more than both 5% and twice the noise.")
    summary, ran, shifted = [], sum(unmatched.values()), []
    for phase, r in phases.items():
        diff = 100 * (r["now"] / r["then"] - 1)
        noise = np.maximum(r["then_spread"], r["now_spread"])
        beyond = np.abs(diff) > np.maximum(5, 2 * noise)
        title = "hybrid (prompts / decodes)" if phase == "hybrid" else phase
        print(f"\n  {title:<26}{'reference':>10}{'this run':>10}{'diff':>9}{'noise':>8}")
        for batch, then, now, d, n, b in zip(r["batches"], r["then"], r["now"], diff, noise, beyond):
            name = label(phase, batch)
            name = name if len(name) <= 26 else name[:23] + "..."
            print(f"  {name:<26}{then:>10.2f}{now:>10.2f}{d:>+8.1f}%{n:>7.1f}%{'  !' if b else ''}")
        # One batch can be off by chance; a setup difference moves the phase as a whole
        differs = abs(diff.mean()) > MAX_MEAN_SHIFT or beyond.mean() > MAX_SHARE_BEYOND
        summary.append(f"  {phase:<9}{len(diff):>8}{diff.mean():>+10.1f}%{np.abs(diff).mean():>9.1f}%"
                       f"{np.abs(diff).max():>8.1f}%{beyond.sum():>14}  {'DIFFERS' if differs else 'ok'}")
        if differs:
            shifted.append(f"{phase} times differ (mean {diff.mean():+.1f}%, {beyond.sum()} of {len(diff)} "
                           f"batches beyond noise)")
        ran += len(diff)
    for phase, count in unmatched.items():
        print(f"\n  {count} {phase} batches are not in the reference and are left out.")

    print("\nBy phase:\n\n"
          f"  {'phase':<9}{'batches':>8}{'mean diff':>11}{'mean abs':>10}{'max abs':>9}{'beyond noise':>14}")
    print("\n".join(summary))

    problems = ["setup lines differ"] if changed else []
    problems += shifted
    if ran < planned:
        problems.append(f"{planned - ran} of {planned} batches did not run")
    if problems:
        print(f"\nDIFFERS from the reference: {'; '.join(problems)}.")
        return 1
    if changed is None:
        print("\nMATCHES the reference in step times. The setup was not compared.")
    else:
        print("\nMATCHES the reference: the setup lines are the same, and so are the step times.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", help="the command that starts the SGLang server on this system, in quotes")
    parser.add_argument("--batches", default=os.path.join(SRC, "sanity_batches.json"), help="the batch plan to time")
    parser.add_argument("--out", help="a new directory for this run's results.jsonl and run.log")
    parser.add_argument("--force", action="store_true", help="run even if other processes are using a GPU")
    parser.add_argument("--dry-run", action="store_true", help="print the command that would run, and stop")
    args = parser.parse_args()

    command = foreground(args.command)
    if len(LAUNCHER.findall(command)) != 1:
        fail("The command has to start SGLang once, with `python -m sglang.launch_server` or `sglang serve`.")
    if CONFIG.search(command):
        fail("runner.py can't read --config files; put their flags in the command instead.")
    if any(int(size) != 1 for size in TP_SIZE.findall(command)):
        fail("The latency models are for one GPU, and this command uses tensor parallelism.")

    host, now = socket.gethostname(), time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out = os.path.abspath(args.out or os.path.join(HERE, "results", f"sanity-latency-{host}-{now}"))
    results, log_path = os.path.join(out, "results.jsonl"), os.path.join(out, "run.log")
    batches = os.path.abspath(args.batches)
    runner = swap(command, f"{shlex.quote(os.path.join(SRC, 'runner.py'))} "
                           f"--batches {shlex.quote(batches)} --out {shlex.quote(results)}")
    if args.dry_run:
        print(f"This would print package versions with the same command, then run\n\n{runner}\n\n"
              f"logging to {log_path}.")
        return 0

    if not os.path.exists(REFERENCE):
        fail(f"The reference run, {os.path.relpath(REFERENCE, HERE)}, is missing.")
    busy = gpu_processes()
    if busy and not args.force:
        fail("Other processes are using a GPU, which would skew the times:\n  " + "\n  ".join(busy)
             + "\nStop them, or pass --force.")
    if os.path.exists(results):
        fail(f"{results} already exists; pass a new --out.")
    os.makedirs(out, exist_ok=True)

    with open(batches) as f:
        plan = json.load(f)
    planned = sum(len(plan.get(phase, [])) for phase in PHASES)
    print(f"Timing {planned} batches in place of the server:\n\n{runner}\n\nlogging to {log_path}.\n")
    start, show = time.time(), Console()
    with open(log_path, "wb") as log:
        run(swap(command, shlex.quote(os.path.join(SRC, "versions.py"))), log, show)  # the setup lines
        code = run(runner, log, show)
    show.end_progress()
    minutes, seconds = divmod(round(time.time() - start), 60)
    if code:
        if not show.traceback:
            print("\n".join(f"  {line}" for line in show.tail))
        print(f"\nrunner.py FAILED with exit code {code} after {minutes}m{seconds:02d}s.")
    else:
        print(f"\nrunner.py finished in {minutes}m{seconds:02d}s.")
    if not os.path.exists(results) or os.path.getsize(results) == 0:
        fail(f"No samples to compare; see {log_path}.")
    return max(report(REFERENCE, results, planned), 1 if code else 0)


if __name__ == "__main__":
    sys.exit(main())
