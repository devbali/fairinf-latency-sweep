"""Compare a run of runner.py with a reference run, batch by batch.

This checks that a setup (VM, packages, SGLang flags) gives the step times the
latency models were fitted on: run sanity_batches.json there with vm/run_clean.sh,
then compare with results/gptoss-clean-20261028. For every batch in both runs
it prints the two median latencies and how far apart they are. If each run has
a run.log next to its results, the setup lines in them are compared too:
package versions, GPU, MoE kernel, KV cache and CUDA graph sizes.

Prints markdown. Usage: python src/compare_runs.py REFERENCE.jsonl NEW.jsonl
"""

import difflib
import itertools
import json
import os
import re
import sys

import numpy as np

from fit import load

# run.log lines that describe the setup, and the parts of them that vary from run to run.
# Piecewise CUDA graphs show as the line that disables them or the token counts they capture.
SETUP = re.compile(
    r"^(sglang|sgl_kernel|flashinfer|transformers|torch|gpu) |MOE kernel|memory pool|"
    r"KV Cache is allocated|Capture cuda graph bs|Capture cuda graph num tokens|"
    r"Disable piecewise CUDA graph|Load weight end"
)
VARIES = re.compile(r"(elapsed|avail mem)=[\d.]+ \w+, ")


def setup(results):
    """The setup lines of the run.log next to a results file, or None if there is none."""
    log = os.path.join(os.path.dirname(results), "run.log")
    if not os.path.exists(log):
        return None
    return [VARIES.sub("", line.rstrip()) for line in open(log) if SETUP.search(line)]


def setup_changes(reference_path, new_path):
    """The setup lines only the reference's log has ("- ...") or only the new run's ("+ ...").

    None if either run has no run.log next to its results.
    """
    reference_setup, new_setup = setup(reference_path), setup(new_path)
    if reference_setup is None or new_setup is None:
        return None
    return [line for line in difflib.ndiff(reference_setup, new_setup) if line[:2] in ("- ", "+ ")]


def describe(lens):
    """[6144, 16, 16, 16] -> '6144 + 3x16'."""
    runs = [(length, len(list(group))) for length, group in itertools.groupby(lens)]
    return " + ".join(f"{n}x{length}" if n > 1 else str(length) for length, n in runs)


def label(phase, batch):
    """A batch in short, as 'prompts / decodes' for a hybrid one."""
    return f"{describe(batch[0])} / {describe(batch[1])}" if phase == "hybrid" else describe(batch)


def medians(samples, keys):
    return np.array([np.median(samples[key]) for key in keys]) * 1e3  # ms


def spreads(samples, keys):
    return np.array([100 * np.ptp(samples[key]) / np.median(samples[key]) for key in keys])


def compare(reference_path, new_path):
    """Line up the batches of a new run with the same batches in a reference run.

    Returns (phases, unmatched). phases maps each phase with batches in both
    runs to arrays over those batches: "then" and "now", the median latency in
    the reference and in the new run (ms), and "then_spread" and "now_spread",
    how far each run's samples spread (% of the median). "batches" lists the
    batches themselves. unmatched counts, by phase, the new run's batches that
    the reference lacks.
    """
    reference, new = load([reference_path]), load([new_path])
    phases, unmatched = {}, {}
    for phase in ("prefill", "decode", "hybrid"):
        keys = [key for key in new if key[0] == phase and key in reference]
        missing = sum(1 for key in new if key[0] == phase and key not in reference)
        if missing:
            unmatched[phase] = missing
        if not keys:
            continue
        phases[phase] = {
            "batches": [json.loads(b) for _, b in keys],
            "then": medians(reference, keys),
            "now": medians(new, keys),
            "then_spread": spreads(reference, keys),
            "now_spread": spreads(new, keys),
        }
    return phases, unmatched


def main(reference_path, new_path):
    phases, unmatched = compare(reference_path, new_path)
    print(f"# {new_path} against {reference_path}")

    changed = setup_changes(reference_path, new_path)
    if changed:
        print("\nSetup lines in the run logs that differ (- reference, + this run):\n\n```diff")
        print("\n".join(changed))
        print("```")
    elif changed is not None:
        print("\nEvery setup line in the run logs matches the reference:\n\n```")
        print("\n".join(setup(new_path)))
        print("```")
    for phase, count in unmatched.items():
        print(f"\n{count} {phase} batches of this run are not in the reference and are left out.")

    summary, tables = [], []
    for phase, r in phases.items():
        diff = 100 * (r["now"] / r["then"] - 1)
        summary.append(
            f"| {phase} | {len(diff)} | {diff.mean():+.1f}% | {np.abs(diff).mean():.1f}% | "
            f"{np.abs(diff).max():.1f}% | {np.median(r['then_spread']):.1f}% / {np.median(r['now_spread']):.1f}% |"
        )
        kind = "prompts / decodes" if phase == "hybrid" else "lengths"
        rows = [f"\n### {phase}\n\n| batch ({kind}) | reference | this run | difference |\n" + "|---" * 4 + "|"]
        for batch, t0, t1, d in zip(r["batches"], r["then"], r["now"], diff):
            rows.append(f"| {label(phase, batch)} | {t0:.2f} | {t1:.2f} | {d:+.1f}% |")
        tables.append("\n".join(rows))

    print(
        "\nEach batch's latency is the median of its samples, in ms. The difference is this run's "
        "time against the reference's. The sample spread is how far one batch's samples spread within "
        "a run (max - min over the median), so differences near it are measurement noise.\n\n"
        "| phase | batches | mean difference | mean abs difference | max abs difference | "
        "sample spread, median: reference / this run |\n"
        + "|---" * 6 + "|"
    )
    print("\n".join(summary))
    print("\n".join(tables))


if __name__ == "__main__":
    main(*sys.argv[1:3])
