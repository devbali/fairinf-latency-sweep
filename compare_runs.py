"""Compare a run of runner.py with a reference run, batch by batch.

This checks that a setup (VM, packages, SGLang flags) gives the step times the
latency models were fitted on: run sanity_batches.py there with vm/run_clean.sh,
then compare with results/gptoss-clean-20261028. For every batch in both runs
it prints the two median latencies and how far apart they are, and what
fit.py's model, fitted on the reference, predicts. If each run has a run.log
next to its results, the setup lines in them are compared too: package
versions, GPU, MoE kernel, KV cache and CUDA graph sizes.

Prints markdown. Usage: python compare_runs.py REFERENCE.jsonl NEW.jsonl
"""

import difflib
import itertools
import json
import os
import re
import sys

import numpy as np

from fit import features, fit, load, predict

# run.log lines that describe the setup, and the parts of them that vary from run to run
SETUP = re.compile(
    r"^(sglang|sgl_kernel|flashinfer|transformers|torch|gpu) |MOE kernel|memory pool|"
    r"KV Cache is allocated|Capture cuda graph bs|piecewise CUDA graph|Load weight end"
)
VARIES = re.compile(r"(elapsed|avail mem)=[\d.]+ \w+, ")


def setup(results):
    """The setup lines of the run.log next to a results file, or None if there is none."""
    log = os.path.join(os.path.dirname(results), "run.log")
    if not os.path.exists(log):
        return None
    return [VARIES.sub("", line.rstrip()) for line in open(log) if SETUP.search(line)]


def describe(lens):
    """[6144, 16, 16, 16] -> '6144 + 3x16'."""
    runs = [(length, len(list(group))) for length, group in itertools.groupby(lens)]
    return " + ".join(f"{n}x{length}" if n > 1 else str(length) for length, n in runs)


def design(phase, batches):
    return np.array([[1.0] + list(features(phase, batch).values()) for batch in batches])


def medians(samples, keys):
    return np.array([np.median(samples[key]) for key in keys]) * 1e3  # ms


def spreads(samples, keys):
    return np.array([100 * np.ptp(samples[key]) / np.median(samples[key]) for key in keys])


reference_path, new_path = sys.argv[1:3]
reference, new = load([reference_path]), load([new_path])
print(f"# {new_path} against {reference_path}")

reference_setup, new_setup = setup(reference_path), setup(new_path)
if reference_setup is not None and new_setup is not None:
    changed = [line for line in difflib.ndiff(reference_setup, new_setup) if line[:2] in ("- ", "+ ")]
    if changed:
        print("\nSetup lines in the run logs that differ (- reference, + this run):\n\n```diff")
        print("\n".join(changed))
    else:
        print("\nEvery setup line in the run logs matches the reference:\n\n```")
        print("\n".join(new_setup))
    print("```")

summary, tables = [], []
for phase in ("prefill", "decode", "hybrid"):
    keys = [key for key in new if key[0] == phase and key in reference]
    unmatched = sum(1 for key in new if key[0] == phase and key not in reference)
    if unmatched:
        print(f"\n{unmatched} {phase} batches of this run are not in the reference and are left out.")
    if not keys:
        continue
    # fit.py's model, fitted on every batch of the reference, in fit.py's order
    reference_keys = sorted(key for key in reference if key[0] == phase)
    model = fit(phase, design(phase, [json.loads(b) for _, b in reference_keys]),
                medians(reference, reference_keys))

    batches = [json.loads(b) for _, b in keys]
    then, now = medians(reference, keys), medians(new, keys)
    pred = predict(model, design(phase, batches))
    diff, error = 100 * (now / then - 1), 100 * (pred / now - 1)
    summary.append(
        f"| {phase} | {len(keys)} | {diff.mean():+.1f}% | {np.abs(diff).mean():.1f}% | "
        f"{np.abs(diff).max():.1f}% | {np.median(spreads(reference, keys)):.1f}% / "
        f"{np.median(spreads(new, keys)):.1f}% | {np.abs(100 * (pred / then - 1)).mean():.1f}% / "
        f"{np.abs(error).mean():.1f}% |"
    )

    label = "prompts / decodes" if phase == "hybrid" else "lengths"
    rows = [f"\n### {phase}\n\n| batch ({label}) | reference | this run | difference | model | "
            "model error |\n" + "|---" * 6 + "|"]
    for batch, t0, t1, d, p, e in zip(batches, then, now, diff, pred, error):
        name = f"{describe(batch[0])} / {describe(batch[1])}" if phase == "hybrid" else describe(batch)
        rows.append(f"| {name} | {t0:.2f} | {t1:.2f} | {d:+.1f}% | {p:.2f} | {e:+.1f}% |")
    tables.append("\n".join(rows))

print(
    "\nEach batch's latency is the median of its samples, in ms. The difference is this run's "
    "time against the reference's. The sample spread is how far one batch's samples spread within "
    "a run (max - min over the median), so differences near it are measurement noise. The model is "
    "fit.py's, fitted on the reference, and its error is relative to the measured time, as in "
    "fit.py.\n\n"
    "| phase | batches | mean difference | mean abs difference | max abs difference | "
    "sample spread, median: reference / this run | model error, mean abs: reference / this run |\n"
    + "|---" * 7 + "|"
)
print("\n".join(summary))
print("\n".join(tables))
