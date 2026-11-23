"""One prefill step of gpt-oss-120b with tensor parallelism over four A100 80GB.

Degree 2 + sum l^2 (see steps.py), fitted on results/gptoss-120b-tp4-20261103.
Input: the prompt lengths, e.g. [512, 1024]. Output: ms.

Usage: python -m estimators.prefill_tp4   (from src/) prints the fit and the percentile estimators.
"""

from .estimator import report
from .steps import step_model

MODEL = step_model("prefill", "gptoss-120b-tp4-20261103", "prefill, gpt-oss-120b, TP 4")

if __name__ == "__main__":
    report(MODEL)
