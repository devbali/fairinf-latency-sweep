"""One decode step of gpt-oss-120b with tensor parallelism over four A100 80GB, run as a CUDA graph.

Degree 2 + sum l^2 (see steps.py), fitted on results/gptoss-120b-tp4-20261103.
Input: each running request's context length, e.g. [2048] * 32. Output: ms.

Usage: python -m estimators.decode_tp4   (from src/) prints the fit and the percentile estimators.
"""

from .estimator import report
from .steps import step_model

MODEL = step_model("decode", "gptoss-120b-tp4-20261103", "decode, gpt-oss-120b, TP 4")

if __name__ == "__main__":
    report(MODEL)
