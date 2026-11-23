"""One decode step of gpt-oss-20b on one A100 80GB, run as a CUDA graph.

Degree 2 + sum l^2 (see steps.py), fitted on results/gptoss-cuda-graphs-20261029.
Input: each running request's context length, e.g. [2048] * 32. Output: ms.

Usage: python -m estimators.decode   (from src/) prints the fit and the percentile estimators.
"""

from .estimator import report
from .steps import step_model

MODEL = step_model("decode", "gptoss-cuda-graphs-20261029", "decode, gpt-oss-20b, 1 A100, CUDA graphs")

if __name__ == "__main__":
    report(MODEL)
