"""One prefill step of gpt-oss-20b on one A100 80GB with piecewise CUDA graphs.

Degree 2 + sum l^2 (see steps.py), fitted on results/gptoss-cuda-graphs-20261029.
Input: the prompt lengths, e.g. [512, 1024]. Output: ms.

Usage: python -m estimators.prefill   (from src/) prints the fit and the percentile estimators.
"""

from .estimator import report
from .steps import step_model

MODEL = step_model("prefill", "gptoss-cuda-graphs-20261029", "prefill, gpt-oss-20b, 1 A100, CUDA graphs")

if __name__ == "__main__":
    report(MODEL)
