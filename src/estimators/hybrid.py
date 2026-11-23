"""One hybrid (mixed-chunk) step of gpt-oss-20b on one A100 80GB with CUDA graphs: new prompts plus one decode token per running request.

Degree 2 + sum l^2 (see steps.py), fitted on results/gptoss-cuda-graphs-20261029.
Input: [prompt lengths, decode context lengths], both non-empty, e.g. [[1024], [2048] * 16]. Output: ms.

Usage: python -m estimators.hybrid   (from src/) prints the fit and the percentile estimators.
"""

from .estimator import report
from .steps import step_model

MODEL = step_model("hybrid", "gptoss-cuda-graphs-20261029", "hybrid, gpt-oss-20b, 1 A100, CUDA graphs")

if __name__ == "__main__":
    report(MODEL)
