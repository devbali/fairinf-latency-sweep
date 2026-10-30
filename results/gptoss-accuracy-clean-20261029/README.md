# gpt-oss-20b accuracy on eu1, without CUDA graphs

`sanity_accuracy.py` run on eu1 on 2026-10-29, with the server command of
`results/gptoss-clean-20261028`'s setup: the command in
`results/gptoss-accuracy-cuda-graphs-20261029`, with
`--disable-piecewise-cuda-graph` instead of `--piecewise-cuda-graph-compiler
inductor` and no `TORCHINDUCTOR_FORCE_DISABLE_CACHES`.

- `server.log`, `mmlu.log`, `gpqa.log`, `scores.json`: as there.

At low reasoning effort, MMLU scored 68.0% (2,000 questions) and GPQA Diamond
33.3% (198), against OpenAI's 80.4% and 56.8%; with CUDA graphs, 68.4% and
36.4%. `results/gptoss-accuracy-cuda-graphs-20261029/README.md` compares
them. The gap to OpenAI's numbers comes from SGLang's eval script, which this
first version of `sanity_accuracy.py` ran: with OpenAI's own eval code, the
setup with CUDA graphs scores GPQA Diamond 59.0%
(`results/gptoss-openai-evals-cuda-graphs-20261029`).
