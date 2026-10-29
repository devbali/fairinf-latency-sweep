"""A short batch plan for checking that a setup matches the latency models.

Timing it takes about 1.5 minutes on one A100, after the model loads. Every
batch here is also in batches.py, so the full sweep in
results/gptoss-clean-20261028 has a time for each one, and compare_runs.py
compares the two runs batch by batch.
The batches cover each regime the models describe:
- prefill: one prompt from the CPU floor up to the full 8,192-token budget, the
  budget split into 4 to 64 equal prompts, and one long prompt with short ones;
- decode: 1 to 256 requests, and contexts from 64 tokens to 98k;
- hybrid: every prefill part with every decode part, as in batches.py.
"""

PREFILL_BATCHES = [[length] for length in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)]
PREFILL_BATCHES += [[length] * (8192 // length) for length in (128, 256, 512, 1024, 2048)]
PREFILL_BATCHES += [
    [6144] + [16] * 31,
    [4096] + [64] * 31,
    [2048] + [256] * 15,
    [1024] + [16] * 15,
    [512] + [256] * 7,
]

DECODE_BATCHES = [[1024] * n for n in (1, 2, 4, 8, 16, 32, 64, 128, 256)]
DECODE_BATCHES += [[64], [4096], [65536], [64] * 256, [4096] * 64, [16384] * 16, [65536] * 4]
DECODE_BATCHES += [
    [98304] + [128] * 7,
    [98304] + [1024] * 31,
    [32768] + [1024] * 127,
    [8192] + [128] * 255,
]

PREFILL_PARTS = [[256], [1024], [4096], [1024] * 4, [256] * 8, [128] * 32]
DECODE_PARTS = [[length] * n for n in (1, 8, 32, 128) for length in (512, 2048)]
HYBRID_BATCHES = [(prefill, decode) for prefill in PREFILL_PARTS for decode in DECODE_PARTS]
