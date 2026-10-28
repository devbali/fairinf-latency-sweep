"""The fake batches runner.py times.

Each batch is a list with the prompt length of every request in it. A batch
is prefilled in a single forward pass, so one with too many tokens in total
can run out of GPU memory. The largest batch here has 32k tokens.
"""

# N requests of the same length
BATCHES = [[length] * n for n in (1, 4, 16) for length in (128, 512, 2048)]

# mixed lengths
BATCHES += [
    [2048, 512, 512, 128],
    [4096] + [64] * 15,
]
