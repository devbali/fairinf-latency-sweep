"""A no-op scheduling policy that also exposes a no-op cache policy.

The fork's scheduler calls tree_cache.cache_hooks_policy on every admission
(scheduler.py:2941), but only RadixCache defines it. gpt-oss's sliding-window
layers get SWARadixCache instead, so a plain gpt-oss server crashes on its
first request. The scheduler attaches a policy's `cache_policy` to whatever
tree cache is in use (scheduler.py:1131), so serve gpt-oss with
    --scheduling-policy-path noop_cache_policy.NoOpWithCachePolicy
and vm/ on PYTHONPATH. Scheduling stays SGLang's default.
"""

from sglang.srt.cache_hooks.no_op_cache_policy import NoOpCachePolicy
from sglang.srt.scheduling_hooks.no_op_policy import NoOpSchedulingPolicy


class NoOpWithCachePolicy(NoOpSchedulingPolicy):
    cache_policy = NoOpCachePolicy()
