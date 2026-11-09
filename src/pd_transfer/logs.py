"""The prefill and decode servers' step lines, with SGLANG_LOG_MS=1's millisecond timestamps.

In PD mode the prefill server logs a "Prefill batch" line once it has processed a step's results and posted
its transfers; the decode server logs a "Decode batch" line as it processes each step's results (every step
with --decode-log-interval 1), just before its next poll for arrived KV caches."""
import calendar, gzip, re, time

TS = r"^\[(?P<ts>\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?:\.(?P<ms>\d{3}))?[^\]]*\] "
PREFILL = re.compile(TS + r"Prefill batch[^,]*, #new-seq: (?P<seqs>\d+), #new-token: (?P<tokens>\d+), "
                     r"#cached-token: (?P<cached>\d+),.*?#running-req: (?P<running>\d+), #queue-req: (?P<queue>\d+)"
                     r".*?#bootstrap-req: (?P<bootstrap>\d+), #inflight-req: (?P<inflight>\d+)")
DECODE = re.compile(TS + r"Decode batch[^,]*, #running-req: (?P<running>\d+), #full token: (?P<full>\d+),"
                    r".*?#prealloc-req: (?P<prealloc>\d+), #transfer-req: (?P<transfer>\d+), "
                    r"#retracted-req: (?P<retracted>\d+)")


def stamp(m):
    """Epoch seconds of a line: SGLang logs local time, and the VM runs on UTC."""
    return calendar.timegm(time.strptime(m["ts"], "%Y-%m-%d %H:%M:%S")) + int(m["ms"] or 0) / 1000


def read(path, pattern):
    opener = gzip.open if path.endswith(".gz") else open
    rows = []
    with opener(path, "rt", errors="replace") as f:
        for line in f:
            if m := pattern.match(line):
                rows.append({"t": stamp(m), **{k: int(v) for k, v in m.groupdict().items() if k not in ("ts", "ms")}})
    return rows


if __name__ == "__main__":
    import sys
    for path, pat in ((sys.argv[1], PREFILL), (sys.argv[2], DECODE)):
        rows = read(path, pat)
        print(path, len(rows), rows[:2])
