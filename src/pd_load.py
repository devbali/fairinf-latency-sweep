r"""Send PD probe batches under background load, to see what a KV transfer's time depends on besides its batch.

Takes the two server commands pd_runner.py takes, starts both servers and runs
the scenarios of a load plan (src/pd_load.json by default, written by
generate_batches.py):

    python src/pd_load.py --prefill "$PREFILL" --decode "$DECODE" --out DIR

In each scenario, the plan's loads keep the prefill server, the decode server,
both or neither busy, and its probe batches go out one at a time, the way
pd_runner.py sends a batch, --gap seconds after the last one's answers. A load
is clients that each send one request at a time, again and again, until the
scenario ends; then the ones still decoding are aborted. The prefill server
never decodes: after a prefill step comes another, if one is queued, or
nothing. The decode server finds arrived KV caches between its decode steps.
So in the plan:
- "prefill" keeps a prefill step queued, with one output token per request,
  so the decode server only receives KV;
- "decode" keeps the decode server running decode steps;
- "pressure" is more than a decode server with a capped KV cache
  (--max-total-tokens in its command) holds, so it has to make requests wait
  and retract running ones.
The probes go round-robin, every batch once before any twice, so drifting
load reaches all of them alike. --probes sends another plan's "batches" as the
probes instead, e.g. the PD plan's, src/pd_batches.json, to fit a transfer
model under each load.

results.jsonl has pd_runner.py's line per request, which pd_fits.py's load()
reads, plus kind: "probe" for a probe, with phase the scenario, or "load", with
phase SCENARIO:LOAD. Aborted load requests have an error. src/pd_load_report.py
reports on a run.

--scenarios picks scenarios by name, in order. NAME:profile runs scenario NAME
again with --profile running on each server's scheduler process (SGLang titles
it sglang::scheduler) while it runs. {pid} in the template becomes that process,
{out} DIR/profiles/NAME-SERVER.txt. A profiler is stopped with SIGINT when its
scenario ends, e.g. py-spy, which writes its output then:

    --profile "py-spy record --nonblocking --idle --threads --rate 250 --format raw --pid {pid} --output {out}"

py-spy needs ptrace rights over the servers: run as root, or as their user
with kernel.yama.ptrace_scope set to 0.

NAME:torch runs scenario NAME again with SGLang's torch profiler (CPU and GPU)
on both servers, through their /start_profile and /stop_profile, for the first
--torch-s seconds of its probes. Each server writes its trace to
DIR/profiles/NAME-torch-SERVER/.

A load with "spread" asks for fewer output tokens in each client's first
request, (k + 1) / clients of new_tokens for client k, so the clients'
requests end at spread-out times instead of together.

Exits with 0 if every probe got both answers, 1 if some did not, and 2 if the
run could not start, stopped early or a server exited.
"""

import argparse
import json
import os
import random
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

from pd_runner import (  # noqa: E402
    BOOTSTRAP_PORT, MAX_FAILED_IN_A_ROW, NVLINK_COUNTERS, WARMUP, RunFailed, Server, command_for, merge, post,
    send_batch, setup_lines, wait_ready,
)
from server_command import gpu_processes  # noqa: E402

ABORT_AFTER_S = 2  # after a scenario stops its loads: time for just-sent requests to reach the decode server
DRAIN_S = 120  # seconds to wait for a scenario's load requests to end after aborting them
SETTLE_S = 5  # seconds between scenarios
PROFILE_STOP_S = 60  # seconds a profiler gets to write its output after SIGINT
TORCH_STOP_S = 900  # seconds a server gets to write its torch trace


class Writer:
    """requests.jsonl, written from several threads."""

    def __init__(self, path):
        self.file, self.lock = open(path, "w"), threading.Lock()

    def write(self, records, **fields):
        with self.lock:
            for record in records:
                self.file.write(json.dumps({**fields, **record}) + "\n")
            self.file.flush()

    def close(self):
        self.file.close()


class Load:
    """A load's clients, each sending one request at a time until stopped."""

    def __init__(self, spec, scenario, ctx):
        self.spec, self.scenario, self.ctx = spec, scenario, ctx
        self.stop = threading.Event()
        self.in_flight = {}  # client -> rid of its request out now
        self.sent = self.failed = 0
        self.threads = [threading.Thread(target=self.client, args=(k,), daemon=True)
                        for k in range(spec["clients"])]

    def client(self, k):
        ctx, name = self.ctx, self.spec["name"]
        tokens, rooms = random.Random(f"{self.scenario}-{name}-{k}"), random.Random()
        sampling = SimpleNamespace(max_new_tokens=self.spec["new_tokens"], timeout=ctx.args.timeout)
        if self.spec.get("spread"):  # the first request is shorter, so the clients' requests end at spread-out times
            sampling.max_new_tokens = max(1, self.spec["new_tokens"] * (k + 1) // self.spec["clients"])
        i = 0
        while not self.stop.is_set():
            rid = f"{self.scenario}-load-{name}-{k:02d}-{i:05d}"
            self.in_flight[k] = rid
            records = send_batch(ctx.prefill, ctx.decode, ctx.bootstrap_port, [self.spec["len"]], [rid], sampling,
                                 tokens, rooms)
            self.in_flight.pop(k, None)
            ctx.writer.write(records, phase=f"{self.scenario}:{name}", kind="load", batch=[self.spec["len"]],
                             index=k, sample=i)
            self.sent += 1
            self.failed += any("error" in r["prefill"] or "error" in r["decode"] for r in records)
            sampling.max_new_tokens = self.spec["new_tokens"]
            i += 1

    def start(self):
        for thread in self.threads:
            thread.start()

    def finish(self):
        """Stop sending, abort what is still decoding, and wait for every client to end."""
        self.stop.set()
        time.sleep(ABORT_AFTER_S)
        for rid in list(self.in_flight.values()):
            for server in (self.ctx.decode, self.ctx.prefill):
                post(server.url + "/abort_request", json.dumps({"rid": rid}).encode(), 30)
        deadline = time.time() + DRAIN_S
        for thread in self.threads:
            thread.join(max(0, deadline - time.time()))
        return sum(thread.is_alive() for thread in self.threads)


def scheduler_pid(server):
    """The process id of the server's scheduler: in its process group, titled sglang::scheduler."""
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            with open(f"/proc/{pid}/stat") as f:
                group = int(f.read().rsplit(")", 1)[1].split()[2])
            if group != server.proc.pid:
                continue
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                if f.read().startswith(b"sglang::scheduler"):
                    return int(pid)
        except (OSError, ValueError, IndexError):
            continue
    return None


def start_profilers(ctx, name):
    profilers = []
    os.makedirs(os.path.join(ctx.out, "profiles"), exist_ok=True)
    for server in (ctx.prefill, ctx.decode):
        pid = scheduler_pid(server)
        if pid is None:
            print(f"  no scheduler process found for the {server.name} server; not profiling it", flush=True)
            continue
        out = os.path.join(ctx.out, "profiles", f"{name}-{server.name}.txt")
        log = open(out + ".log", "wb")
        command = shlex.split(ctx.args.profile.format(pid=pid, out=out))
        profilers.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log))
    return profilers


def stop_profilers(profilers):
    for proc, _ in profilers:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
    for proc, log in profilers:
        try:
            proc.wait(PROFILE_STOP_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log.close()


def call(url, body, timeout):
    """POST body as JSON to url; returns the answer's text, or the error."""
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read().decode(errors="replace").strip()
    except (urllib.error.URLError, OSError) as error:
        return f"{type(error).__name__}: {error}"


def torch_profile(ctx, tag, action):
    """Start or stop SGLang's torch profiler on both servers; stopping returns once their traces are written."""
    for server in (ctx.prefill, ctx.decode):
        if action == "start":
            body = {"output_dir": os.path.join(ctx.out, "profiles", f"{tag}-{server.name}"),
                    "activities": ["CPU", "GPU"], "with_stack": False, "record_shapes": False}
            answer = call(server.url + "/start_profile", body, 60)
        else:
            answer = call(server.url + "/stop_profile", {}, TORCH_STOP_S)
        print(f"  torch profiler {action}, {server.name} server: {answer}", flush=True)


def send_probes(ctx, tag, samples, max_s):
    """Send the probes round-robin, samples times or until max_s seconds are up. Returns (sent, failed)."""
    sampling = SimpleNamespace(max_new_tokens=ctx.args.max_new_tokens, timeout=ctx.args.timeout)
    sent = failed = in_a_row = 0
    start = time.time()
    for sample in range(samples):
        for index, lens in enumerate(ctx.probes):
            if max_s and time.time() - start > max_s:
                print(f"  {max_s} s are up: stopping after {sent} probes", flush=True)
                return sent, failed
            rids = [f"{tag}-probe-{index:02d}-{sample}-{j:03d}" for j in range(len(lens))]
            records = send_batch(ctx.prefill, ctx.decode, ctx.bootstrap_port, lens, rids, sampling,
                                 ctx.tokens, ctx.rooms)
            ctx.writer.write(records, phase=tag, kind="probe", batch=lens, index=index, sample=sample)
            ctx.prefill.check()
            ctx.decode.check()
            sent += 1
            errors = [r for r in records if "error" in r["prefill"] or "error" in r["decode"]]
            in_a_row = in_a_row + 1 if errors else 0
            if errors:
                failed += 1
                first = errors[0]
                print(f"  probe {index} sample {sample}: {len(errors)} of {len(records)} requests failed: "
                      f"{first['prefill'].get('error') or first['decode'].get('error')}", flush=True)
                if in_a_row >= MAX_FAILED_IN_A_ROW:
                    raise RunFailed(f"Stopping: {MAX_FAILED_IN_A_ROW} probes in a row failed in {tag}")
            time.sleep(ctx.args.gap)
    return sent, failed


def run_scenario(ctx, scenario, mode):
    """Start the scenario's loads, send its probes, then stop the loads. Returns the probes with failures.

    mode is "", or "profile" or "torch" to run it with --profile or the torch profiler."""
    tag = scenario["name"] + (f"-{mode}" if mode else "")
    loads = [Load(ctx.loads[load], tag, ctx) for load in scenario["loads"]]
    samples = ctx.args.samples or scenario["samples"]
    print(f"== {time.strftime('%H:%M:%S')} {tag}: loads {', '.join(scenario['loads']) or 'none'}, "
          f"{samples} x {len(ctx.probes)} probes", flush=True)
    profilers = start_profilers(ctx, tag) if mode == "profile" else []
    start = time.time()
    try:
        for load in loads:
            load.start()
            time.sleep(load.spec["warm_s"])
        if mode == "torch":
            torch_profile(ctx, tag, "start")
        sent, failed = send_probes(ctx, tag, samples, ctx.args.torch_s if mode == "torch" else scenario.get("max_s"))
    finally:
        if mode == "torch":
            torch_profile(ctx, tag, "stop")
        stuck = sum(load.finish() for load in loads)
        stop_profilers(profilers)
    print(f"  {sent} probes in {(time.time() - start) / 60:.1f} min, {failed} with failures; load requests: "
          + (", ".join(f"{load.spec['name']} {load.sent} ({load.failed} aborted or failed)" for load in loads)
             or "none") + (f"; {stuck} clients still waiting" if stuck else ""), flush=True)
    time.sleep(SETTLE_S)
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefill", help="the command that starts the prefill server, in quotes")
    parser.add_argument("--decode", help="the command that starts the decode server, in quotes")
    parser.add_argument("--plan", default=os.path.join(SRC, "pd_load.json"),
                        help='JSON plan with "probes", "loads" and "scenarios"')
    parser.add_argument("--probes", help='a JSON plan whose "batches" to send as the probes instead of the plan\'s, '
                        "e.g. src/pd_batches.json")
    parser.add_argument("--scenarios", default="idle,prefill,decode,both",
                        help="comma-separated scenario names, in order; NAME:profile runs NAME with --profile, "
                        "NAME:torch with SGLang's torch profiler")
    parser.add_argument("--profile", help="profiler command for each scheduler process, with {pid} and {out}")
    parser.add_argument("--torch-s", type=float, default=60,
                        help="seconds of probes in a NAME:torch scenario, with the torch profiler on")
    parser.add_argument("--samples", type=int, help="times to send each probe, instead of the plan's")
    parser.add_argument("--max-new-tokens", type=int, default=2, help="output tokens per probe request")
    parser.add_argument("--gap", type=float, default=0.5, help="seconds between one probe's answers and the next")
    parser.add_argument("--timeout", type=float, default=300, help="seconds to wait for a request's answers")
    parser.add_argument("--seed", type=int, default=0, help="seed of the probes' random tokens")
    parser.add_argument("--out", required=True, help="a new directory for the results and logs")
    parser.add_argument("--start-timeout", type=int, default=1800, help="seconds to wait for the servers")
    parser.add_argument("--force", action="store_true", help="run even if other processes are using a GPU")
    parser.add_argument("--dry-run", action="store_true", help="print what would run, and stop")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    if not args.prefill or not args.decode:
        parser.error("--prefill and --decode are required")

    with open(args.plan) as f:
        plan = json.load(f)
    probes = plan["probes"]
    if args.probes:
        with open(args.probes) as f:
            probes = json.load(f)["batches"]
    scenarios = {s["name"]: s for s in plan["scenarios"]}
    loads = {load["name"]: load for load in plan["loads"]}
    order = []
    for item in args.scenarios.split(","):
        name, _, flag = item.partition(":")
        if name not in scenarios or flag not in ("", "profile", "torch"):
            parser.error(f"unknown scenario {item!r}; the plan has {', '.join(scenarios)}")
        if flag == "profile" and not args.profile:
            parser.error(f"{item} needs --profile")
        order.append((scenarios[name], flag))
    out = os.path.abspath(args.out)
    try:
        prefill = Server("prefill", command_for("prefill", args.prefill), out)
        decode = Server("decode", command_for("decode", args.decode), out)
    except RunFailed as error:
        print(error, file=sys.stderr)
        return 2
    found = BOOTSTRAP_PORT.findall(prefill.command)
    bootstrap_port = int(found[-1]) if found else 8998  # SGLang's default --disaggregation-bootstrap-port
    if args.dry_run:
        print(f"This would start\n\n  {prefill.command}\n\nand\n\n  {decode.command}\n\nthen run "
              + ", ".join(s["name"] + (f":{mode}" if mode else "") for s, mode in order)
              + f" with {len(probes)} probe batches each, logging to {out}.")
        return 0

    busy = gpu_processes()
    if busy and not args.force:
        print("Other processes are using a GPU:\n  " + "\n  ".join(busy) + "\nStop them, or pass --force.",
              file=sys.stderr)
        return 2
    for host, port in ((prefill.host, prefill.port), (decode.host, decode.port), (prefill.host, bootstrap_port)):
        with socket.socket() as s:
            if s.connect_ex((host, port)) == 0:
                print(f"Something already listens on {host}:{port}; stop it first.", file=sys.stderr)
                return 2
    if os.path.exists(os.path.join(out, "requests.jsonl")):
        print(f"{out} already has a run; pass a new --out.", file=sys.stderr)
        return 2
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(out, "run.log"), "w") as log:
        log.write("\n".join(setup_lines(prefill, decode)) + "\n")
    print(f"Starting both servers, logging to {out}.", flush=True)
    ctx = SimpleNamespace(args=args, out=out, prefill=prefill, decode=decode, bootstrap_port=bootstrap_port,
                          probes=probes, loads=loads, tokens=random.Random(args.seed), rooms=random.Random(),
                          writer=Writer(os.path.join(out, "requests.jsonl")))
    failed = None
    try:
        prefill.start()
        decode.start()
        wait_ready((prefill, decode), args.start_timeout)
        warm = SimpleNamespace(max_new_tokens=args.max_new_tokens, timeout=args.timeout)
        for index, lens in enumerate(WARMUP):
            rids = [f"warmup-{index:04d}-0-{j:03d}" for j in range(len(lens))]
            records = send_batch(prefill, decode, bootstrap_port, lens, rids, warm, ctx.tokens, ctx.rooms)
            ctx.writer.write(records, phase="warmup", kind="warmup", batch=lens, index=index, sample=0)
            if any("error" in r["prefill"] or "error" in r["decode"] for r in records):
                raise RunFailed(f"Stopping: warmup batch {index} failed")
            time.sleep(args.gap)
        failed = sum(run_scenario(ctx, scenario, mode) for scenario, mode in order)
    except RunFailed as error:
        print(f"\n{error}", file=sys.stderr)
    finally:
        ctx.writer.close()
        decode.stop()
        prefill.stop()
        counters = subprocess.run(["bash", "-c", NVLINK_COUNTERS], capture_output=True, text=True).stdout
        with open(os.path.join(out, "run.log"), "a") as log:
            log.write(f"== nvlink data counters after the run\n{counters.rstrip()}\n")
        print(f"Stopped both servers. Still on a GPU: {gpu_processes() or 'nothing'}", flush=True)
        if os.path.exists(os.path.join(out, "requests.jsonl")):
            print(f"Wrote {merge(out)} requests to {out}/results.jsonl.", flush=True)
    if failed is None:
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
