r"""Run SGLang's PD disaggregation on one machine, and record what each request goes through.

Pass the commands that start the prefill server and the decode server, one
string each, as the sanity checks take a server command:

    python src/pd_runner.py \
        --prefill "CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path MODEL ... \
            --disaggregation-mode prefill --port 30000" \
        --decode "CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server --model-path MODEL ... \
            --disaggregation-mode decode --port 30001"

This starts both servers, sends them the plan (src/pd_batches.json by default)
and stops them. How the KV cache moves is up to the commands: the transfer
backend (--disaggregation-transfer-backend, mooncake by default), its
environment variables and the GPUs. NIXL's UCX copies from GPU to GPU only if
each server sees both GPUs: then leave CUDA_VISIBLE_DEVICES out and give the
decode server --base-gpu-id 1, as vm/README.md does.

Each command gets three flags if it lacks them, none of which changes how a
server schedules or transfers: --enable-metrics and --skip-tokenizer-init put
the scheduler's times in each answer, and --enable-request-time-stats-logging
logs the stages of each request as it finishes. --skip-tokenizer-init sends a
server's output straight from its scheduler to its HTTP server, without
detokenizing it; through the detokenizer the times get lost (it re-sends the
scheduler's time stats, and a re-sent copy is empty). The prompts are token
ids, so no tokenizer is needed.

A batch is sent the way SGLang's router sends a request in PD mode: to both
servers at once, as one /generate request with the same prompts, rids and
sampling parameters, plus the prefill server's bootstrap host and port and a
random bootstrap room per prompt. The prefill server answers once a prompt's KV
cache has reached the decode server, the decode server once it has generated
--max-new-tokens tokens. The next batch goes out --gap seconds after both
answers are in, so the transfers of different batches never overlap. Prompts
are random token ids, new for every request, so the prefill server's radix
cache never matches one.

First the four batches of WARMUP are sent once, recorded as phase "warmup".
Then each of the plan's "batches" is sent --samples times, then each of its
"bursts". results.jsonl gets one line per request:
- phase, batch (its lengths), index (the batch's place in its list), sample,
  and the request's place in the batch, length, rid and bootstrap room;
- "prefill" and "decode": the times in each server's answer (meta_info, in
  seconds since the epoch), then the stage durations of its ReqTimeStats log
  line (in ms; transfer_speed in GB/s and transfer_total in MB);
- "client": when the batch went out and when each answer came back.
requests.jsonl holds the same lines without the log stages, written as the run
goes. `--parse DIR` rebuilds DIR/results.jsonl from it and the server logs.
run.log holds the setup: both commands as run, package versions, the GPUs and
how they are linked, and each NVLink's data counters before and after the run.
The servers' logs, prefill.log and decode.log, go next to it.

The GPUs have to be free. The run stops if a warmup batch fails, or five
batches in a row do. Exits with 0 if every request got both answers, 1 if some
did not, and 2 if the run could not start, stopped early or a server exited.
"""

import argparse
import json
import os
import random
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

from server_command import LAUNCHER, foreground, gpu_processes, instead  # noqa: E402

MODE = re.compile(r"--disaggregation-mode(?:=|\s+)(\S+)")
PORT = re.compile(r"--port(?:=|\s+)(\d+)")
HOST = re.compile(r"--host(?:=|\s+)(\S+)")
BOOTSTRAP_PORT = re.compile(r"--disaggregation-bootstrap-port(?:=|\s+)(\d+)")
# Flags each command gets if it lacks them. An answer's meta_info gets the scheduler's
# times only with --enable-metrics (tokenizer_manager.py, where it builds meta_info), and
# only if they skip the detokenizer: SchedulerReqTimeStats.__getstate__ sends them only
# while enable_metrics is set, which a copy unpickled in the detokenizer no longer is.
ADDED_FLAGS = [
    (re.compile(r"--enable-metrics(?:\s|$)"), "--enable-metrics"),
    (re.compile(r"--skip-tokenizer-init(?:\s|$)"), "--skip-tokenizer-init"),
    (re.compile(r"--enable-request-time-stats-logging(?:\s|$)"), "--enable-request-time-stats-logging"),
]
# The line each server logs as a request finishes (Req.log_time_stats)
STATS_LINE = re.compile(
    r"ReqTimeStats\(rid=(?P<rid>[^,]+), (?:bootstrap_room=(?P<room>\d+), )?input_len=(?P<input_len>\d+), "
    r"cached_input_len=(?P<cached>\d+), output_len=(?P<output_len>\d+), type=(?P<type>\w+)\): (?P<stats>.*)"
)
FIELD = re.compile(r"#?(\w+)=(-?[\d.]+)")
# Sent once before the recorded batches: one short, one long and two many-prompt batches
WARMUP = [[1024], [64] * 8, [8192], [16] * 64]
VOCAB = 10_000  # prompts are token ids below this, as in runner.py
PROGRESS_EVERY = 60  # seconds between progress lines
# Data each GPU sent and received over each NVLink so far: run.log gets it before and after
# the run, so the difference shows whether the KV cache went over NVLink
NVLINK_COUNTERS = "nvidia-smi nvlink -gt d 2>&1"
MAX_FAILED_IN_A_ROW = 5  # batches with failed requests, one after another, before the run stops


class RunFailed(Exception):
    pass


class Server:
    """One SGLang server, started from its command in a process group of its own."""

    def __init__(self, name, command, out):
        self.name, self.command = name, command
        port, host = PORT.findall(command), HOST.findall(command)
        self.port = int(port[-1]) if port else 30000
        host = host[-1] if host else "127.0.0.1"
        self.host = "127.0.0.1" if host in ("0.0.0.0", "localhost") else host
        self.url = f"http://{self.host}:{self.port}"
        self.log_path = os.path.join(out, f"{name}.log")
        self.proc = None

    def start(self):
        self.log = open(self.log_path, "wb")
        self.proc = subprocess.Popen(["bash", "-c", self.command], stdout=self.log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ, PYTHONUNBUFFERED="1"), start_new_session=True)

    def check(self):
        if self.proc.poll() is not None:
            raise RunFailed(f"The {self.name} server exited with code {self.proc.returncode}:\n{tail(self.log_path)}")

    def stop(self):
        """Stop the server and everything it started, and wait until all of it has exited."""
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            for _ in range(60):
                self.proc.poll()  # reaps the shell, which would otherwise stay in the group
                os.killpg(self.proc.pid, 0)  # raises once no process of the group is left
                time.sleep(1)
            os.killpg(self.proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.proc.wait()
        self.log.close()


def tail(path, n=20):
    with open(path, errors="replace") as f:
        return "".join(f.readlines()[-n:])


def wait_ready(servers, timeout):
    """Wait until every server answers /health_generate. Returns the seconds it took."""
    start, waiting = time.time(), list(servers)
    while waiting and time.time() - start < timeout:
        for server in list(waiting):
            server.check()
            try:
                with urllib.request.urlopen(server.url + "/health_generate", timeout=60) as response:
                    if response.status == 200:
                        waiting.remove(server)
                        print(f"  the {server.name} server was ready after {time.time() - start:.0f} s", flush=True)
            except (urllib.error.URLError, OSError):
                pass
        if waiting:
            time.sleep(5)
    if waiting:
        raise RunFailed(f"The {waiting[0].name} server was not ready after {timeout} s:\n{tail(waiting[0].log_path)}")
    return time.time() - start


def post(url, data, timeout):
    """POST JSON bytes; returns (the parsed answer or None, an error or None, when it came back)."""
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response), None, time.time()
    except urllib.error.HTTPError as error:
        return None, f"HTTP {error.code}: {error.read()[:500].decode(errors='replace')}", time.time()
    except (urllib.error.URLError, OSError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}", time.time()


def meta(answer, j):
    """The scalar fields of the j-th answer's meta_info, or an error."""
    try:
        info = answer[j]["meta_info"]
    except (TypeError, LookupError):
        return {"error": f"no meta_info in the answer: {str(answer)[:300]}"}
    kept = {key: value for key, value in info.items() if not isinstance(value, (list, dict))}
    finish = info.get("finish_reason")
    if isinstance(finish, dict):
        kept["finish_reason"] = finish.get("type")
        if finish.get("type") == "abort":
            kept["error"] = finish.get("message", "aborted")
    return kept


def send_batch(prefill, decode, bootstrap_port, lens, rids, args, tokens, rooms):
    """Send one batch to both servers as SGLang's router would. Returns one record per request."""
    batch_rooms = [rooms.getrandbits(63) for _ in lens]
    data = json.dumps({
        "input_ids": [[tokens.randrange(VOCAB) for _ in range(n)] for n in lens],
        "rid": rids,
        "sampling_params": {"max_new_tokens": args.max_new_tokens, "temperature": 0.0, "ignore_eos": True},
        "bootstrap_host": prefill.host,
        "bootstrap_port": bootstrap_port,
        "bootstrap_room": batch_rooms,
    }).encode()
    answers = {}

    def call(server):
        answers[server.name] = post(server.url + "/generate", data, args.timeout)

    threads = [threading.Thread(target=call, args=(server,)) for server in (prefill, decode)]
    sent = time.time()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    records = []
    for j, (n, rid, room) in enumerate(zip(lens, rids, batch_rooms)):
        record = {"request": j, "len": n, "rid": rid, "room": room,
                  "client": {"sent": sent}}
        for name in ("prefill", "decode"):
            answer, error, back = answers[name]
            record[name] = {"error": error} if error else meta(answer, j)
            record["client"][name] = back
        records.append(record)
    return records


def log_stats(path):
    """The stage durations each request's ReqTimeStats line gives, by rid."""
    stats = {}
    with open(path, errors="replace") as f:
        for line in f:
            match = STATS_LINE.search(line)
            if match:
                fields = {key: float(value) for key, value in FIELD.findall(match["stats"])}
                fields.update(input_len=int(match["input_len"]), cached_input_len=int(match["cached"]),
                              output_len=int(match["output_len"]), type=match["type"])
                stats[match["rid"]] = fields
    return stats


def merge(out):
    """Write out/results.jsonl: requests.jsonl with each server's log stages added. Returns its line count."""
    logged = {name: log_stats(os.path.join(out, f"{name}.log")) for name in ("prefill", "decode")}
    count = 0
    with open(os.path.join(out, "requests.jsonl")) as f, open(os.path.join(out, "results.jsonl"), "w") as results:
        for line in f:
            record = json.loads(line)
            for name in ("prefill", "decode"):
                record[name]["log"] = logged[name].get(record["rid"])
            results.write(json.dumps(record) + "\n")
            count += 1
    return count


def setup_lines(prefill, decode):
    """What run.log starts with: the commands, package versions, the GPUs and their links."""
    probe = ("import importlib.metadata as m\n"
             "for name in ('mooncake-transfer-engine-cuda13', 'mooncake-transfer-engine', 'nixl', 'nixl-cu13', 'nixl-cu12',\n"
             "             'sglang-router'):\n"
             "    try: print(f'{name:<32} {m.version(name)}')\n"
             "    except m.PackageNotFoundError: pass\n")
    shell = [
        ("versions", instead(prefill.command, shlex.quote(os.path.join(SRC, "versions.py")))),
        ("transfer packages", instead(prefill.command, f"-c {shlex.quote(probe)}")),
        ("gpus", "nvidia-smi --query-gpu=index,name,pci.bus_id,driver_version,memory.total --format=csv"),
        ("gpu links", "nvidia-smi topo -m"),
        ("rdma devices", "ls /sys/class/infiniband 2>/dev/null || echo none"),
        ("nvlink data counters before the run", NVLINK_COUNTERS),
    ]
    lines = [f"prefill command: {prefill.command}", f"decode command: {decode.command}", f"cpus: {os.cpu_count()}"]
    for title, command in shell:
        try:
            output = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=600)
            lines += [f"== {title}", (output.stdout + output.stderr).rstrip()]
        except (OSError, subprocess.SubprocessError) as error:
            lines += [f"== {title}", f"unavailable: {error}"]
    return lines


def command_for(name, command):
    """The command, checked to start SGLang once in the given PD mode, with ADDED_FLAGS."""
    command = foreground(command)
    if len(LAUNCHER.findall(command)) != 1:
        raise RunFailed(f"The {name} command has to start SGLang once, with `python -m sglang.launch_server` "
                        "or `sglang serve`.")
    modes = MODE.findall(command)
    if not modes or modes[-1] != name:
        raise RunFailed(f"The {name} command has to pass --disaggregation-mode {name}.")
    missing = [flag for pattern, flag in ADDED_FLAGS if not pattern.search(command)]
    if missing:
        match = LAUNCHER.search(command)
        command = f"{command[:match.end()]} {' '.join(missing)}{command[match.end():]}"
    return command


def run(args, out, prefill, decode, bootstrap_port, plan):
    tokens, rooms = random.Random(args.seed), random.Random()
    phases = [("warmup", WARMUP, 1), ("batches", plan.get("batches", []), args.samples),
              ("bursts", plan.get("bursts", []), args.samples)]
    total = sum(len(batches) * samples for _, batches, samples in phases)
    done = failed = in_a_row = 0
    start = shown = time.time()
    with open(os.path.join(out, "requests.jsonl"), "w") as requests:
        for phase, batches, samples in phases:
            for index, lens in enumerate(batches):
                for sample in range(samples):
                    rids = [f"{phase}-{index:04d}-{sample}-{j:03d}" for j in range(len(lens))]
                    records = send_batch(prefill, decode, bootstrap_port, lens, rids, args, tokens, rooms)
                    for record in records:
                        requests.write(json.dumps({"phase": phase, "batch": lens, "index": index, "sample": sample,
                                                   **record}) + "\n")
                    requests.flush()
                    errors = [r for r in records if "error" in r["prefill"] or "error" in r["decode"]]
                    prefill.check()
                    decode.check()
                    in_a_row = in_a_row + 1 if errors else 0
                    if errors:
                        failed += 1
                        first = errors[0]
                        message = (f"{phase} {index} sample {sample}: {len(errors)} of {len(records)} requests failed: "
                                   f"{first['prefill'].get('error') or first['decode'].get('error')}")
                        # A failed warmup, or batch after batch failing, means the setup is broken:
                        # stop, rather than wait out the timeout of every batch left
                        if phase == "warmup" or in_a_row >= MAX_FAILED_IN_A_ROW:
                            raise RunFailed(f"Stopping: {message}")
                        print(f"  {message}", flush=True)
                    done += 1
                    if time.time() - shown > PROGRESS_EVERY:
                        print(f"  {done}/{total} batches sent, {failed} with failures, "
                              f"{(time.time() - start) / 60:.1f} min", flush=True)
                        shown = time.time()
                    time.sleep(args.gap)
    print(f"Sent {done} batches in {(time.time() - start) / 60:.1f} min; {failed} had failed requests.", flush=True)
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prefill", help="the command that starts the prefill server, in quotes")
    parser.add_argument("--decode", help="the command that starts the decode server, in quotes")
    parser.add_argument("--plan", default=os.path.join(SRC, "pd_batches.json"),
                        help='JSON plan with "batches" and/or "bursts" lists of prompt lengths')
    parser.add_argument("--samples", type=int, default=3, help="times to send each batch and burst")
    parser.add_argument("--max-new-tokens", type=int, default=2,
                        help="tokens per request: the prefill server's first one, then decode steps")
    parser.add_argument("--gap", type=float, default=0.5, help="seconds between one batch's answers and the next")
    parser.add_argument("--timeout", type=float, default=300, help="seconds to wait for a batch's answers")
    parser.add_argument("--seed", type=int, default=0, help="seed of the prompts' random tokens")
    parser.add_argument("--out", help="a new directory for the results and logs")
    parser.add_argument("--start-timeout", type=int, default=1800, help="seconds to wait for the servers")
    parser.add_argument("--force", action="store_true", help="run even if other processes are using a GPU")
    parser.add_argument("--dry-run", action="store_true", help="print the commands that would run, and stop")
    parser.add_argument("--parse", metavar="DIR", help="only rebuild DIR/results.jsonl from its requests and logs")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)

    if args.parse:
        print(f"Wrote {merge(os.path.abspath(args.parse))} requests to {args.parse}/results.jsonl.")
        return 0
    if not args.prefill or not args.decode:
        parser.error("--prefill and --decode are required")
    machine, now = socket.gethostname(), time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out = os.path.abspath(args.out or os.path.join(SRC, "..", "results", f"pd-{machine}-{now}"))
    try:
        prefill = Server("prefill", command_for("prefill", args.prefill), out)
        decode = Server("decode", command_for("decode", args.decode), out)
    except RunFailed as error:
        print(error, file=sys.stderr)
        return 2
    found = BOOTSTRAP_PORT.findall(prefill.command)
    bootstrap_port = int(found[-1]) if found else 8998  # SGLang's default --disaggregation-bootstrap-port
    with open(args.plan) as f:
        plan = json.load(f)
    if args.dry_run:
        print(f"This would start\n\n  {prefill.command}\n\nand\n\n  {decode.command}\n\nthen send "
              f"{len(plan.get('batches', []))} batches and {len(plan.get('bursts', []))} bursts of {args.plan}, "
              f"{args.samples} times each, with the prefill server's bootstrap at {prefill.host}:{bootstrap_port}, "
              f"logging to {out}.")
        return 0

    busy = gpu_processes()
    if busy and not args.force:
        print("Other processes are using a GPU:\n  " + "\n  ".join(busy) + "\nStop them, or pass --force.",
              file=sys.stderr)
        return 2
    if prefill.port == decode.port:
        print("The two servers need different --port values.", file=sys.stderr)
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
    failed = None
    try:
        prefill.start()
        decode.start()
        wait_ready((prefill, decode), args.start_timeout)
        failed = run(args, out, prefill, decode, bootstrap_port, plan)
    except RunFailed as error:
        print(f"\n{error}", file=sys.stderr)
    finally:
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
