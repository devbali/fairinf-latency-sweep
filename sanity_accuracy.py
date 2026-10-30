r"""Check that SGLang on a system serves gpt-oss-20b at its published accuracy.

Pass the command that starts the SGLang server there, as one string, as for
sanity_latency.py:

    python sanity_accuracy.py "TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 \
        PYTHONPATH=/path/to/fairinf-sglang/python:/path/to/fairinf-latency-sweep/vm \
        python3 -m sglang.launch_server --model-path lmsys/gpt-oss-20b-bf16 --dtype bfloat16 \
        --mem-fraction-static 0.7 --sampling-backend pytorch --piecewise-cuda-graph-compiler inductor \
        --scheduling-policy-path noop_cache_policy.NoOpWithCachePolicy --port 30000"

This starts that server, asks it the 198 GPQA Diamond questions with OpenAI's
own eval code for gpt-oss, and stops it. That code is gpt_oss.evals from
github.com/openai/gpt-oss, which OpenAI's guide to verifying gpt-oss
implementations runs. Its GPQA prompt, its answer grading and its client run
unchanged, with its settings: temperature 1, up to 131,072 tokens per answer,
and every question in flight at once. The one difference is --attempts, how
many times each question is asked: 1 by default, to keep the check short;
OpenAI's `python -m gpt_oss.evals --eval gpqa` asks 8 times.

The score is compared with OpenAI's for gpt-oss-20b (model card, arXiv
2508.10925, Table 3, GPQA Diamond without tools): 56.8% at low reasoning
effort, 66.0% at medium and 71.5% at high. It runs at low effort unless
--effort says otherwise. The score passes if it is at most max(5 points,
3 standard errors) below OpenAI's, with the standard error over the 198
questions: 46.2% at low effort. A broken setup scores near chance, 25%.

About the command:
- As for sanity_latency.py, it may start with environment assignments or nohup
  and end in a redirect or &; those are dropped.
- It gets two flags if it lacks them, neither of which changes how the model
  runs. --reasoning-parser gpt-oss keeps the model's reasoning out of the
  response's content, where the eval looks for the answer. With
  --allow-auto-truncate, SGLang lowers the eval's max_tokens, the model's whole
  context, to what fits after the prompt; without it, SGLang rejects every
  request and the eval scores each one as wrong.
- The server answers on its --port, 30000 by default, which must be free.

The eval code comes from --gpt-oss, a checkout of github.com/openai/gpt-oss; by
default this clones it, at GPT_OSS_COMMIT below, into ~/.cache. It runs with
this script's Python, or --python, which needs gpt-oss's eval packages: openai,
pandas, numpy, jinja2, tqdm and blobfile. The server's environment is not
touched. The questions come from OpenAI's simple-evals bucket.

The GPU has to be free. The server's log, the eval's output, every answer and
its grade, and the score go to --out, by default a new directory under
results/. Exits with 0 if the score passes, 1 if it is too low, and 2 if the
check could not run.
"""

import argparse
import collections
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from server_command import LAUNCHER, foreground, gpu_processes  # noqa: E402

# OpenAI's GPQA Diamond scores for gpt-oss-20b, in %, by reasoning effort: model card (arXiv 2508.10925), Table 3
REFERENCE = {"low": 56.8, "medium": 66.0, "high": 71.5}
QUESTIONS = 198

# OpenAI's eval code: main on 2026-10-29 (gpt_oss/evals last changed 2025-09-08), and gpt-oss's "eval" packages
GPT_OSS_REPO = "https://github.com/openai/gpt-oss.git"
GPT_OSS_COMMIT = "277020131d524f06bed5f66b694a92ed7f6e5224"
GPT_OSS_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "sanity-accuracy", f"gpt-oss-{GPT_OSS_COMMIT[:12]}")
EVAL_PACKAGES = ["openai", "pandas", "numpy", "jinja2", "tqdm", "blobfile"]

# Runs in the gpt-oss checkout: the GPQA eval and its client, built as `python -m gpt_oss.evals --eval gpqa
# --sampler chat_completions` builds them (gpt_oss/evals/__main__.py), but asking each question `attempts` times
EVAL = """
import gzip, json, sys
from gpt_oss.evals.chat_completions_sampler import ChatCompletionsSampler
from gpt_oss.evals.gpqa_eval import GPQAEval

base_url, model, effort, attempts, scores, answers = sys.argv[1:]
sampler = ChatCompletionsSampler(model=model, reasoning_model=True, reasoning_effort=effort, temperature=1.0,
                                 base_url=base_url, max_tokens=131_072)
result = GPQAEval(n_repeats=int(attempts), num_examples=None, debug=False, n_threads=1584)(sampler)
with gzip.open(answers, "wt") as f:
    json.dump({"score": result.score, "metrics": result.metrics, "htmls": result.htmls, "convos": result.convos}, f)
with open(scores, "w") as f:
    json.dump(dict(sorted((result.metrics | {"score": result.score}).items())), f, indent=2)
"""

# The flags the eval needs from the server, and why
FLAGS = [
    (re.compile(r"--reasoning-parser(?:=|\s|$)"), "--reasoning-parser gpt-oss",
     "keeps the model's reasoning out of the content the eval reads the answer from"),
    (re.compile(r"--allow-auto-truncate(?:\s|$)"), "--allow-auto-truncate",
     "lets SGLang take the eval's max_tokens, the model's whole context, and lower it to what fits"),
]
PORT = re.compile(r"--port(?:=|\s+)(\d+)")
HOST = re.compile(r"--host(?:=|\s+)(\S+)")
API_KEY = re.compile(r"--api-key(?:=|\s|$)")
PROGRESS = re.compile(rb"(\d+)/(\d+) \[")  # tqdm's progress bar


class CheckFailed(Exception):
    pass


def fail(message):
    print(message, file=sys.stderr)
    sys.exit(2)


def tail(path, n=20):
    with open(path, errors="replace") as f:
        return "".join(collections.deque(f, n))


def wait_ready(server, host, port, log_path, timeout):
    """Wait until the server answers a generation request. Returns the seconds it took."""
    start = time.time()
    while time.time() - start < timeout:
        if server.poll() is not None:
            raise CheckFailed(f"The server exited with code {server.returncode} while starting:\n{tail(log_path)}")
        try:
            with urllib.request.urlopen(f"http://{host}:{port}/health_generate", timeout=60) as response:
                if response.status == 200:
                    return time.time() - start
        except (urllib.error.URLError, OSError):
            pass
        if sys.stdout.isatty():
            print(f"\r  starting the server: {time.time() - start:.0f} s".ljust(40), end="", flush=True)
        time.sleep(5)
    raise CheckFailed(f"The server was not ready after {timeout} s:\n{tail(log_path)}")


def served_model(host, port):
    """The name the server gives its model, which the eval sends with each request."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=60) as response:
            return json.load(response)["data"][0]["id"]
    except (urllib.error.URLError, OSError, ValueError, LookupError) as error:
        raise CheckFailed(f"The server did not name its model at /v1/models: {error}")


def stop(server):
    """Stop the server and everything it started, and wait until all of it has exited.

    SGLang's scheduler can outlive the launcher by seconds while it frees the GPU.
    """
    try:
        os.killpg(server.pid, signal.SIGTERM)
        for _ in range(60):
            server.poll()  # reaps the shell, which would otherwise stay in the group
            os.killpg(server.pid, 0)  # raises once no process of the group is left
            time.sleep(1)
        os.killpg(server.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    server.wait()


def clone_gpt_oss(path):
    """Clone OpenAI's gpt-oss repository at GPT_OSS_COMMIT into path."""
    print(f"Cloning {GPT_OSS_REPO} at {GPT_OSS_COMMIT[:12]} into {path}.")
    partial = path + ".partial"
    shutil.rmtree(partial, ignore_errors=True)
    try:
        subprocess.run(["git", "clone", "--quiet", GPT_OSS_REPO, partial], check=True, timeout=600)
        subprocess.run(["git", "-C", partial, "checkout", "--quiet", GPT_OSS_COMMIT], check=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as error:
        fail(f"Could not clone {GPT_OSS_REPO}: {error}")
    os.rename(partial, path)


def check_eval(checkout, python):
    """Fail unless checkout has OpenAI's eval code and python can import what it needs. Returns the commit."""
    if not os.path.isfile(os.path.join(checkout, "gpt_oss", "evals", "gpqa_eval.py")):
        fail(f"{checkout} is not a checkout of {GPT_OSS_REPO}: it has no gpt_oss/evals.")
    try:
        imports = subprocess.run([python, "-c", "import " + ", ".join(EVAL_PACKAGES)], cwd=checkout,
                                 capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as error:
        fail(f"Could not run {python}: {error}")
    if imports.returncode:
        error = (imports.stderr.strip().splitlines() or [f"exit code {imports.returncode}"])[-1]
        fail(f"{python} can't run OpenAI's eval ({error}). It needs gpt-oss's eval packages, which a venv of\n"
             f"their own can hold:\n\n  python3 -m venv evals-venv && evals-venv/bin/pip install "
             f"{' '.join(EVAL_PACKAGES)}\n\nThen run this with evals-venv/bin/python, or pass --python.")
    try:
        commit = subprocess.run(["git", "-C", checkout, "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    return commit or "unknown"


def run_eval(argv, cwd, log_path, timeout, server):
    """Run the eval, writing its output to log_path and showing its progress.

    Stops it if it runs past timeout or the server exits, since the eval retries a
    failed request forever. Returns its exit code and why it was stopped, if it was.
    """
    # The eval's client wants an API key; the server takes any
    env = dict(os.environ, PYTHONUNBUFFERED="1", OPENAI_API_KEY="none")
    start, shown, stopped = time.time(), time.time(), []
    with open(log_path, "wb") as log, subprocess.Popen(
        argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, start_new_session=True
    ) as proc:
        def watch():
            while proc.poll() is None:
                if time.time() - start > timeout:
                    stopped.append(f"it ran past --eval-timeout, {timeout} s")
                elif server.poll() is not None:
                    stopped.append(f"the server exited with code {server.returncode}")
                if stopped:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    return
                time.sleep(5)

        threading.Thread(target=watch, daemon=True).start()
        for chunk in iter(lambda: proc.stdout.read1(65536), b""):
            log.write(chunk)
            log.flush()
            progress = PROGRESS.findall(chunk)
            if not progress:
                continue
            done, total = progress[-1]
            line = f"  {int(done)}/{int(total)} answered, {time.time() - start:.0f} s"
            if sys.stdout.isatty():
                print(f"\r{line}".ljust(40), end="", flush=True)
            elif time.time() - shown > 60:  # a log file gets a line a minute
                print(line, flush=True)
                shown = time.time()
        code = proc.wait()
    if sys.stdout.isatty():
        print()
    return code, stopped[0] if stopped else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", help="the command that starts the SGLang server on this system, in quotes")
    parser.add_argument("--effort", default="low", choices=list(REFERENCE), help="gpt-oss reasoning effort")
    parser.add_argument("--attempts", type=int, default=1,
                        help="times to ask each question (OpenAI's eval script asks 8)")
    parser.add_argument("--gpt-oss", help=f"a checkout of {GPT_OSS_REPO} (default: clone it into ~/.cache)")
    parser.add_argument("--python", default=sys.executable, help="the Python to run the eval with (default: this one)")
    parser.add_argument("--out", help="a new directory for the logs, the answers and the score")
    parser.add_argument("--start-timeout", type=int, default=1800, help="seconds to wait for the server")
    parser.add_argument("--eval-timeout", type=int, default=3600, help="seconds to allow the eval")
    parser.add_argument("--force", action="store_true", help="run even if other processes are using a GPU")
    parser.add_argument("--dry-run", action="store_true", help="print the commands that would run, and stop")
    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # keeps a log of this in order with the errors

    if args.attempts < 1:
        fail("--attempts has to be at least 1.")
    command = foreground(args.command)
    if len(LAUNCHER.findall(command)) != 1:
        fail("The command has to start SGLang once, with `python -m sglang.launch_server` or `sglang serve`.")
    if API_KEY.search(command):
        fail("The command sets --api-key, which the eval would need and this check does not pass on; leave it out.")
    missing = [(flag, why) for pattern, flag, why in FLAGS if not pattern.search(command)]
    added = ""
    if missing:
        match = LAUNCHER.search(command)
        command = f"{command[:match.end()]} {' '.join(f for f, _ in missing)}{command[match.end():]}"
        added = "\n(Added: " + "; ".join(f"{flag}, which {why}" for flag, why in missing) + ".)\n"
    port = int(PORT.findall(command)[-1]) if PORT.search(command) else 30000
    host = HOST.findall(command)[-1] if HOST.search(command) else "127.0.0.1"
    host = "127.0.0.1" if host in ("0.0.0.0", "localhost") else host
    checkout = os.path.abspath(args.gpt_oss) if args.gpt_oss else GPT_OSS_CACHE
    python = os.path.abspath(args.python) if os.sep in args.python else args.python

    machine, now = socket.gethostname(), time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    out = os.path.abspath(args.out or os.path.join(HERE, "results", f"sanity-accuracy-{machine}-{now}"))
    key = f"gpqa-{args.effort}"
    times = f"{args.attempts} time{'s' * (args.attempts > 1)}"

    def eval_argv(model):
        return [python, "-c", EVAL, f"http://{host}:{port}/v1", model, args.effort, str(args.attempts),
                os.path.join(out, f"{key}.json"), os.path.join(out, f"{key}-answers.json.gz")]

    if args.dry_run:
        cloned = not args.gpt_oss and not os.path.exists(checkout)
        source = f", cloned from {GPT_OSS_REPO} at {GPT_OSS_COMMIT[:12]}" if cloned else ""
        print(f"This would start the server with\n\n{command}\n{added}\nthen run OpenAI's GPQA eval in "
              f"{checkout}{source}, asking each question {times} at {args.effort} reasoning effort:\n\n"
              f"{shlex.join(eval_argv('MODEL'))}\n\nwith MODEL the name the server gives its model, logging to {out}.")
        return 0

    if not args.gpt_oss and not os.path.exists(checkout):
        clone_gpt_oss(checkout)
    commit = check_eval(checkout, python)
    busy = gpu_processes()
    if busy and not args.force:
        fail("Other processes are using a GPU:\n  " + "\n  ".join(busy) + "\nStop them, or pass --force.")
    with socket.socket() as s:
        if s.connect_ex((host, port)) == 0:
            fail(f"Something already listens on {host}:{port}; stop it first.")
    if os.path.exists(os.path.join(out, "scores.json")):
        fail(f"{out} already has scores; pass a new --out.")
    os.makedirs(out, exist_ok=True)

    start = time.time()
    server_log = os.path.join(out, "server.log")
    print(f"Starting the server with\n\n{command}\n{added}\nlogging to {server_log}.")
    log = open(server_log, "wb")
    server = subprocess.Popen(["bash", "-c", command], stdout=log, stderr=subprocess.STDOUT,
                              env=dict(os.environ, PYTHONUNBUFFERED="1"), start_new_session=True)
    try:
        ready = wait_ready(server, host, port, server_log, args.start_timeout)
        print(f"\r  the server was ready after {ready:.0f} s".ljust(40))
        model = served_model(host, port)
        print(f"\nAsking the {QUESTIONS} GPQA Diamond questions {times} each at {args.effort} reasoning effort, "
              f"with OpenAI's eval code at {commit[:12]} in {checkout}.")
        eval_log = os.path.join(out, f"{key}.log")
        began = time.time()
        code, stopped = run_eval(eval_argv(model), checkout, eval_log, args.eval_timeout, server)
        eval_minutes = (time.time() - began) / 60
    except CheckFailed as error:
        print(f"\n{error}", file=sys.stderr)
        return 2
    finally:
        stop(server)
        log.close()
    minutes, seconds = divmod(round(time.time() - start), 60)
    print(f"\nStopped the server after {minutes}m{seconds:02d}s.")
    left = gpu_processes()
    if left:
        print("  These are still on a GPU:\n    " + "\n    ".join(left))
    scores = os.path.join(out, f"{key}.json")
    if code or stopped or not os.path.exists(scores):
        print(f"\nCOULD NOT FINISH: the eval failed, {stopped or f'with exit code {code}'}:\n{tail(eval_log)}")
        return 2

    with open(scores) as f:
        score = 100 * json.load(f)["score"]
    with open(eval_log, errors="replace") as f:
        output = f.read()
    rejected = output.count("Bad Request Error")  # scored as wrong
    retried = output.count("Rate limit exception")  # errors and empty answers, asked again
    reference = REFERENCE[args.effort]
    p = reference / 100
    lowest = reference - max(5.0, 300 * math.sqrt(p * (1 - p) / QUESTIONS))
    passed = score >= lowest
    print(f"\nGPQA Diamond at {args.effort} reasoning effort, {QUESTIONS} questions x {args.attempts}, "
          f"in {eval_minutes:.1f} minutes:\n\n"
          f"  score {score:.1f}%, OpenAI's {reference:.1f}% ({score - reference:+.1f}), lowest that passes "
          f"{lowest:.1f}%")
    if retried:
        print(f"  {retried} answer{'s' * (retried > 1)} came back empty or failed, and {'were' if retried > 1 else 'was'} "
              "asked again")
    if rejected:
        print(f"  the server rejected {rejected} requests, which count as wrong answers; see {eval_log}")
    with open(os.path.join(out, "scores.json"), "w") as f:
        json.dump({"server_command": command, "model": model, "gpt_oss_commit": commit, "effort": args.effort,
                   "questions": QUESTIONS, "attempts": args.attempts, "score": round(score, 2), "openai": reference,
                   "lowest_pass": round(lowest, 2), "passed": passed, "rejected_requests": rejected,
                   "retries": retried, "eval_minutes": round(eval_minutes, 1)}, f, indent=2)
    if not passed:
        print(f"\nBELOW OpenAI's accuracy: {reference - score:.1f} points below, more than sampling explains.")
        return 1
    print("\nMATCHES OpenAI's accuracy: within what sampling explains.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
