"""The command that starts an SGLang server, as the sanity checks take it.

Both checks get that command as one string, the way a launch script runs it.
These helpers drop what would send it to the background, find where it starts
SGLang, and run something else there with the same environment and Python.
"""

import re
import subprocess

# How a command starts SGLang, and the parts of one that would send it to the background
LAUNCHER = re.compile(r"(?P<module>-m\s+sglang\.launch_server)\b|(?P<bin>\S*/)?\bsglang\s+serve\b")
NOHUP = re.compile(r"\bnohup\s+")
TRAILER = re.compile(r"\s*(?:\\|&|&>>?\s*\S+|\d*>>?\s*(?:&\d+|\"[^\"]*\"|'[^']*'|[^\s;&|]+))\s*$")


def foreground(command):
    """The command without nohup, or a redirect or & at its end, so its output comes here."""
    command = NOHUP.sub("", command).strip()
    while match := TRAILER.search(command):
        command = command[: match.start()]
    return command


def swap(command, python_args):
    """The command with Python running python_args where it would start SGLang."""
    match = LAUNCHER.search(command)
    python = "" if match["module"] else f"{match['bin'] or ''}python3 "
    return f"{command[:match.start()]}{python}{python_args}{command[match.end():]}"


def instead(command, python_args):
    """Like swap, but without the rest of the command: the server's flags are left out."""
    match = LAUNCHER.search(command)
    python = "" if match["module"] else f"{match['bin'] or ''}python3 "
    return f"{command[:match.start()]}{python}{python_args}"


def gpu_processes():
    """The processes on any GPU, as nvidia-smi lists them; [] if it can't tell."""
    try:
        listing = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in listing.splitlines() if line.strip()]
