"""Keep one worker warm for a coding session, then let the endpoint scale back to zero.

    python3 scripts/session.py on      # 1 active worker, wait until the model answers
    python3 scripts/session.py off     # back to 0 active workers (the worker stops after its idle timeout)
    python3 scripts/session.py status  # workers, jobs, balance and current spend

An agent sends the whole conversation on every step. If the worker scales to
zero between two steps, the next one pays a cold start *and* re-reads the whole
context, because the prompt cache lived on that worker. An active worker costs
the GPU price per hour even when idle, so turn it off when you are done.

Needs RUNPOD_API_KEY in the environment (or in a .env file passed with --env).
"""

import argparse
import json
import os
import sys
import time
import urllib.request

ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "oqjn0uboys2uoz")
WARMUP_TIMEOUT = 45 * 60  # seconds; a cold start on a fresh host reads ~240 GB from the network volume


def load_env_file(path: str) -> None:
    with open(path) as f:
        for line in f:
            if "=" in line and not line.lstrip().startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ.setdefault(key, value)


def call(method: str, url: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        url,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read() or b"{}")


def set_min_workers(n: int) -> None:
    endpoint = call("PATCH", f"https://api.runpod.io/v2/serverless/{ENDPOINT_ID}", {"workers": {"min": n}})
    print(f"active workers -> {endpoint['workers']['min']} (max {endpoint['workers']['max']})")


def warm_up() -> None:
    """Send a 1-token job: it only completes once llama-server has loaded the model."""
    started = time.monotonic()
    job = call("POST", f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run",
               {"input": {"messages": [{"role": "user", "content": "ok"}], "max_tokens": 1}})
    while time.monotonic() - started < WARMUP_TIMEOUT:
        status = call("GET", f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job['id']}")["status"]
        if status == "COMPLETED":
            print(f"model ready after {time.monotonic() - started:.0f} s")
            return
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            sys.exit(f"warm-up job {status}")
        print(f"  {time.monotonic() - started:5.0f} s  {status}", flush=True)
        time.sleep(15)
    sys.exit("warm-up timed out")


def status() -> None:
    health = call("GET", f"https://api.runpod.ai/v2/{ENDPOINT_ID}/health")
    endpoint = call("GET", f"https://api.runpod.io/v2/serverless/{ENDPOINT_ID}")
    me = call("POST", "https://api.runpod.io/graphql",
              {"query": "query { myself { clientBalance currentSpendPerHr } }"})["data"]["myself"]
    print(f"workers {health['workers']} | jobs {health['jobs']}")
    print(f"active workers {endpoint['workers']['min']}, idle timeout {endpoint['workers']['idleTimeout']} s")
    print(f"balance {me['clientBalance']:.2f} $, current spend {me['currentSpendPerHr']:.3f} $/h")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["on", "off", "status"])
    parser.add_argument("--env", help="file with RUNPOD_API_KEY=... (e.g. a .env)")
    args = parser.parse_args()
    if args.env:
        load_env_file(args.env)
    if "RUNPOD_API_KEY" not in os.environ:
        sys.exit("RUNPOD_API_KEY is not set")

    if args.action == "on":
        set_min_workers(1)
        warm_up()
    elif args.action == "off":
        set_min_workers(0)
    status()


if __name__ == "__main__":
    main()
