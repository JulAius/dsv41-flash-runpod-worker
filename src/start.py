"""Worker entrypoint: start llama-server and the RunPod job loop side by side.

RunPod stops a worker whose job loop has not started after about 8 minutes,
and a cold read of ~240 GB from the network volume takes longer than that.
So the job loop starts right away and the handler holds each job until
llama-server answers /health with 200. If llama-server dies, or is still not
ready after STARTUP_TIMEOUT, the whole worker exits so RunPod replaces it.

Once the model is loaded, the Engram shard is read once to fill the page
cache: llama.cpp maps it with MADV_RANDOM (no readahead), so otherwise every
row lookup during generation is a round trip to the network volume.

Everything is configured through environment variables (see README.md).
"""

import asyncio
import glob
import logging
import os
import re
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.request

import runpod

import bench
import handler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Layout produced by the `hf download ... --local-dir` command in README.md.
MODEL_DIR = os.getenv("MODEL_DIR", "/runpod-volume/models/dsv41-flash-abliterated")
VARIANT = os.getenv("VARIANT", "Mix-256G")
STARTUP_TIMEOUT = int(os.getenv("STARTUP_TIMEOUT", "3600"))  # seconds; loading ~240 GB from a network volume is slow
PREWARM_ENGRAM = os.getenv("PREWARM_ENGRAM", "on").lower() in ("on", "1", "true", "yes")
HEALTH_POLL_INTERVAL = 5  # seconds
READ_CHUNK = 64 * 1024 * 1024


def resolve_model_path() -> str:
    """MODEL_PATH if set, else the first GGUF shard of VARIANT under MODEL_DIR."""
    explicit = os.getenv("MODEL_PATH")
    if explicit:
        return explicit
    pattern = f"{MODEL_DIR}/{VARIANT}/*-00001-of-*.gguf"
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No GGUF matches {pattern}; set MODEL_PATH.")
    return matches[0]


def build_llama_args(model_path: str) -> list[str]:
    argv = [
        "llama-server",
        "--model", model_path,
        "--host", "127.0.0.1",
        "--port", handler.LLAMA_PORT,
        "--alias", os.getenv("SERVED_MODEL_NAME", "deepseek-v4.1-flash-abliterated"),
        "--n-gpu-layers", "999",
        # Agent clients send 15-30K tokens of system prompt and tools on every turn.
        "--ctx-size", os.getenv("CTX_SIZE", "131072"),
        "--parallel", os.getenv("PARALLEL", "1"),
        "--flash-attn", "on",
        "--jinja",
        # Reasoning stays in the history (llama-server's default): DeepSeek's encoder keeps it
        # whenever tools are declared, and an append-only prompt is what lets the prompt cache be
        # reused. Chat-only use can pass --no-reasoning-preserve through LLAMA_EXTRA_ARGS.
        # Engram: rows read from disk on demand instead of being kept resident (see README.md).
        "--load-mode", "mmap",
        "--lazy-mode", os.getenv("LAZY_MODE", "on"),
        # Sampling recommended by DeepSeek; each request can still override it.
        "--temp", "1.0",
        "--top-p", "1.0",
        "--top-k", "0",
        "--min-p", "0.0",
        # The Engram table is a CPU gather table; PR #30817's loader forces this anyway.
        "--override-tensor", "per_layer_token_embd.weight=CPU",
    ]
    argv += shlex.split(os.getenv("LLAMA_EXTRA_ARGS", ""))
    return argv


def available_memory() -> int:
    """Bytes the page cache can grow into: host MemAvailable, capped by the container's cgroup limit."""
    with open("/proc/meminfo") as f:
        meminfo = {line.split(":")[0]: int(line.split()[1]) * 1024 for line in f}
    available = meminfo["MemAvailable"]
    try:
        with open("/sys/fs/cgroup/memory.max") as f:
            limit = f.read().strip()
        if limit != "max":
            with open("/sys/fs/cgroup/memory.current") as f:
                available = min(available, int(limit) - int(f.read()))
    except OSError:
        pass  # no cgroup v2 limit visible
    return available


def prewarm_engram(model_path: str) -> None:
    """Read the Engram shard once so its lazily-read rows come from the page cache."""
    shards = glob.glob(re.sub(r"-\d{5}-of-(\d{5})\.gguf$", r"-*-of-\1.gguf", model_path))
    engram = max(shards, key=os.path.getsize)  # the Engram table is by far the largest shard
    size = os.path.getsize(engram)
    available = available_memory()
    if size > available:
        logging.warning("Not prewarming %s: %.0f GB > %.0f GB available", engram, size / 1e9, available / 1e9)
        return
    started = time.monotonic()
    buf = bytearray(READ_CHUNK)
    with open(engram, "rb", buffering=0) as f:
        while f.readinto(buf):
            pass
    elapsed = time.monotonic() - started
    logging.info("Prewarmed %s: %.0f GB in %.0f s (%.2f GB/s)", engram, size / 1e9, elapsed, size / 1e9 / elapsed)


def wait_until_ready(proc: subprocess.Popen, started: float, model_path: str) -> None:
    """Poll /health until the model is loaded, then let jobs through; kill llama-server on timeout."""
    url = f"{handler.LLAMA_BASE_URL}/health"
    while time.monotonic() < started + STARTUP_TIMEOUT:
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                if resp.status == 200:
                    logging.info("llama-server ready after %.0f s", time.monotonic() - started)
                    handler.llama_ready.set()
                    if PREWARM_ENGRAM:
                        prewarm_engram(model_path)
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass  # 503 while loading, or not listening yet
        time.sleep(HEALTH_POLL_INTERVAL)
    logging.error("llama-server not ready after %s s", STARTUP_TIMEOUT)
    proc.kill()  # exit_when_dead then stops the worker


def exit_when_dead(proc: subprocess.Popen) -> None:
    proc.wait()
    logging.error("llama-server exited with code %s; stopping the worker", proc.returncode)
    os._exit(1)


def benchmark_main(model_path: str) -> None:
    """BENCHMARK=1: no resident llama-server; each job benchmarks the variants it lists."""
    async def bench_handler(job: dict) -> dict:
        variants = (job.get("input") or {}).get("benchmark", {}).get("variants") or [{"name": "default"}]
        return {"results": await asyncio.to_thread(
            bench.run, variants, build_llama_args(model_path), lambda: prewarm_engram(model_path))}

    runpod.serverless.start({"handler": bench_handler, "concurrency_modifier": lambda _current: 1})


def main() -> None:
    model_path = resolve_model_path()
    logging.info("Memory available to the page cache: %.0f GB", available_memory() / 1e9)
    if os.getenv("BENCHMARK"):
        benchmark_main(model_path)
        return
    argv = build_llama_args(model_path)
    logging.info("Starting: %s", shlex.join(argv))
    proc = subprocess.Popen(argv)
    threading.Thread(target=exit_when_dead, args=(proc,), daemon=True).start()
    threading.Thread(target=wait_until_ready, args=(proc, time.monotonic(), model_path), daemon=True).start()

    # One job per llama-server slot: extra requests go to the RunPod queue
    # (and may start another worker) instead of piling up on a busy one.
    max_concurrency = int(os.getenv("MAX_CONCURRENCY", os.getenv("PARALLEL", "1")))
    runpod.serverless.start(
        {
            "handler": handler.handler,
            "concurrency_modifier": lambda _current: max_concurrency,
            "return_aggregate_stream": True,
        }
    )


if __name__ == "__main__":
    main()
