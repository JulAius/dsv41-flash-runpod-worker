"""Worker entrypoint: start llama-server, wait for the model to load, then pull RunPod jobs.

Same pattern as runpod-workers/worker-vllm: the RunPod job loop only starts
once llama-server answers /health with 200, so no job is taken before the
weights are in VRAM. If llama-server dies later, the whole worker exits so
RunPod replaces it instead of keeping a broken worker in the pool.

Everything is configured through environment variables (see README.md).
"""

import glob
import logging
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import runpod

import handler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Layout produced by the `hf download ... --local-dir` command in README.md.
MODEL_DIR = os.getenv("MODEL_DIR", "/runpod-volume/models/dsv41-flash-abliterated")
VARIANT = os.getenv("VARIANT", "Mix-256G")
STARTUP_TIMEOUT = int(os.getenv("STARTUP_TIMEOUT", "3600"))  # seconds; loading ~240 GB from a network volume is slow
HEALTH_POLL_INTERVAL = 5  # seconds


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
        "--ctx-size", os.getenv("CTX_SIZE", "32768"),
        "--parallel", os.getenv("PARALLEL", "1"),
        "--flash-attn", "on",
        "--jinja",
        # Drop earlier turns' thinking, like DeepSeek's reference encoder.
        "--no-reasoning-preserve",
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


def wait_until_ready(proc: subprocess.Popen) -> None:
    """Poll /health until the model is loaded; exit if llama-server dies or the timeout expires."""
    url = f"{handler.LLAMA_BASE_URL}/health"
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            logging.error("llama-server exited during startup with code %s", proc.returncode)
            sys.exit(1)
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError):
            pass  # 503 while loading, or not listening yet
        time.sleep(HEALTH_POLL_INTERVAL)
    logging.error("llama-server not ready after %s s", STARTUP_TIMEOUT)
    proc.kill()
    sys.exit(1)


def exit_when_dead(proc: subprocess.Popen) -> None:
    proc.wait()
    logging.error("llama-server exited with code %s; stopping the worker", proc.returncode)
    os._exit(1)


def main() -> None:
    argv = build_llama_args(resolve_model_path())
    logging.info("Starting: %s", shlex.join(argv))
    proc = subprocess.Popen(argv)

    started = time.monotonic()
    wait_until_ready(proc)
    logging.info("llama-server ready after %.0f s", time.monotonic() - started)
    threading.Thread(target=exit_when_dead, args=(proc,), daemon=True).start()

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
