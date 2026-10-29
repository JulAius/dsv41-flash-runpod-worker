"""Proxy RunPod jobs to the local llama-server (OpenAI-compatible API).

Accepted job["input"] shapes:

1. What RunPod sends for https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1/...:
       {"openai_route": "/v1/chat/completions", "openai_input": {...}}
2. Shorthand for /run and /runsync (every key is passed to /v1/chat/completions):
       {"messages": [...], "max_tokens": 1024, "stream": false}

With "stream": true the SSE lines are yielded as they arrive; otherwise the
JSON response is yielded once.
"""

import asyncio
import logging
import os
import threading
from typing import Any, AsyncGenerator, Optional, Tuple

import aiohttp

LLAMA_PORT = os.getenv("LLAMA_PORT", "8080")
LLAMA_BASE_URL = f"http://127.0.0.1:{LLAMA_PORT}"
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "3600"))

# Set by start.py once llama-server has loaded the model. Jobs taken earlier wait for it.
llama_ready = threading.Event()

CHAT_ROUTE = "/v1/chat/completions"


def _normalize_job_input(job_input: dict) -> Tuple[str, str, Optional[dict]]:
    """Return (route, method, body) for any accepted job input shape."""
    if job_input.get("openai_input"):
        return job_input.get("openai_route") or CHAT_ROUTE, "POST", job_input["openai_input"]

    if job_input.get("openai_route"):
        # Bare route without a body (e.g. /v1/models) is proxied as a GET.
        return job_input["openai_route"], "GET", None

    if "messages" in job_input:
        return CHAT_ROUTE, "POST", job_input

    raise ValueError("Job input must contain openai_input (+openai_route) or messages.")


def _error(message: str) -> dict:
    return {"error": {"message": message, "type": "worker_error", "code": None}}


async def handler(job: dict) -> AsyncGenerator[Any, None]:
    try:
        route, method, body = _normalize_job_input(job.get("input") or {})
    except ValueError as e:
        yield _error(str(e))
        return

    # start.py exits the worker if the model never loads, so this cannot spin forever.
    while not llama_ready.is_set():
        await asyncio.sleep(2)

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(method, f"{LLAMA_BASE_URL}{route}", json=body) as resp:
                if resp.status >= 400:
                    detail = await resp.text()
                    logging.error("llama-server %s %s returned HTTP %s: %s", method, route, resp.status, detail)
                    yield _error(f"llama-server returned HTTP {resp.status}: {detail}")
                    return

                if isinstance(body, dict) and body.get("stream") is True:
                    # Line by line so a multi-byte UTF-8 character is never split across chunks.
                    async for line in resp.content:
                        yield line.decode("utf-8")
                else:
                    yield await resp.json(content_type=None)
    except aiohttp.ClientError as e:
        logging.exception("Request to llama-server failed")
        yield _error(f"Request to llama-server failed: {e}")
