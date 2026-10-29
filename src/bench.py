"""Benchmark llama-server variants on this worker's own GPUs (endpoint env BENCHMARK=1).

Job input: {"benchmark": {"variants": [{"name": "tensor", "args": "-sm tensor"}, ...]}}

Each variant restarts llama-server with the worker's default arguments plus its
own, then measures what matters for an agent: decode speed, prefill of an
agent-sized prompt, an append-only follow-up turn (prompt cache) and a
code-rewrite task (where speculative decoding pays off). The weights stay in
the page cache between variants, so a restart reloads from RAM, not from the
network volume.
"""

import json
import logging
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from typing import Callable

import handler

LOAD_TIMEOUT = 1800  # seconds per variant; the first load may come from the network volume

# ~20K tokens of synthetic code: the size of an agent's system prompt plus repository context.
REPO = "\n".join(
    f"def fonction_{i}(x, y):\n    # calcule la valeur {i} du module {i % 37}\n    return (x * {i} + y) % {i + 7}\n"
    for i in range(1, 650)
)
# ~150 lines to rewrite almost verbatim: most output tokens already appear in the prompt.
EDIT_FILE = "\n".join(
    f"def etape_{i}(valeurs):\n    total = 0\n    for v in valeurs:\n        total += v * {i}\n    return total\n"
    for i in range(1, 31)
)
SYSTEM = {"role": "system", "content": "Tu es un agent de code. Voici le dépôt :\n" + REPO}


def _chat(body: dict, timeout: float = 900) -> dict:
    body = {"model": "bench", "chat_template_kwargs": {"thinking_mode": "chat"}, **body}
    req = urllib.request.Request(f"{handler.LLAMA_BASE_URL}/v1/chat/completions", method="POST",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _timings(out: dict) -> dict:
    keep = ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_per_second",
            "cache_n", "draft_n", "draft_n_accepted")
    t = out.get("timings", {})
    return {k: round(t[k], 1) if isinstance(t.get(k), float) else t.get(k) for k in keep if k in t}


def _wait_ready(proc: subprocess.Popen) -> float:
    started = time.monotonic()
    while time.monotonic() - started < LOAD_TIMEOUT:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {proc.returncode}")
        try:
            with urllib.request.urlopen(f"{handler.LLAMA_BASE_URL}/health", timeout=10) as resp:
                if resp.status == 200:
                    return time.monotonic() - started
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(3)
    raise RuntimeError(f"not ready after {LOAD_TIMEOUT} s")


def _measure() -> dict:
    res = {}
    out = _chat({"messages": [{"role": "user", "content": "Écris un texte d'environ 500 mots sur l'histoire de l'informatique."}],
                 "max_tokens": 700, "cache_prompt": False})
    res["decode"] = _timings(out)

    first = [SYSTEM, {"role": "user", "content": "Que retourne fonction_456(2, 3) ? Réponds juste avec le nombre."}]
    out = _chat({"messages": first, "max_tokens": 16, "cache_prompt": False})
    res["prefill_20k"] = _timings(out)
    res["prefill_20k"]["answer"] = out["choices"][0]["message"].get("content")  # expected 452

    # Prime the cache with the same turn, then send one more turn on top of it (append-only).
    out = _chat({"messages": first, "max_tokens": 16, "cache_prompt": True})
    follow = first + [{"role": "assistant", "content": out["choices"][0]["message"].get("content") or ""},
                      {"role": "user", "content": "Et fonction_77(1, 1) ? Juste le nombre."}]
    out = _chat({"messages": follow, "max_tokens": 16, "cache_prompt": True})
    res["followup_cached"] = _timings(out)
    res["followup_cached"]["answer"] = out["choices"][0]["message"].get("content")  # expected 78

    out = _chat({"messages": [{"role": "user", "content": "Réécris ce fichier en entier en renommant la variable "
                               "`total` en `somme`. Réponds uniquement avec le code.\n\n" + EDIT_FILE}],
                 "max_tokens": 1500, "cache_prompt": False})
    res["code_rewrite"] = _timings(out)
    return res


def run(variants: list[dict], base_argv: list[str], prewarm: Callable[[], None]) -> list[dict]:
    results = []
    for i, variant in enumerate(variants):
        name = variant.get("name", f"variant-{i}")
        argv = base_argv + shlex.split(variant.get("args", ""))
        log_path = f"/tmp/bench-{i}.log"
        logging.info("bench %s: %s", name, shlex.join(argv))
        result = {"name": name, "args": variant.get("args", "")}
        with open(log_path, "w") as log:
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
        try:
            result["load_s"] = round(_wait_ready(proc), 1)
            if i == 0:
                prewarm()  # Engram into the page cache once, so every variant is measured the same way
            result.update(_measure())
        except Exception as e:  # keep going: one failing variant must not hide the others
            result["error"] = str(e)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        with open(log_path) as log:
            lines = log.read().splitlines()
        result["memory"] = [l[-160:] for l in lines if "buffer size" in l][-8:]
        if "error" in result:
            result["log_tail"] = [l[-200:] for l in lines[-15:]]
        logging.info("bench %s: %s", name, json.dumps(result, ensure_ascii=False))
        results.append(result)
    return results
