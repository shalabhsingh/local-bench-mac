#!/usr/bin/env python3
"""
Engine comparison: vllm-metal vs Ollama.
Runs TTFT, throughput, and prefix-cache timing against either engine
using the OpenAI-compatible chat completions API.

Usage:
  # vllm (must already be running on port 8000)
  python compare.py --engine vllm --model /path/to/model --config "qwen3-8b-4bit-vllm"

  # Ollama (must already be running: ollama serve)
  python compare.py --engine ollama --model qwen3.5:9b --config "qwen3.5-9b-ollama"

Then diff two result JSONs to compare engines at matched quant level.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ENGINE_PORTS = {"vllm": 8000, "ollama": 11434}

SYSTEM_PROMPT = """\
You are a senior software engineer assistant. You help with code review, \
debugging, architecture decisions, and writing clean, well-tested code. \
You have access to the following tools: read_file, write_file, run_tests, \
search_codebase, get_git_diff, create_pr. Prefer minimal diffs. Do not \
refactor beyond what is asked."""

CODING_QUESTIONS = [
    "Write a Python function that finds all prime numbers up to n using the Sieve of Eratosthenes.",
    "Explain the difference between a mutex and a semaphore, with a code example in Go.",
    "Write a SQL query to find the top 5 customers by revenue in the last 30 days.",
    "How would you design a rate limiter for an API that allows 100 requests per minute per user?",
    "Debug this Python code: `def fib(n): return fib(n-1) + fib(n-2)` — what's wrong and fix it.",
]

REPEATED_PROMPT = "Write a Python function that reverses a linked list in place."


# ---------------------------------------------------------------------------
# HTTP helpers (no dependencies)
# ---------------------------------------------------------------------------

def _post(base_url: str, model: str, messages: list, max_tokens: int = 256, stream: bool = False) -> dict:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def _stream_ttft(base_url: str, model: str, messages: list, max_tokens: int = 128,
                 think: bool = False, debug: bool = False, use_ollama_api: bool = False) -> tuple[float, float, int]:
    """Returns (ttft_s, total_s, output_tokens).

    use_ollama_api: use /api/chat (Ollama native) instead of /v1/chat/completions.
    Ollama's native endpoint respects think=false; the OpenAI-compat one does not.
    """
    if use_ollama_api:
        payload = json.dumps({
            "model": model,
            "messages": messages,
            "think": think,
            "stream": True,
            "options": {"num_predict": max_tokens, "temperature": 0},
        }).encode()
        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        t0 = time.perf_counter()
        ttft = None
        tokens = 0
        with urllib.request.urlopen(req, timeout=120) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if debug:
                    print(f"  [chunk] {json.dumps(chunk)[:200]}", flush=True)
                content = (chunk.get("message") or {}).get("content") or ""
                if content and ttft is None:
                    ttft = time.perf_counter() - t0
                if content:
                    tokens += 1
                if chunk.get("done"):
                    break
        total = time.perf_counter() - t0
        return ttft or total, total, tokens

    # OpenAI-compatible path (vllm-metal)
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    try:
        _r = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code} from vllm: {body[:500]}") from None
    with _r as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            if debug:
                print(f"  [chunk] {json.dumps(chunk)[:200]}", flush=True)
            choices = chunk.get("choices") or []
            d = choices[0].get("delta", {}) if choices else {}
            content = d.get("content") or ""
            if content and ttft is None:
                ttft = time.perf_counter() - t0
            if content:
                tokens += 1
    total = time.perf_counter() - t0
    return ttft or total, total, tokens


def _rss_gb(pid: int) -> float | None:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
        return int(out) / 1_048_576 if out else None
    except Exception:
        return None


def _find_engine_pid(engine: str) -> int | None:
    pattern = "vllm.entrypoints" if engine == "vllm" else "ollama"
    try:
        out = subprocess.check_output(["pgrep", "-f", pattern], text=True).strip()
        return int(out.splitlines()[0]) if out else None
    except Exception:
        return None


class MemorySampler:
    def __init__(self, pid: int):
        self._pid = pid
        self._samples: list[float] = []
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._t.start()

    def stop(self) -> dict:
        self._stop.set()
        self._t.join()
        s = self._samples
        if not s:
            return {}
        return {"min_gb": min(s), "max_gb": max(s), "mean_gb": statistics.mean(s)}

    def _loop(self):
        while not self._stop.wait(1.0):
            v = _rss_gb(self._pid)
            if v is not None:
                self._samples.append(v)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ttft(base_url: str, model: str, runs: int, think: bool = False, debug: bool = False, use_ollama_api: bool = False) -> dict:
    print("\n--- TTFT ---")
    results = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": CODING_QUESTIONS[0]},
    ]
    for i in range(runs):
        ttft, total, tokens = _stream_ttft(base_url, model, messages, max_tokens=512, think=think, debug=debug, use_ollama_api=use_ollama_api)
        tps = tokens / total if total > 0 else 0
        print(f"  run {i+1}: ttft={ttft*1000:.0f}ms  total={total:.1f}s  {tokens}tok  {tps:.1f}tok/s")
        results.append({"ttft_ms": ttft * 1000, "total_s": total, "tokens": tokens, "tok_per_s": tps})
    ttfts = [r["ttft_ms"] for r in results]
    return {
        "runs": results,
        "median_ttft_ms": statistics.median(ttfts),
        "mean_ttft_ms": statistics.mean(ttfts),
    }


def test_throughput(base_url: str, model: str, runs: int, think: bool = False, debug: bool = False, use_ollama_api: bool = False) -> dict:
    print("\n--- Throughput ---")
    results = []
    for i, q in enumerate(CODING_QUESTIONS[:runs]):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        _, total, tokens = _stream_ttft(base_url, model, messages, max_tokens=512, think=think, debug=debug, use_ollama_api=use_ollama_api)
        tps = tokens / total if total > 0 else 0
        print(f"  q{i+1}: {tokens}tok in {total:.1f}s = {tps:.1f}tok/s")
        results.append({"question": i + 1, "tokens": tokens, "total_s": total, "tok_per_s": tps})
    tps_list = [r["tok_per_s"] for r in results]
    return {
        "runs": results,
        "median_tok_per_s": statistics.median(tps_list),
        "mean_tok_per_s": statistics.mean(tps_list),
    }


def test_prefix_cache(base_url: str, model: str, pairs: int = 4, think: bool = False, debug: bool = False, use_ollama_api: bool = False) -> dict:
    """
    Send the same prompt twice in a row. Second call should hit prefix cache
    (vllm) or show KV reuse (Ollama, via timing difference).
    Reports speedup ratio: first_ttft / second_ttft.
    """
    print("\n--- Prefix Cache (timing proxy) ---")
    results = []
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": REPEATED_PROMPT},
    ]
    for i in range(pairs):
        ttft1, _, _ = _stream_ttft(base_url, model, messages, max_tokens=64, think=think, debug=debug, use_ollama_api=use_ollama_api)
        time.sleep(0.5)
        ttft2, _, _ = _stream_ttft(base_url, model, messages, max_tokens=64, think=think, debug=debug, use_ollama_api=use_ollama_api)
        speedup = ttft1 / ttft2 if ttft2 > 0 else 0
        print(f"  pair {i+1}: first={ttft1*1000:.0f}ms  second={ttft2*1000:.0f}ms  speedup={speedup:.2f}x")
        results.append({"first_ttft_ms": ttft1 * 1000, "second_ttft_ms": ttft2 * 1000, "speedup": speedup})
    speedups = [r["speedup"] for r in results]
    return {
        "pairs": results,
        "median_speedup": statistics.median(speedups),
        "note": "speedup>1 means second call was faster (cache hit); Ollama may not expose hit rate directly",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vllm", "ollama"], required=True)
    ap.add_argument("--model", required=True, help="Model name/path as the engine expects it")
    ap.add_argument("--config", required=True, help="Short label for this run, e.g. 'qwen3-8b-4bit-vllm'")
    ap.add_argument("--port", type=int, help="Override default port (vllm=8000, ollama=11434)")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--test", choices=["ttft", "throughput", "prefix", "all"], default="all")
    ap.add_argument("--think", action="store_true", help="Enable Qwen3 thinking mode (disabled by default)")
    ap.add_argument("--debug", action="store_true", help="Print raw streaming chunks for diagnostics")
    args = ap.parse_args()

    port = args.port or ENGINE_PORTS[args.engine]
    base_url = f"http://localhost:{port}"

    # Resolve model name from engine if not overridden
    model = args.model
    if args.engine == "vllm" and model == "auto":
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as r:
            model = json.load(r)["data"][0]["id"]
    print(f"\nEngine : {args.engine}  port={port}")
    print(f"Model  : {model}")
    print(f"Config : {args.config}")

    pid = _find_engine_pid(args.engine)
    mem_sampler = MemorySampler(pid) if pid else None
    if mem_sampler:
        mem_sampler.start()
        print(f"Memory : sampling PID {pid}")
    else:
        print("Memory : PID not found, skipping RSS sampling")

    results = {}
    use_ollama_api = (args.engine == "ollama")
    if args.test in ("ttft", "all"):
        results["ttft"] = test_ttft(base_url, model, args.runs, think=args.think, debug=args.debug, use_ollama_api=use_ollama_api)
    if args.test in ("throughput", "all"):
        results["throughput"] = test_throughput(base_url, model, min(args.runs, len(CODING_QUESTIONS)), think=args.think, debug=args.debug, use_ollama_api=use_ollama_api)
    if args.test in ("prefix", "all"):
        results["prefix_cache"] = test_prefix_cache(base_url, model, pairs=args.runs, think=args.think, debug=args.debug, use_ollama_api=use_ollama_api)

    mem = mem_sampler.stop() if mem_sampler else {}

    out = {
        "config": args.config,
        "engine": args.engine,
        "model": model,
        "port": port,
        "timestamp": datetime.now().isoformat(),
        "memory_rss": mem,
        "results": results,
    }

    slug = f"{args.config.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    json_path = os.path.join(RESULTS_DIR, f"compare_{slug}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results → {json_path}")


class _Tee:
    def __init__(self, path):
        self._file = open(path, "w")
        self._stdout = sys.stdout
    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)
    def flush(self):
        self._stdout.flush()
        self._file.flush()
    def close(self):
        self._file.close()


if __name__ == "__main__":
    _slug = None
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            _slug = sys.argv[i + 1].replace(" ", "_")
    _ts = datetime.now().strftime("%Y%m%d_%H%M")
    _stdout_path = os.path.join(RESULTS_DIR, f"stdout_compare_{_slug or 'run'}_{_ts}.txt")
    _tee = _Tee(_stdout_path)
    sys.stdout = _tee
    print(f"  Stdout  → {_stdout_path}")
    try:
        main()
    finally:
        sys.stdout = _tee._stdout
        _tee.close()
