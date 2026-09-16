#!/usr/bin/env python3
"""
End-to-end benchmark: inference performance + quality (HumanEval) for two
model/engine combos side by side.

Usage:
  # Start your engines first, then:
  python run_benchmark.py \\
    --model-a "vllm:auto" \\
    --model-b "ollama:qwen3.5:9b" \\
    --label-a "Qwen3.5-9B vllm-MLX4bit" \\
    --label-b "Qwen3.5-9B Ollama-Q4KM" \\
    --runs 3 \\
    --quality-limit 20

  Engine spec format: "<engine>:<model>"
    vllm:auto          — queries /v1/models to resolve the loaded model name
    vllm:/path/to/model
    ollama:qwen3.5:9b

  Both engines must already be running before invoking this script.
  vllm: VLLM_METAL_MEMORY_FRACTION=0.95 vllm serve <model> --enable-prefix-caching
  Ollama: ollama serve  (then: ollama pull <model>)
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ENGINE_PORTS = {"vllm": 8000, "ollama": 11434}

SYSTEM_PROMPT = """\
You are a senior software engineer assistant. You help with code review, \
debugging, architecture decisions, and writing clean, well-tested code. \
You have access to the following tools: read_file, write_file, run_tests, \
search_codebase, get_git_diff, create_pr. Prefer minimal diffs. Do not \
refactor beyond what is asked."""

INFERENCE_QUESTIONS = [
    "Write a Python function that finds all prime numbers up to n using the Sieve of Eratosthenes.",
    "Explain the difference between a mutex and a semaphore, with a code example in Go.",
    "Write a SQL query to find the top 5 customers by revenue in the last 30 days.",
    "How would you design a rate limiter for an API that allows 100 requests per minute per user?",
    "Debug this Python code: `def fib(n): return fib(n-1) + fib(n-2)` — what's wrong and fix it.",
]
CACHE_PROMPT = "Write a Python function that reverses a linked list in place."


# ---------------------------------------------------------------------------
# HTTP / engine helpers
# ---------------------------------------------------------------------------

def resolve_model(engine: str, model: str, port: int) -> str:
    if model != "auto":
        return model
    base = f"http://localhost:{port}"
    with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as r:
        return json.load(r)["data"][0]["id"]


def _stream_vllm(base_url: str, model: str, messages: list, max_tokens: int = 512) -> tuple[float, float, int]:
    payload = json.dumps({
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "stream": True, "temperature": 0,
    }).encode()
    req = urllib.request.Request(f"{base_url}/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        conn = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"vllm HTTP {e.code}: {e.read().decode()[:300]}") from None
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    with conn as r:
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
            content = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content") or ""
            if content and ttft is None:
                ttft = time.perf_counter() - t0
            if content:
                tokens += 1
    total = time.perf_counter() - t0
    return ttft or total, total, tokens


def _stream_ollama(base_url: str, model: str, messages: list, max_tokens: int = 512) -> tuple[float, float, int]:
    payload = json.dumps({
        "model": model, "messages": messages, "think": False, "stream": True,
        "options": {"num_predict": max_tokens, "temperature": 0},
    }).encode()
    req = urllib.request.Request(f"{base_url}/api/chat", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
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
            content = (chunk.get("message") or {}).get("content") or ""
            if content and ttft is None:
                ttft = time.perf_counter() - t0
            if content:
                tokens += 1
            if chunk.get("done"):
                break
    total = time.perf_counter() - t0
    return ttft or total, total, tokens


def stream(engine: str, base_url: str, model: str, messages: list, max_tokens: int = 512):
    if engine == "ollama":
        return _stream_ollama(base_url, model, messages, max_tokens)
    return _stream_vllm(base_url, model, messages, max_tokens)


# ---------------------------------------------------------------------------
# Inference benchmark
# ---------------------------------------------------------------------------

def run_inference(engine: str, base_url: str, model: str, runs: int, label: str) -> dict:
    print(f"\n{'='*55}")
    print(f"  Inference: {label}")
    print(f"{'='*55}")

    # TTFT
    print("\n  [TTFT]")
    ttft_results = []
    msgs = [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": INFERENCE_QUESTIONS[0]}]
    for i in range(runs):
        ttft, total, toks = stream(engine, base_url, model, msgs)
        tps = toks / total if total > 0 else 0
        print(f"    run {i+1}: {ttft*1000:.0f}ms ttft  {toks}tok  {tps:.1f}tok/s")
        ttft_results.append({"ttft_ms": ttft * 1000, "tok_per_s": tps, "tokens": toks})

    # Throughput
    print("\n  [Throughput]")
    tput_results = []
    for i, q in enumerate(INFERENCE_QUESTIONS[:runs]):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
        _, total, toks = stream(engine, base_url, model, msgs)
        tps = toks / total if total > 0 else 0
        print(f"    q{i+1}: {toks}tok  {tps:.1f}tok/s")
        tput_results.append({"tok_per_s": tps, "tokens": toks})

    # Prefix cache
    print("\n  [Prefix cache]")
    cache_results = []
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": CACHE_PROMPT}]
    for i in range(runs):
        t1, _, _ = stream(engine, base_url, model, msgs, max_tokens=64)
        time.sleep(0.5)
        t2, _, _ = stream(engine, base_url, model, msgs, max_tokens=64)
        speedup = t1 / t2 if t2 > 0 else 0
        print(f"    pair {i+1}: {t1*1000:.0f}ms → {t2*1000:.0f}ms  {speedup:.2f}x")
        cache_results.append({"first_ms": t1 * 1000, "second_ms": t2 * 1000, "speedup": speedup})

    ttfts = [r["ttft_ms"] for r in ttft_results]
    tputs = [r["tok_per_s"] for r in tput_results]
    speedups = [r["speedup"] for r in cache_results]

    summary = {
        "median_ttft_ms": statistics.median(ttfts),
        "median_tok_per_s": statistics.median(tputs),
        "median_cache_speedup": statistics.median(speedups),
    }
    print(f"\n  Summary: ttft={summary['median_ttft_ms']:.0f}ms  "
          f"tput={summary['median_tok_per_s']:.1f}tok/s  "
          f"cache={summary['median_cache_speedup']:.2f}x")
    return {"ttft": ttft_results, "throughput": tput_results,
            "prefix_cache": cache_results, "summary": summary}


# ---------------------------------------------------------------------------
# Quality benchmark (HumanEval via lm-eval)
# ---------------------------------------------------------------------------

def run_quality(engine: str, port: int, model: str, label: str, limit: int, output_dir: str) -> dict:
    """Run HumanEval subset via eval_quality.py against a running engine."""
    print(f"\n{'='*55}")
    print(f"  Quality (HumanEval@{limit}): {label}")
    print(f"{'='*55}")

    slug = label.replace(" ", "_").replace("/", "-")
    eval_script = os.path.join(os.path.dirname(__file__), "eval_quality.py")
    cmd = [
        sys.executable, eval_script,
        "--engine", engine,
        "--model", model if model != "auto" else "auto",
        "--config", slug,
        "--port", str(port),
        "--limit", str(limit),
    ]
    print(f"  Running eval_quality.py --limit {limit}\n")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        return {"error": f"eval_quality exit code {result.returncode}"}

    # Find the latest quality result JSON
    result_files = sorted(Path(RESULTS_DIR).glob(f"quality_{slug}*.json"))
    if not result_files:
        return {"error": "no quality result file found"}
    data = json.loads(result_files[-1].read_text())
    return {"pass_at_1": data["pass_at_1"], "passed": data["passed"], "limit": data["limit"]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_spec(spec: str) -> tuple[str, str]:
    """'vllm:auto' → ('vllm', 'auto');  'ollama:qwen3.5:9b' → ('ollama', 'qwen3.5:9b')"""
    idx = spec.index(":")
    return spec[:idx], spec[idx + 1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-a", required=True, help="Engine:model spec for A, e.g. vllm:auto")
    ap.add_argument("--model-b", required=True, help="Engine:model spec for B, e.g. ollama:qwen3.5:9b")
    ap.add_argument("--label-a", default="Model-A", help="Human-readable label for A")
    ap.add_argument("--label-b", default="Model-B", help="Human-readable label for B")
    ap.add_argument("--port-a", type=int, help="Override port for A")
    ap.add_argument("--port-b", type=int, help="Override port for B")
    ap.add_argument("--runs", type=int, default=3, help="Inference test repetitions (default 3)")
    ap.add_argument("--tests", default="inference,quality", help="Comma-separated: inference,quality")
    ap.add_argument("--quality-limit", type=int, default=20, help="HumanEval problems to run (default 20)")
    ap.add_argument("--skip-quality-for", choices=["a", "b", "none"], default="none",
                    help="Skip quality eval for one side (useful if one engine is down)")
    args = ap.parse_args()

    tests = [t.strip() for t in args.tests.split(",")]
    engine_a, model_spec_a = parse_spec(args.model_a)
    engine_b, model_spec_b = parse_spec(args.model_b)
    port_a = args.port_a or ENGINE_PORTS[engine_a]
    port_b = args.port_b or ENGINE_PORTS[engine_b]
    base_a = f"http://localhost:{port_a}"
    base_b = f"http://localhost:{port_b}"

    model_a = resolve_model(engine_a, model_spec_a, port_a)
    model_b = resolve_model(engine_b, model_spec_b, port_b)

    print(f"\nBenchmark: {args.label_a}  vs  {args.label_b}")
    print(f"  A: {engine_a} / {model_a} (:{port_a})")
    print(f"  B: {engine_b} / {model_b} (:{port_b})")
    print(f"  Tests: {tests}  runs={args.runs}  quality_limit={args.quality_limit}")

    slug = datetime.now().strftime("%Y%m%d_%H%M")
    out = {
        "timestamp": datetime.now().isoformat(),
        "label_a": args.label_a, "label_b": args.label_b,
        "engine_a": engine_a, "model_a": model_a,
        "engine_b": engine_b, "model_b": model_b,
        "results_a": {}, "results_b": {},
    }
    quality_dir = os.path.join(RESULTS_DIR, f"lmeval_{slug}")

    if "inference" in tests:
        out["results_a"]["inference"] = run_inference(engine_a, base_a, model_a, args.runs, args.label_a)
        out["results_b"]["inference"] = run_inference(engine_b, base_b, model_b, args.runs, args.label_b)

    if "quality" in tests:
        os.makedirs(quality_dir, exist_ok=True)
        if args.skip_quality_for != "a":
            out["results_a"]["quality"] = run_quality(engine_a, port_a, model_a, args.label_a, args.quality_limit, quality_dir)
        if args.skip_quality_for != "b":
            out["results_b"]["quality"] = run_quality(engine_b, port_b, model_b, args.label_b, args.quality_limit, quality_dir)

    # Print comparison table
    print(f"\n{'='*55}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Metric':<28} {'A':>10} {'B':>10}")
    print(f"  {'-'*48}")
    if "inference" in tests:
        sa = out["results_a"]["inference"]["summary"]
        sb = out["results_b"]["inference"]["summary"]
        print(f"  {'TTFT (ms)':<28} {sa['median_ttft_ms']:>9.0f} {sb['median_ttft_ms']:>9.0f}")
        print(f"  {'Throughput (tok/s)':<28} {sa['median_tok_per_s']:>9.1f} {sb['median_tok_per_s']:>9.1f}")
        print(f"  {'Cache speedup':<28} {sa['median_cache_speedup']:>9.2f} {sb['median_cache_speedup']:>9.2f}")
    if "quality" in tests:
        qa = out["results_a"].get("quality", {})
        qb = out["results_b"].get("quality", {})
        print(f"  {'HumanEval pass@1':<28} {str(qa.get('pass_at_1','—')):>10} {str(qb.get('pass_at_1','—')):>10}")
    print(f"\n  A = {args.label_a}")
    print(f"  B = {args.label_b}")

    json_path = os.path.join(RESULTS_DIR, f"benchmark_{slug}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Full results → {json_path}")


class _Tee:
    def __init__(self, path):
        self._file = open(path, "w")
        self._stdout = sys.stdout
    def write(self, data): self._stdout.write(data); self._file.write(data)
    def flush(self): self._stdout.flush(); self._file.flush()
    def close(self): self._file.close()


if __name__ == "__main__":
    _ts = datetime.now().strftime("%Y%m%d_%H%M")
    _stdout_path = os.path.join(RESULTS_DIR, f"stdout_benchmark_{_ts}.txt")
    _tee = _Tee(_stdout_path)
    sys.stdout = _tee
    print(f"  Stdout → {_stdout_path}")
    try:
        main()
    finally:
        sys.stdout = _tee._stdout
        _tee.close()
