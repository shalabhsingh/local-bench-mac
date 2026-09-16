#!/usr/bin/env python3
"""
Benchmark script for vllm-metal local inference.
Measures TTFT, throughput, and prefix cache hit rate.

Usage:
  python bench.py                    # all tests
  python bench.py --test ttft        # TTFT only
  python bench.py --test prefix      # prefix cache only
  python bench.py --test throughput  # throughput only
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import statistics
import urllib.request
import urllib.error
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BASE_URL = "http://localhost:8000"


def get_model():
    """Fetch the model ID from the running vllm server."""
    try:
        with urllib.request.urlopen(f"{BASE_URL}/v1/models", timeout=5) as r:
            return json.load(r)["data"][0]["id"]
    except Exception as e:
        raise SystemExit(f"Cannot fetch model from {BASE_URL}/v1/models: {e}")


MODEL = None  # resolved at runtime

# A realistic coding-agent system prompt (~500 tokens) for prefix cache testing.
# The point: coding agents resend this on every call. Cache hit rate here is the finding.
SYSTEM_PROMPT = """You are an expert software engineer assistant. You help with code review,
debugging, refactoring, and implementation tasks. You have access to the following tools:

Tool: read_file
  Description: Read the contents of a file from the filesystem
  Parameters:
    path (string, required): Absolute path to the file to read
    offset (integer, optional): Line number to start reading from
    limit (integer, optional): Maximum number of lines to read

Tool: write_file
  Description: Write content to a file, creating it if it does not exist
  Parameters:
    path (string, required): Absolute path to write to
    content (string, required): Content to write

Tool: bash
  Description: Execute a bash command and return its output
  Parameters:
    command (string, required): The shell command to execute
    timeout (integer, optional): Timeout in seconds, default 30

Tool: search_code
  Description: Search for a pattern in the codebase using ripgrep
  Parameters:
    pattern (string, required): The regex pattern to search for
    path (string, optional): Directory to search in, defaults to current
    file_glob (string, optional): File pattern to limit search scope

When using tools, respond with a JSON object in this format:
{"tool": "<tool_name>", "parameters": {<key>: <value>}}

Always prefer reading existing code before modifying it. For bugs, identify the
root cause before suggesting a fix. For refactoring, ensure tests pass before
and after. Be concise — one-sentence explanations, no unnecessary commentary."""

CODING_QUESTIONS = [
    "There's a bug in my Python function that calculates fibonacci numbers recursively — it's extremely slow for n > 30. What's wrong and how do I fix it?",
    "I have a React component that re-renders on every keystroke even though the data hasn't changed. What's the likely cause?",
    "My SQL query with a JOIN on a 10M row table takes 45 seconds. The query uses WHERE on a non-indexed column. What should I check first?",
    "I'm getting a 'maximum recursion depth exceeded' error in Python. The stack trace shows the same function called 1000 times. What's happening?",
    "My git push is rejected with 'non-fast-forward'. I need to push my changes without losing the upstream commits. What's the safest approach?",
]


def post_stream(endpoint, payload):
    """POST to endpoint with streaming=True, yield (chunk, elapsed_since_start)."""
    data = json.dumps({**payload, "stream": True}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            yield json.loads(chunk), time.perf_counter() - start


def complete_streaming(prompt, max_tokens=200, system=None):
    """
    Run one streaming completion. Returns:
      ttft       — seconds to first non-empty token
      total_time — seconds to last token
      tokens     — number of output tokens
      text       — full generated text
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {"model": MODEL, "messages": messages, "max_tokens": max_tokens}
    ttft = None
    text = ""
    total_time = 0
    tokens = 0

    for chunk, elapsed in post_stream("/v1/chat/completions", payload):
        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
        if delta and ttft is None:
            ttft = elapsed
        if delta:
            text += delta
            tokens += 1
        total_time = elapsed

    return ttft, total_time, tokens, text


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_ttft(n_runs=3):
    """TTFT at 1 concurrent request, varied prompts (no prefix cache benefit)."""
    print("\n── TTFT (time to first token) ──────────────────────────────────")
    print(f"Runs: {n_runs}, concurrency: 1, no shared prefix\n")

    ttfts = []
    for i, question in enumerate(CODING_QUESTIONS[:n_runs]):
        ttft, total, tokens, _ = complete_streaming(question, max_tokens=150)
        tps = tokens / total if total > 0 else 0
        ttfts.append(ttft)
        print(f"  run {i+1}: TTFT={ttft*1000:.0f}ms  tokens/s={tps:.1f}  tokens={tokens}")

    print(f"\n  median TTFT : {statistics.median(ttfts)*1000:.0f} ms")
    print(f"  mean TTFT   : {statistics.mean(ttfts)*1000:.0f} ms")
    if len(ttfts) > 1:
        print(f"  stdev       : {statistics.stdev(ttfts)*1000:.0f} ms")
    return ttfts


def test_throughput(n_runs=3):
    """Tokens/sec at 1 concurrent request."""
    print("\n── Throughput (tokens/sec) ─────────────────────────────────────")
    print(f"Runs: {n_runs}, max_tokens=256, concurrency: 1\n")

    tps_list = []
    for i, question in enumerate(CODING_QUESTIONS[:n_runs]):
        ttft, total, tokens, _ = complete_streaming(question, max_tokens=256)
        tps = tokens / total if total > 0 else 0
        tps_list.append(tps)
        print(f"  run {i+1}: {tps:.1f} tok/s  ({tokens} tokens in {total:.2f}s)")

    print(f"\n  median tok/s: {statistics.median(tps_list):.1f}")
    return tps_list


def test_prefix_cache(n_pairs=3):
    """
    Prefix cache hit/miss comparison.
    Cold request: unique prompt, no shared prefix → measures baseline TTFT.
    Warm request: same system prompt sent again with a different question.
    If prefix caching works, warm TTFT should be measurably lower.
    """
    print("\n── Prefix Cache Hit Rate ───────────────────────────────────────")
    print(f"System prompt length: ~{len(SYSTEM_PROMPT.split())} words")
    print(f"Pairs: {n_pairs} (cold then warm per pair)\n")

    cold_ttfts = []
    warm_ttfts = []

    for i in range(n_pairs):
        q_cold = CODING_QUESTIONS[i % len(CODING_QUESTIONS)]
        q_warm = CODING_QUESTIONS[(i + 1) % len(CODING_QUESTIONS)]

        # Cold: send with system prompt, first time this prefix is seen
        ttft_cold, _, _, _ = complete_streaming(q_cold, max_tokens=100, system=SYSTEM_PROMPT)
        cold_ttfts.append(ttft_cold)
        print(f"  pair {i+1} cold (cache miss): TTFT={ttft_cold*1000:.0f}ms")

        # Warm: same system prompt, different user question — should hit prefix cache
        time.sleep(0.2)
        ttft_warm, _, _, _ = complete_streaming(q_warm, max_tokens=100, system=SYSTEM_PROMPT)
        warm_ttfts.append(ttft_warm)
        print(f"  pair {i+1} warm (cache hit?) : TTFT={ttft_warm*1000:.0f}ms  "
              f"speedup={ttft_cold/ttft_warm:.2f}x\n")

    median_cold = statistics.median(cold_ttfts)
    median_warm = statistics.median(warm_ttfts)
    speedup = median_cold / median_warm if median_warm > 0 else 0

    print(f"  median cold TTFT : {median_cold*1000:.0f} ms")
    print(f"  median warm TTFT : {median_warm*1000:.0f} ms")
    print(f"  median speedup   : {speedup:.2f}x")
    if speedup < 1.1:
        print("  NOTE: speedup <1.1x — prefix cache may not be active or KV budget too small")
    return cold_ttfts, warm_ttfts


def find_vllm_pid():
    """Find the main vllm server process pid via pgrep."""
    try:
        out = subprocess.check_output(["pgrep", "-f", "vllm.entrypoints"], text=True).strip()
        pids = [int(p) for p in out.splitlines() if p.strip()]
        return pids[0] if pids else None
    except Exception:
        return None


def _rss_gb(pid):
    """RSS of a process in GB via ps (no dependencies)."""
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
        return round(int(out) / 1024**2, 3)  # KB → GB
    except Exception:
        return None


class MemorySampler:
    """Sample vllm process RSS during a benchmark run using ps."""

    def __init__(self, pid=None):
        self.pid = pid
        self._samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if self.pid is None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _loop(self):
        while not self._stop.wait(1.0):
            v = _rss_gb(self.pid)
            if v is not None:
                self._samples.append(v)

    @property
    def peak_gb(self):
        return max(self._samples) if self._samples else None

    @property
    def mean_gb(self):
        return statistics.mean(self._samples) if self._samples else None


def sample_system_memory():
    """Unified memory snapshot via vm_stat + sysctl — no dependencies."""
    try:
        total_bytes = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True).strip())
        total_gb = total_bytes / 1024**3

        vm = subprocess.check_output(["vm_stat"], text=True)
        page_size = 16384  # Apple Silicon uses 16KB pages

        def pages(key):
            for line in vm.splitlines():
                if key in line:
                    return int(line.split(":")[1].strip().rstrip("."))
            return 0

        free       = pages("Pages free")
        active     = pages("Pages active")
        inactive   = pages("Pages inactive")
        wired      = pages("Pages wired down")
        compressed = pages("Pages occupied by compressor")

        used_gb      = (active + inactive + wired + compressed) * page_size / 1024**3
        available_gb = (free + inactive) * page_size / 1024**3
        return {
            "total_gb":     round(total_gb, 2),
            "used_gb":      round(used_gb, 2),
            "available_gb": round(available_gb, 2),
            "percent":      round(used_gb / total_gb * 100, 1),
        }
    except Exception:
        return {}


def print_header(config_label, startup_stats=None):
    print("=" * 60)
    print(f"  vllm-metal benchmark — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  config : {config_label}")
    print(f"  model  : {MODEL}")
    print(f"  server : {BASE_URL}")
    if startup_stats:
        print(f"  weights: {startup_stats.get('model_memory_gb', '?')} GB  "
              f"kv_budget: {startup_stats.get('kv_budget_gb', '?')} GB  "
              f"max_tokens: {startup_stats.get('max_tokens_cached', '?')}")
    sys_mem = sample_system_memory()
    if sys_mem:
        print(f"  system : {sys_mem['used_gb']}/{sys_mem['total_gb']} GB used "
              f"({sys_mem['percent']}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["ttft", "throughput", "prefix", "all"], default="all")
    parser.add_argument("--config", default="BF16-unquantized", help="Label for results (e.g. '4bit-MLX')")
    parser.add_argument("--runs", type=int, default=3)
    # Startup stats — copy from vllm serve log, recorded alongside benchmark results.
    # These change with each quant config and are the key memory story.
    parser.add_argument("--model-memory-gb", type=float, help="Weight footprint from startup log")
    parser.add_argument("--kv-budget-gb", type=float, help="KV cache budget from startup log")
    parser.add_argument("--max-tokens-cached", type=int, help="Max tokens cached from startup log")
    args = parser.parse_args()

    startup_stats = {}
    if args.model_memory_gb:
        startup_stats["model_memory_gb"] = args.model_memory_gb
    if args.kv_budget_gb:
        startup_stats["kv_budget_gb"] = args.kv_budget_gb
    if args.max_tokens_cached:
        startup_stats["max_tokens_cached"] = args.max_tokens_cached

    global MODEL
    MODEL = get_model()

    print_header(args.config, startup_stats)

    # Start memory sampler
    vllm_pid = find_vllm_pid()
    sampler = MemorySampler(pid=vllm_pid)
    sampler.start()
    mem_before = sample_system_memory()

    results = {}

    if args.test in ("ttft", "all"):
        results["ttft_ms"] = [t * 1000 for t in test_ttft(args.runs)]

    if args.test in ("throughput", "all"):
        results["throughput_tps"] = test_throughput(args.runs)

    if args.test in ("prefix", "all"):
        cold, warm = test_prefix_cache(args.runs)
        results["prefix_cold_ms"] = [t * 1000 for t in cold]
        results["prefix_warm_ms"] = [t * 1000 for t in warm]

    sampler.stop()
    mem_after = sample_system_memory()

    # Memory summary
    print("\n── Memory ──────────────────────────────────────────────────")
    if startup_stats:
        print(f"  weight footprint : {startup_stats.get('model_memory_gb', '?')} GB")
        print(f"  kv cache budget  : {startup_stats.get('kv_budget_gb', '?')} GB")
        print(f"  max tokens cached: {startup_stats.get('max_tokens_cached', '?')}")
    if mem_before:
        print(f"  system before    : {mem_before['used_gb']} GB used")
        print(f"  system after     : {mem_after['used_gb']} GB used")
    if sampler.peak_gb:
        print(f"  vllm peak RSS    : {sampler.peak_gb:.2f} GB")

    memory_stats = {
        **startup_stats,
        "system_mem_before": mem_before,
        "system_mem_after": mem_after,
    }
    if sampler.peak_gb:
        memory_stats["vllm_peak_rss_gb"] = round(sampler.peak_gb, 3)
        memory_stats["vllm_mean_rss_gb"] = round(sampler.mean_gb, 3)

    # Save raw results
    out = {
        "timestamp": datetime.now().isoformat(),
        "config": args.config,
        "model": MODEL,
        "vllm_metal_version": "0.27.1",
        "memory": memory_stats,
        "results": results,
    }
    slug = f"{args.config.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    json_path = os.path.join(RESULTS_DIR, f"results_{slug}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Results → {json_path}")


class _Tee:
    """Write stdout to both terminal and a file simultaneously."""
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
    # Parse config early to name the stdout file
    _slug = None
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            _slug = sys.argv[i + 1].replace(" ", "_")
    _ts = datetime.now().strftime("%Y%m%d_%H%M")
    _stdout_path = os.path.join(RESULTS_DIR, f"stdout_{_slug or 'bench'}_{_ts}.txt")
    _tee = _Tee(_stdout_path)
    sys.stdout = _tee
    print(f"  Stdout  → {_stdout_path}")
    try:
        main()
    finally:
        sys.stdout = _tee._stdout
        _tee.close()
