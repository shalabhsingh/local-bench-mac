#!/usr/bin/env python3
"""
Quantization script for Qwen3-8B on Apple Silicon.

Produces:
  - MLX 4-bit   (naive, fast)        → ~/models/Qwen3-8B-4bit-mlx
  - MLX 8-bit   (naive, fast)        → ~/models/Qwen3-8B-8bit-mlx
  - AWQ W4A16   (calibrated, slow)   → ~/models/Qwen3-8B-awq-4bit

Note: AWQ 8-bit is not supported by AutoAWQ — the GEMM kernel is 4-bit only.
This is expected: AWQ was designed to recover quality lost at 4-bit; at 8-bit
naive rounding error is already small enough that calibration adds little value.

Usage:
  python quantize.py --method mlx4     # ~15s
  python quantize.py --method mlx8     # ~15s
  python quantize.py --method awq4     # ~60-90 min on CPU
  python quantize.py --method all      # all three
"""

import argparse
import time
from pathlib import Path

SRC  = Path.home() / "models" / "Qwen--Qwen3-8B"
OUT  = Path.home() / "models"


def quantize_mlx(bits: int):
    from mlx_lm import convert
    out = OUT / f"Qwen3-8B-{bits}bit-mlx"
    print(f"\n── MLX {bits}-bit → {out}")
    t = time.time()
    convert.convert(
        hf_path=str(SRC),
        mlx_path=str(out),
        quantize=True,
        q_bits=bits,
    )
    print(f"   done in {time.time()-t:.0f}s")


# Coding-relevant calibration samples — used instead of the default Pile dataset
# (which requires a HuggingFace download). Using coding data is more principled
# for a coding-agent benchmark: protects weights that matter for code tasks.
CALIB_DATA = [
    "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
    "class LRUCache:\n    def __init__(self, capacity):\n        self.capacity = capacity\n        self.cache = {}\n        self.order = []\n    def get(self, key):\n        if key not in self.cache:\n            return -1\n        self.order.remove(key)\n        self.order.append(key)\n        return self.cache[key]\n    def put(self, key, value):\n        if key in self.cache:\n            self.order.remove(key)\n        elif len(self.cache) >= self.capacity:\n            oldest = self.order.pop(0)\n            del self.cache[oldest]\n        self.cache[key] = value\n        self.order.append(key)",
    "import asyncio\nasync def fetch_all(urls):\n    async with aiohttp.ClientSession() as session:\n        tasks = [fetch(session, url) for url in urls]\n        return await asyncio.gather(*tasks, return_exceptions=True)",
    "SELECT u.id, u.name, COUNT(o.id) as order_count, SUM(o.total) as revenue\nFROM users u\nLEFT JOIN orders o ON u.id = o.user_id\nWHERE o.created_at >= NOW() - INTERVAL '30 days'\nGROUP BY u.id, u.name\nHAVING COUNT(o.id) > 5\nORDER BY revenue DESC;",
    "const useDebounce = (fn, delay) => {\n  const timeoutRef = useRef(null);\n  return useCallback((...args) => {\n    clearTimeout(timeoutRef.current);\n    timeoutRef.current = setTimeout(() => fn(...args), delay);\n  }, [fn, delay]);\n};",
    "func (s *Server) handleRequest(w http.ResponseWriter, r *http.Request) {\n\tctx, cancel := context.WithTimeout(r.Context(), 30*time.Second)\n\tdefer cancel()\n\tresult, err := s.service.Process(ctx, r.Body)\n\tif err != nil {\n\t\thttp.Error(w, err.Error(), http.StatusInternalServerError)\n\t\treturn\n\t}\n\tjson.NewEncoder(w).Encode(result)\n}",
    "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)\n\ndef merge(left, right):\n    result = []\n    i = j = 0\n    while i < len(left) and j < len(right):\n        if left[i] <= right[j]:\n            result.append(left[i]); i += 1\n        else:\n            result.append(right[j]); j += 1\n    return result + left[i:] + right[j:]",
    "interface Repository<T> {\n  findById(id: string): Promise<T | null>;\n  findAll(filter?: Partial<T>): Promise<T[]>;\n  save(entity: T): Promise<T>;\n  delete(id: string): Promise<void>;\n}\n\nclass UserRepository implements Repository<User> {\n  constructor(private db: Database) {}\n  async findById(id: string) {\n    return this.db.query('SELECT * FROM users WHERE id = $1', [id]);\n  }\n}",
    "You are a code review assistant. Review the following Python function for bugs, performance issues, and style violations. Be specific about line numbers and explain the impact of each issue you find.",
    "Explain the difference between process memory and GPU VRAM in the context of serving large language models. Why does quantization reduce memory usage, and what are the tradeoffs at 4-bit vs 8-bit precision?",
    "Given a React component that re-renders on every keystroke, diagnose the likely causes and provide a fix using useMemo and useCallback. Show before and after code.",
    "Write a Python context manager that measures and logs execution time and peak memory usage for any block of code.",
    "Debug this SQL query that returns duplicate rows when joining three tables. Explain why the duplicates occur and how to fix them without using DISTINCT.",
    "Implement a thread-safe singleton pattern in Python. Explain why the naive implementation fails under concurrency.",
    "You are helping a software engineer debug a production incident. The service is returning 500 errors intermittently. The logs show a database connection pool exhaustion. Walk through the debugging steps.",
    "Refactor this deeply nested callback code to use async/await. Preserve error handling and ensure all promises are properly awaited.",
]


def quantize_awq(bits: int):
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    out = OUT / f"Qwen3-8B-awq-{bits}bit"
    print(f"\n── AWQ W{bits}A16 (calibrated, local calib data) → {out}")
    print("   Running on CPU — expect 60–90 min. Don't close the terminal.\n")

    t = time.time()
    tokenizer = AutoTokenizer.from_pretrained(str(SRC))
    model = AutoAWQForCausalLM.from_pretrained(str(SRC), device_map="cpu")
    model.quantize(tokenizer, quant_config={
        "zero_point": True,
        "q_group_size": 128,
        "w_bit": bits,
        "version": "GEMM",
    }, calib_data=CALIB_DATA)
    model.save_quantized(str(out))
    tokenizer.save_pretrained(str(out))

    # AutoAWQ omits some fields from config.json that mlx-lm requires.
    # Patch them back from the source config.
    import json as _json
    src_cfg = _json.load(open(str(SRC / "config.json")))
    awq_cfg_path = out / "config.json"
    awq_cfg = _json.load(open(str(awq_cfg_path)))
    for key in ("rope_theta", "rope_scaling", "torch_dtype"):
        if key in src_cfg and key not in awq_cfg:
            awq_cfg[key] = src_cfg[key]
    _json.dump(awq_cfg, open(str(awq_cfg_path), "w"), indent=2)
    print("   patched config.json (rope_theta, rope_scaling, torch_dtype)")

    print(f"\n   done in {(time.time()-t)/60:.1f} min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["mlx4", "mlx8", "awq4", "all"],
                        required=True)
    args = parser.parse_args()

    if args.method in ("mlx4", "all"):
        quantize_mlx(4)
    if args.method in ("mlx8", "all"):
        quantize_mlx(8)
    if args.method in ("awq4", "all"):
        quantize_awq(4)

    print("\nDone. Serve with:")
    paths = {
        "mlx4": "~/models/Qwen3-8B-4bit-mlx",
        "mlx8": "~/models/Qwen3-8B-8bit-mlx",
        "awq4": "~/models/Qwen3-8B-awq-4bit",
        "all":  "~/models/Qwen3-8B-{4bit-mlx,8bit-mlx,awq-4bit}",
    }
    print(f"  vllm serve {paths[args.method]} --enable-prefix-caching")


if __name__ == "__main__":
    main()
