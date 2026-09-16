#!/usr/bin/env python3
"""
Download all 164 HumanEval problems from the HuggingFace datasets-server API
(reachable even when huggingface.co is blocked) and save to data/humaneval.jsonl.

Run once before eval_quality.py:
  python fetch_humaneval.py
"""

import gzip
import io
import json
import urllib.request
from pathlib import Path

OUT = Path(__file__).parent / "data" / "humaneval.jsonl"

# GitHub raw — the original OpenAI HumanEval release, no auth needed
SOURCES = [
    "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
    "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl",
]

def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()

def main():
    OUT.parent.mkdir(exist_ok=True)
    raw = None
    for url in SOURCES:
        try:
            print(f"Trying {url} ...")
            raw = _fetch(url)
            print(f"  downloaded {len(raw):,} bytes")
            break
        except Exception as e:
            print(f"  failed: {e}")

    if raw is None:
        raise RuntimeError("All sources failed. Check network connectivity.")

    # Decompress if gzipped
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)

    lines = [l for l in raw.decode().splitlines() if l.strip()]
    with open(OUT, "w") as f:
        for line in lines:
            f.write(line + "\n")

    print(f"Saved {len(lines)} problems → {OUT}")

if __name__ == "__main__":
    main()
