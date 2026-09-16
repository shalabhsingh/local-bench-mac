#!/usr/bin/env python3
"""
HumanEval pass@1 quality benchmark for a running vllm or Ollama engine.
Reads data/humaneval.jsonl (produced by fetch_humaneval.py).

Usage:
  # vllm (server running on :8000)
  python eval_quality.py --engine vllm --model auto --config "qwen3.5-9b-vllm" --limit 20

  # Ollama
  python eval_quality.py --engine ollama --model qwen3.5:9b --config "qwen3.5-9b-ollama" --limit 20
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
DATA_FILE = Path(__file__).parent / "data" / "humaneval.jsonl"
ENGINE_PORTS = {"vllm": 8000, "ollama": 11434}

# Stop tokens: end generation at the next top-level function/class definition
STOP = ["\ndef ", "\nclass ", "\n# "]


def _call_vllm(base_url: str, model: str, prompt: str) -> str:
    """Raw text completion via /v1/completions — most natural for HumanEval."""
    payload = json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": 512, "temperature": 0, "stop": STOP,
    }).encode()
    req = urllib.request.Request(f"{base_url}/v1/completions", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.load(r)
        return resp["choices"][0]["text"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"vllm HTTP {e.code}: {e.read().decode()[:200]}") from None


def _call_ollama(base_url: str, model: str, prompt: str) -> str:
    """Raw text completion via /api/generate — avoids chat-mode thinking overhead."""
    payload = json.dumps({
        "model": model, "prompt": prompt, "think": False, "stream": False,
        "options": {"num_predict": 512, "temperature": 0, "stop": STOP},
    }).encode()
    req = urllib.request.Request(f"{base_url}/api/generate", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        resp = json.load(r)
    return resp["response"]


def generate(engine: str, base_url: str, model: str, prompt: str) -> str:
    if engine == "ollama":
        return _call_ollama(base_url, model, prompt)
    return _call_vllm(base_url, model, prompt)


# ---------------------------------------------------------------------------
# Code execution
# ---------------------------------------------------------------------------

def _clean_completion(code: str, entry_point: str) -> str:
    """
    Clean raw-completion output for HumanEval execution.
    With /v1/completions the model sees the prompt (with docstring at 4-space indent)
    and naturally outputs the body at 4-space indent. This handles edge cases only.
    """
    import re

    # Safety: strip <think> blocks if any model emits them
    code = re.sub(r"<think>.*?</think>", "", code, flags=re.DOTALL).strip()

    # Strip markdown fences (some models still wrap output)
    lines = code.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    code = "\n".join(lines)

    # If model repeated the def signature, skip the whole def line (don't use ":"
    # because type annotations like "numbers: List[float]" contain ":" too).
    def_pattern = re.compile(rf"^def {re.escape(entry_point)}\b", re.MULTILINE)
    matches = list(def_pattern.finditer(code))
    if matches:
        rest = code[matches[-1].start():]
        newline_idx = rest.find("\n")
        if newline_idx != -1:
            code = rest[newline_idx + 1:]

    # Normalize indentation: rank each unique indent level and map to 4*(rank+1).
    # This handles any indent unit (0/8/12 → 4/8/12, or 0/2/4 → 4/8/12, etc.)
    lines = code.splitlines()
    non_empty = [l for l in lines if l.strip()]
    if not non_empty:
        return "    pass"

    indent_values = sorted(set(len(l) - len(l.lstrip()) for l in non_empty))
    indent_map = {v: 4 * (i + 1) for i, v in enumerate(indent_values)}

    result = []
    for l in lines:
        if l.strip():
            cur = len(l) - len(l.lstrip())
            result.append(" " * indent_map[cur] + l.lstrip())
        else:
            result.append("")
    return "\n".join(result)


def execute_solution(prompt: str, completion: str, test_code: str, entry_point: str, timeout: int = 10) -> tuple[bool, str]:
    """
    Assemble: prompt + completion + test_code + check(entry_point)
    Run in a subprocess with a timeout. Returns (passed, error_msg).
    """
    completion = _clean_completion(completion, entry_point)
    full_code = f"{prompt}{completion}\n\n{test_code}\n\ncheck({entry_point})\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        tmp = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return True, ""
        return False, (result.stderr or result.stdout)[:300]
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    except Exception as e:
        return False, str(e)
    finally:
        os.unlink(tmp)


# ---------------------------------------------------------------------------
# Main eval loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["vllm", "ollama"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", required=True, help="Short label for this run")
    ap.add_argument("--port", type=int)
    ap.add_argument("--limit", type=int, default=20, help="Number of HumanEval problems (max 164)")
    ap.add_argument("--timeout", type=int, default=10, help="Per-problem execution timeout (s)")
    ap.add_argument("--debug", action="store_true", help="Print cleaned completion on failure")
    args = ap.parse_args()

    if not DATA_FILE.exists():
        print(f"ERROR: {DATA_FILE} not found. Run: python fetch_humaneval.py")
        sys.exit(1)

    port = args.port or ENGINE_PORTS[args.engine]
    base_url = f"http://localhost:{port}"

    model = args.model
    if model == "auto":
        with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as r:
            model = json.load(r)["data"][0]["id"]

    problems = []
    with open(DATA_FILE) as f:
        for line in f:
            problems.append(json.loads(line))
    problems = problems[: args.limit]

    print(f"\nQuality eval: {args.config}")
    print(f"Engine : {args.engine}  model={model}")
    print(f"Problems: {len(problems)} of 164 HumanEval\n")

    passed = 0
    details = []
    for i, prob in enumerate(problems):
        task_id = prob["task_id"]
        try:
            t0 = time.perf_counter()
            completion = generate(args.engine, base_url, model, prob["prompt"])
            gen_s = time.perf_counter() - t0
            ok, err = execute_solution(prob["prompt"], completion, prob["test"], prob["entry_point"], args.timeout)
        except Exception as e:
            ok, err, gen_s = False, str(e)[:200], 0.0

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{i+1:3d}/{len(problems)}] {task_id:<30} {status}  ({gen_s:.1f}s)")
        if not ok and err:
            print(f"           err: {err[:120]}")
            if args.debug:
                print(f"           raw:     {completion[:200]!r}")
                cleaned = _clean_completion(completion, prob["entry_point"])
                print(f"           cleaned: {cleaned[:200]!r}")
        details.append({"task_id": task_id, "passed": ok, "gen_s": gen_s, "error": err if not ok else ""})

    pass_at_1 = passed / len(problems)
    print(f"\n  pass@1 = {passed}/{len(problems)} = {pass_at_1:.3f}  ({pass_at_1*100:.1f}%)")

    out = {
        "config": args.config, "engine": args.engine, "model": model,
        "timestamp": datetime.now().isoformat(),
        "limit": len(problems), "passed": passed,
        "pass_at_1": pass_at_1, "details": details,
    }
    slug = f"{args.config.replace(' ','_')}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    json_path = RESULTS_DIR / f"quality_{slug}.json"
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results → {json_path}")


class _Tee:
    def __init__(self, path):
        self._file = open(path, "w"); self._stdout = sys.stdout
    def write(self, d): self._stdout.write(d); self._file.write(d)
    def flush(self): self._stdout.flush(); self._file.flush()
    def close(self): self._file.close()


if __name__ == "__main__":
    _slug = None
    for i, a in enumerate(sys.argv):
        if a == "--config" and i + 1 < len(sys.argv):
            _slug = sys.argv[i + 1].replace(" ", "_")
    _ts = datetime.now().strftime("%Y%m%d_%H%M")
    _tee = _Tee(RESULTS_DIR / f"stdout_quality_{_slug or 'run'}_{_ts}.txt")
    sys.stdout = _tee
    try:
        main()
    finally:
        sys.stdout = _tee._stdout
        _tee.close()
