# Local Coding-Agent Inference Benchmark — Apple Silicon

Benchmark measuring how serving-engine and quantization choices affect a coding-agent workload on a 24GB M4 Mac: TTFT, throughput, prefix-cache speedup, and code-generation quality.

See `writeup.md` for the full write-up and findings. See `goal.md` for scope and design decisions.

---

## Hardware

- Apple M4, 24GB unified memory
- macOS Darwin 25.6.0

## Pinned versions

| Component | Version |
|-----------|---------|
| vllm-metal | 0.27.1 |
| mlx-lm | 0.31.3 |
| Ollama | 0.33.3 |
| Python | 3.12 |

## Models tested

| Model | Engine | Quant | Size |
|-------|--------|-------|------|
| Qwen/Qwen3-8B | vllm | BF16 / MLX 8-bit / MLX 4-bit / AWQ 4-bit | 16.4 → 4.6 GB |
| mlx-community/Qwen3.5-9B-MLX-4bit | vllm | MLX 4-bit | ~5.95 GB |
| qwen3.5:9b | Ollama | Q4_K_M | ~6.6 GB |

---

## Reproduce

### 1. Install

```bash
# vllm-metal (pinned)
pip install vllm-metal==0.27.1

# Ollama
brew install ollama        # or: https://ollama.com

# Optional: quantization tools (only needed to self-produce quant checkpoints)
pip install mlx-lm>=0.22.0 autoawq>=0.2.9 transformers>=4.51.3

# Quality eval
pip install evalplus
```

### 2. Download models

```bash
# vllm: Qwen3.5-9B MLX 4-bit from ModelScope (huggingface.co blocked on some networks)
python download_model.py

# Ollama
ollama pull qwen3.5:9b    # Q4_K_M, ~6.6GB
```

### 3. Serve

```bash
# vllm-metal
VLLM_METAL_MEMORY_FRACTION=0.95 vllm serve ~/models/mlx-community--Qwen3.5-9B-MLX-4bit \
  --max-model-len 32768 --enable-prefix-caching

# Ollama (separate terminal)
ollama serve
```

### 4. Run inference benchmark

```bash
# Compare two engines side by side
python run_benchmark.py \
  --model-a "vllm:auto" \
  --model-b "ollama:qwen3.5:9b" \
  --label-a "Qwen3.5-9B vllm-MLX4bit" \
  --label-b "Qwen3.5-9B Ollama-Q4KM" \
  --tests inference

# Individual engine benchmark
python bench.py --config "qwen3.5-9b-vllm" --runs 3
```

### 5. Run quality eval (HumanEval via evalplus)

```bash
# Fetch HumanEvalPlus dataset (requires external download — GitHub releases CDN
# may be blocked; download HumanEvalPlus.jsonl.gz and place at the path below)
mkdir -p ~/Library/Caches/evalplus
gunzip -c HumanEvalPlus.jsonl.gz > ~/Library/Caches/evalplus/HumanEvalPlus-v0.1.10.jsonl

# macOS patch required (resource.setrlimit incompatibility)
# Already applied if you see ValueError: current limit exceeds maximum limit
# Fix: wrap setrlimit calls in try/except in evalplus/eval/utils.py

# vllm
python -m evalplus.evaluate \
  --model "/path/to/mlx-community--Qwen3.5-9B-MLX-4bit" \
  --dataset humaneval --backend openai \
  --base-url http://localhost:8000/v1 --greedy

# Ollama (uses OpenAI-compatible endpoint)
python -m evalplus.evaluate \
  --model "qwen3.5:9b" --dataset humaneval \
  --backend openai --base-url http://localhost:11434/v1 --greedy
```

**Note:** evalplus defaults to 768 `max_new_tokens`. Qwen3.5's thinking mode consumes token budget — patch `evalplus/provider/base.py` to `max_new_tokens=2048` before running Ollama eval.

---

## Key results

| Model | Engine | Quant | TTFT (ms) | Throughput (tok/s) | HumanEval pass@1 | HumanEval+ pass@1 |
|-------|--------|-------|-----------|-------------------|-----------------|------------------|
| Qwen3-8B | vllm | BF16 | 349 | 10.8 | — | — |
| Qwen3-8B | vllm | MLX 8-bit | 390 | 17.8 | — | — |
| Qwen3-8B | vllm | MLX 4-bit | 348 | 25.0 | — | — |
| Qwen3-8B | vllm | AWQ 4-bit | 463 | 7.4 | — | — |
| Qwen3-8B | Ollama | Q4_K_M | 95 | 21.3 | — | — |
| Qwen3.5-9B | Ollama | Q4_K_M | 158 | 16.5 | **90.9%** | **87.2%** |
| Qwen3.5-9B | vllm | MLX 4-bit | 554 | 29.6 | **86.6%** | **82.3%** |

Full analysis in `writeup.md`.

---

## Scripts

| Script | Purpose |
|--------|---------|
| `bench.py` | Single-engine inference benchmark (TTFT, throughput, prefix cache) |
| `compare.py` | Side-by-side engine comparison |
| `run_benchmark.py` | End-to-end: inference + quality for two engine specs |
| `eval_quality.py` | HumanEval pass@1 (custom harness — use evalplus instead) |
| `download_model.py` | Download model shards from ModelScope with retry logic |
| `fetch_humaneval.py` | Download original HumanEval dataset from GitHub |
| `quantize.py` | Self-produce MLX / AWQ quantized checkpoints |

All benchmark scripts use only the Python standard library — no extra pip installs required beyond vllm-metal and Ollama.
