# local-bench-mac

**Benchmark how serving-engine and quantization choices affect a local coding-agent workload on Apple Silicon.**

Measures TTFT, throughput, prefix-cache speedup, memory footprint and HumanEval pass@1 across [vllm-metal](https://github.com/vllm-project/vllm-metal) (MLX backend) and [Ollama](https://ollama.com) (llama.cpp backend), at matched quantization levels — on a 24 GB M4 MacBook, with enough memory left over to keep an editor and a browser open.

📝 **Write-up:** [Benchmarking Local LLM Performance on a MacBook](https://shalabhsingh.github.io/Benchmarking-Local-LLM-Performance-on-a-MacBook/) — the full story, reasoning and findings.
📐 **Scope and design decisions:** [`goal.md`](goal.md) · **Raw notes:** [`writeup.md`](writeup.md)

All benchmark scripts use **only the Python standard library** — nothing to install beyond the engines themselves.

---

## Headline results

Apple M4, 24 GB unified memory, concurrency 1, prefix caching enabled.

| Model | Engine | Quant | Weights | TTFT | tok/s | Cache | HumanEval | HumanEval+ |
|-------|--------|-------|---------|------|-------|-------|-----------|------------|
| Qwen3-8B | vllm | BF16 | 16.38 GB | 349 ms | 10.8 | 1.43× | — | — |
| Qwen3-8B | vllm | MLX 8-bit | 8.70 GB | 390 ms | 17.8 | 1.12× | — | — |
| Qwen3-8B | vllm | MLX 4-bit | 4.61 GB | 348 ms | 25.0 | 0.80× | — | — |
| Qwen3-8B | vllm | AWQ 4-bit | 6.18 GB | 463 ms | 7.4 | 0.79× | — | — |
| Qwen3-8B | Ollama | Q4_K_M | ~5 GB | **95 ms** | 21.3 | 0.67× | — | — |
| Qwen3.5-9B | Ollama | Q4_K_M | ~6.6 GB | 158 ms | 16.5 | 1.31× | **90.9%** | **87.2%** |
| Qwen3.5-9B | vllm | MLX 4-bit | ~5.95 GB | 554 ms | **29.6** | 1.53× | 86.6% | 82.3% |

Four things worth knowing:

- **Quantization buys speed, not just memory.** BF16 → 4-bit is a 2.3× throughput gain, because single-stream decoding here is memory-bandwidth-bound.
- **4-bit is not 4-bit.** Ollama's `Q4_K_M` beats MLX 4-bit by 4.3 pp on HumanEval at the same bit width. The rounding algorithm matters.
- **AWQ is the wrong tool on Metal.** 3.4× slower than plain MLX 4-bit — a format/hardware mismatch, not a quality trade.
- **vLLM wins throughput (~1.8×), Ollama wins TTFT (~3.5×).** Which matters depends on whether your workload is long generations or short interactive turns.

> **Reading the `Cache` column:** it's a `first_TTFT / second_TTFT` proxy, not an engine-reported hit rate, and the two scripts differ — `bench.py` (Qwen3-8B rows) varies the user question between calls, `compare.py` (Qwen3.5-9B rows) repeats an identical prompt. **Compare within a model, not across those two groups.**

---

## Requirements

- **Apple Silicon Mac** (M1 or later). Tested on M4 / 24 GB.
- **Native arm64 Python 3.12.** Rosetta/x86_64 is not supported by vllm-metal.
- **Xcode Command Line Tools** — `xcode-select --install`. vLLM core compiles from source via `clang++`.
- Disk space for weights: ~16 GB for a BF16 8B source checkpoint, plus ~5–9 GB per quantized copy.

---

## Quickstart (~15 minutes, Ollama only)

If you just want numbers on your machine without the vLLM setup:

```bash
git clone https://github.com/shalabhsingh/local-bench-mac.git
cd local-bench-mac

brew install ollama
ollama serve &                 # leave running
ollama pull qwen3.5:9b         # Q4_K_M, ~6.6 GB

python compare.py --engine ollama --model qwen3.5:9b \
  --config "qwen3.5-9b-ollama" --runs 5
```

Results land in `results/compare_qwen3.5-9b-ollama_<timestamp>.json` with a plain-text log alongside it.

---

## Full setup

### 1. Install vllm-metal

Use the project's own installer — it resolves the core wheel, the plugin and the prebuilt Metal kernels together:

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
source ~/.venv-vllm-metal/bin/activate
vllm --version
```

[`install.sh`](install.sh) in this repo is a **version-pinned variant** of the same flow, for reproducing a specific core release rather than whatever is current:

```bash
./install.sh          # pins vLLM core 0.27.1, installs the plugin's latest release
```

> If you install the plugin from a source checkout rather than a release wheel, you must prebuild the native artifacts (`_paged_ops` + the `.metallib` shaders) or the first request touching paged attention fails with `Prebuilt native extension not found`. The release wheel ships them built; `install.sh` handles both paths.

### 2. Install Ollama

```bash
brew install ollama    # or https://ollama.com
ollama serve
```

### 3. Optional extras

```bash
# only needed to self-produce quantized checkpoints
pip install mlx-lm>=0.22.0 transformers>=4.51.3
pip install autoawq --no-build-isolation --no-deps   # see "AWQ on Mac" below

# only needed for the quality eval
pip install evalplus
```

### 4. Get weights

```bash
# from ModelScope — useful when huggingface.co is blocked.
# Parallel shards, resume, retry-on-disconnect. Prints the local path when done.
python download_model.py Qwen/Qwen3-8B
python download_model.py mlx-community/Qwen3.5-9B-MLX-4bit

# Ollama side
ollama pull qwen3:8b
ollama pull qwen3.5:9b
```

### 5. Quantize (optional)

All quantization starts from the **same BF16 source checkpoint** — that's what makes the algorithm comparison controlled.

```bash
python quantize.py --method mlx4   # ~15s  → ~/models/Qwen3-8B-4bit-mlx
python quantize.py --method mlx8   # ~15s  → ~/models/Qwen3-8B-8bit-mlx
python quantize.py --method awq4   # 60–90 min on CPU
python quantize.py --method all
```

`mlx4`/`mlx8` are round-to-nearest group quantization: per-group scale and bias in 16-bit, weights as 4-/8-bit integers, activations left in BF16 (so W4A16 / W8A16). No calibration, no forward passes — hence 15 seconds. `awq4` is calibrated: it pushes coding-relevant samples through the model, scales per-channel so weights hit by large activations keep more precision, and leaves the most sensitive tensors in bfloat16. AWQ 8-bit doesn't exist — AutoAWQ's GEMM kernel is 4-bit only.

### 6. Serve

```bash
# vllm-metal. Note: AWQ needs 0.90; MLX configs tolerate 0.95.
VLLM_METAL_MEMORY_FRACTION=0.95 vllm serve ~/models/Qwen3-8B-4bit-mlx \
  --max-model-len 32768 --enable-prefix-caching

# Ollama, separate terminal
ollama serve
```

**Read the startup log.** The `Paged attention memory breakdown` line gives you the three numbers to pass into `bench.py`:

```
metal_limit=19.07GB, fraction=0.95, usable_metal=18.12GB,
model_memory=4.61GB, overhead=0.70GB, kv_budget=12.81GB, max_tokens_cached=86880
```

---

## Running the benchmarks

```bash
# Single vllm config. Memory figures come from the serve log above.
python bench.py --config "4bit-MLX" --runs 5 \
  --model-memory-gb 4.61 --kv-budget-gb 12.81 --max-tokens-cached 86880

# Either engine, same three tests — use this for cross-engine comparison
python compare.py --engine vllm   --model ~/models/Qwen3-8B-4bit-mlx --config "qwen3-8b-4bit-vllm" --runs 5
python compare.py --engine ollama --model qwen3.5:9b                 --config "qwen3.5-9b-ollama"  --runs 5

# End-to-end: two engine specs, inference + quality in one go
python run_benchmark.py \
  --model-a "vllm:auto" --model-b "ollama:qwen3.5:9b" \
  --label-a "Qwen3.5-9B vllm-MLX4bit" --label-b "Qwen3.5-9B Ollama-Q4KM" \
  --tests inference,quality --quality-limit 20

# Interactive chat against either engine — useful for a sanity check before benchmarking
python chat.py --engine ollama --model qwen3.5:9b
```

Useful flags: `--test ttft|throughput|prefix|all` to run one test in isolation, `--runs N` for repetitions, `--think` to enable Qwen3 thinking mode (off by default — `compare.py` uses Ollama's native `/api/chat` endpoint, because the OpenAI-compatible one silently ignores `think: false`), `--debug` to dump raw streaming chunks.

### Quality eval

Use **evalplus**, not `eval_quality.py`. The bundled harness is kept as a reference, but it splices completions onto the prompt signature and produces `IndentationError` failures that look like model errors and aren't.

```bash
python -m evalplus.evaluate --model "/path/to/model" --dataset humaneval \
  --backend openai --base-url http://localhost:8000/v1 --greedy

python -m evalplus.evaluate --model "qwen3.5:9b" --dataset humaneval \
  --backend openai --base-url http://localhost:11434/v1 --greedy

# score pre-generated completions, no model needed
python -m evalplus.evaluate --dataset humaneval --samples <path-to.jsonl>
```

Three fixes evalplus needs on macOS, none of them scripted yet:

1. **Dataset download.** If `huggingface.co` or `objects.githubusercontent.com` is blocked, fetch `HumanEvalPlus.jsonl.gz` out of band and place it by hand:
   ```bash
   mkdir -p ~/Library/Caches/evalplus
   gunzip -c HumanEvalPlus.jsonl.gz > ~/Library/Caches/evalplus/HumanEvalPlus-v0.1.10.jsonl
   ```
2. **`resource.setrlimit` raises on Darwin** (`ValueError: current limit exceeds maximum limit`). Wrap the `setrlimit` calls in `try/except` in `evalplus/eval/utils.py`; the memory sandbox is advisory on macOS anyway.
3. **`max_new_tokens=768` is too low for a thinking-mode model** — reasoning tokens eat the budget and completions truncate mid-function, silently halving your score. Patch `evalplus/provider/base.py` to 2048.

---

## Benchmarking your own model

The harness isn't Qwen-specific. In order:

**1. Check support.** Find your model in vllm-metal's [`docs/supported_models.md`](https://github.com/vllm-project/vllm-metal/blob/main/docs/supported_models.md). You want ✅ in both the support and the *Automatic Prefix Cache* column — 🔵 in the cache column means cache results are indicative only.

**2. Do the memory arithmetic first, before downloading anything.**

```
weights    ≈ params × bits/8                                   # 8B @ 4-bit ≈ 4.5 GB
kv_budget  ≈ metal_limit × fraction − weights − overhead        # overhead ran 0.66–1.45 GB
```

`metal_limit` ≈ total RAM minus what macOS and your apps hold (~5 GB with a normal working set). Then convert budget to context — this part is exact:

```
bytes per token = 2 (K and V) × layers × kv_heads × head_dim × 2 bytes
```

Qwen3-8B: `2 × 36 × 8 × 128 × 2 = 147,456 B ≈ 144 KiB/token`, so a 12.81 GB budget is 86,880 tokens — matching the serve log exactly. **The KV cache stays 16-bit no matter how hard you quantize the weights**, so context cost is independent of weight quantization.

**3. Make the workload yours.** Edit `SYSTEM_PROMPT` and `CODING_QUESTIONS` in `bench.py` / `compare.py` to match your real traffic. The prefix-cache number is only meaningful if the system prompt is the length yours actually is.

**4. Log everything.** Every run writes engine version, model path, memory breakdown and all individual timings — not just medians. This ecosystem moves weekly; a number without its provenance is a number you'll distrust in a month.

---

## Repo layout

| Path | Purpose |
|------|---------|
| [`bench.py`](bench.py) | Single-engine benchmark: TTFT, throughput, prefix cache, memory sampling |
| [`compare.py`](compare.py) | Same three tests against either engine — the cross-engine workhorse |
| [`run_benchmark.py`](run_benchmark.py) | End-to-end: inference + quality for two engine specs |
| [`quantize.py`](quantize.py) | Produce MLX 4/8-bit and AWQ 4-bit checkpoints from one BF16 source |
| [`download_model.py`](download_model.py) | Shard download from ModelScope with resume + retry |
| [`chat.py`](chat.py) | Interactive terminal chat against vllm or Ollama |
| [`eval_quality.py`](eval_quality.py) | Bundled HumanEval harness — **reference only, use evalplus** |
| [`fetch_humaneval.py`](fetch_humaneval.py) | Download the original HumanEval dataset |
| `results/` | Raw result JSONs + stdout logs for every run in the write-up |
| `logs/` | `vllm serve` startup logs per quant config (the memory story) |
| `evalplus_results/` | HumanEval / HumanEval+ generations and scores |

---

## Known gotchas

<!-- These are real failures hit during this benchmark. Each costs an hour if you meet it cold. -->

| Symptom | Cause | Fix |
|---------|-------|-----|
| `kv_budget=-0.13GB` at startup | Weights + overhead exceed usable Metal memory | Quantize; or raise `VLLM_METAL_MEMORY_FRACTION`, lower `--max-model-len`, close apps and `sudo purge` |
| Metal OOM inside `mx.eval(logits_2d)` mid-inference | AWQ's non-quantized bfloat16 tensors + activation spikes exceed `wired_limit` | Drop `VLLM_METAL_MEMORY_FRACTION` to `0.90` |
| `[load_safetensors] invalid data offsets ... exceeding the size of the file` | Truncated shard kept by skip-if-exists when the size `HEAD` request also dropped | Delete that shard and re-run `download_model.py` |
| HTTP 400, `default chat template is no longer allowed` | Community MLX upload omits `chat_template` from `tokenizer_config.json` | Copy the field from a model in the same tokenizer family |
| Ollama returns zero content tokens | OpenAI-compatible endpoint ignores `think: false`; reasoning tokens consume the whole budget | Use Ollama's native `/api/chat` with `"think": false` — what `compare.py` does |
| `ModuleNotFoundError: No module named 'torch'` installing AutoAWQ | Build isolation can't see the venv | `pip install autoawq --no-build-isolation` |
| `ResolutionImpossible: autoawq depends on triton` | `triton` is CUDA-only, no Mac wheel | `pip install autoawq --no-deps` — the CPU calibration path doesn't call triton |
| `Couldn't find 'mit-han-lab/pile-val-backup' on the Hub` | AutoAWQ defaults to downloading the Pile for calibration | Pass `calib_data=[...]` to `model.quantize()` — `quantize.py` already does, with coding samples |
| `Prebuilt native extension not found` | Plugin installed from source without building the Metal artifacts | Install the release wheel, or build `_paged_ops` + `.metallib` |

---

## Scope

**In scope:** one model family, two engines, four quantization levels, one clean comparison under a real 24 GB memory constraint.

**Out of scope:** leaderboard chasing, cloud/GPU comparison, fine-tuning, multi-user production serving, and tool-calling agent harnesses on top (worth building, but a layer above this).

## License

MIT — see [`LICENSE`](LICENSE).
