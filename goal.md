# Local Coding-Agent Inference Benchmark on Apple Silicon

## What this is

A benchmark and write-up measuring how serving-engine, quantization, and
concurrency choices affect a *coding-agent* workload — TTFT, throughput,
prefix-caching behavior, memory footprint, and quality degradation — when
running locally on a 24GB M4 Mac. Not a generic "I ran an LLM" post: the
angle is that these inference-layer choices are what make a local coding
agent feel usable or not, and that's a production-engineering argument,
not a demo.

This is a dual-purpose project: a genuine daily-driver tool (local coding
agent I actually use) and a public credibility artifact for AI
infra/inference-optimization positioning.

## Hardware constraints (non-negotiable, design around these)

- Mac M4, 24GB unified memory, shared between OS, model weights, and KV cache
- KV cache scales with context length **independent of weight quantization**
  — cap all benchmark runs at 8K–32K context regardless of the model's
  advertised max context
- Leave realistic headroom for macOS (~3–4GB) — don't chase configs that
  land at 22–23GB total footprint, they're not stable in practice

## Model

- **`Qwen/Qwen3-8B`** — the dense 8B member of the Qwen3 family, not the
  Qwen3.5/3.6/3.8 hybrid/MoE series (important distinction for this benchmark)
- Why dense, not the 3.6/3.8 MoE variants:
  - vllm-metal's automatic prefix caching is ✅ full support on dense Qwen3,
    but only 🔵 experimental on the hybrid/MoE Qwen3.5/3.6/3.8 series
  - Prefix-caching hit rate is a primary benchmark target — experimental
    support makes results unreliable for that finding
  - Dense Qwen3-8B uses standard GQA paged attention, well-understood on Metal
- Memory profile:
  - BF16 (unquantized): ~16GB weights + KV cache — tight on 24GB
  - 4-bit MLX quant: ~4–5GB weights — leaves ~15GB headroom for KV cache,
    OS (~3–4GB), and other processes; the comfortable daily-driver config
- Quantization toolchain: two tools, each serving a different purpose:
  - **AutoAWQ** — calibration-based W4A16. Runs calibration data through the
    model on CPU (~60–90 min), finds high-activation weights and protects them.
    Produces HF-format AWQ checkpoints that vllm-metal loads natively via its
    MLX AWQ repack path. This is the industry-standard method.
  - **mlx-lm** (`mlx_lm.convert`) — naive group quantization. Mathematically
    rounds weights to int4/int8 with no calibration (~15 seconds). Useful as a
    fast baseline and for the 8-bit config where AWQ provides less benefit.
- Quant levels for the comparison curve:
  - BF16 (unquantized baseline — apps closed + sudo purge required)
  - MLX 8-bit — naive rounding, self-produced via mlx-lm
  - MLX 4-bit — naive rounding, self-produced via mlx-lm (fast baseline)
  - AWQ W4A16 — calibration-based, self-produced via AutoAWQ (primary 4-bit config)
- GPTQ, W4A8, W8A8 explicitly excluded: not supported on vllm-metal / Metal
- Rationale for 27B → 8B: 27B (even 4-bit, ~16GB weights) left only ~5GB
  for KV cache + OS — not stable at 8K–32K context. 8B at 4-bit (~4.5GB)
  gives 15+ GB headroom to actually benchmark prefix-caching behavior.

## Serving engines to compare

1. **vllm-metal** (MLX backend, vLLM org) — primary engine
   - Check `docs/supported_models.md` in the repo before assuming Qwen3.6
     loads cleanly; their verified matrix so far calls out Qwen3.8 as the
     tested example, not 3.6
   - If unsupported, fall back to serving via `mlx-lm` directly for that
     leg of the comparison, and say so plainly in the write-up — the point
     is honest reporting, not forcing a specific engine
   - **Pin the exact version/commit used** — this project is weeks old and
     moving fast (v0.1→v0.2 alone was an 83x TTFT change); a benchmark
     without a pinned version is not reproducible
2. **Ollama** (llama.cpp backend) — comparison engine at matched quant level

## What to measure

- **TTFT** at low concurrency (1–2 requests) — this is what determines
  whether inline/autocomplete-style agent interaction feels responsive
- **Tokens/sec** at 1–2 concurrent requests (not high-batch server load —
  that's not this use case's traffic pattern)
- **Prefix-cache hit/miss rate** — coding agents resend a long system
  prompt + tool schema on every call; this is the most specific, least
  commonly reported finding in this space and worth the extra setup effort
- **Memory footprint** at each quant level against the 24GB ceiling
- **Quality vs. quant level**, two tracks:
  - Standardized: `lm-eval-harness` on a small relevant subset (not chasing
    a full leaderboard run — pick 2–3 suites that actually correlate with
    coding/agentic tasks)
  - Practical: my own ~15–20 real prompts pulled from actual daily
    coding-agent usage, self-scored — more honest than a gameable public
    benchmark, and ties the numbers back to the "I use this daily" story

## Deliverables

1. A reproducible benchmark repo (scripts, pinned versions, raw results)
2. A public write-up: methodology, the four-quant-level quality/latency
   curve, the vllm-metal vs. Ollama comparison, the prefix-caching finding,
   and an honest "what I'd do differently on real GPU infra" closing section
   tying it back to production serving economics

## Explicitly out of scope (guard against scope creep)

- Chasing a full public leaderboard score (HumanEval, SWE-bench full run, etc.)
- Any cloud/GPU-instance comparison — this project is specifically about
  the local/edge constraint story, not a cloud vLLM benchmark (that's a
  separate, already-distinct artifact)
- Fine-tuning or training of any kind
- Building this into an actual production deployment / serving other users
- Supporting every possible model — one model, one clean comparison,
  done well beats a sprawling matrix

## Success criteria

- Numbers are reproducible from the repo by someone else on similar hardware
- The prefix-caching and quant-quality-curve findings are specific enough
  that they wouldn't be true of literally every model/engine combination
- The write-up reads as production-engineering reasoning under a real
  constraint (24GB), not a "look what I ran" demo
- Total time budget: fits within the existing 4–6 week / ~4–5hr-per-session
  cadence already in use for this track

## Working notes for Claude Code sessions

- I write the code by hand; use Claude Code for scaffolding, boilerplate,
  and reviewing structure — not first-draft implementation of the core
  benchmark logic
- Each session should end with a runnable increment (a script that
  produces at least one real number), not partial/broken state
- Log versions/commits of vllm-metal, Ollama, llm-compressor, and the
  model checkpoint used in every results file — this space moves weekly
