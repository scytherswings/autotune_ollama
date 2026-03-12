# autotune-ollama: Autonomous Parameter Optimization

## What This Does

Automatically finds the best Ollama model + parameter configuration for agentic tool-calling and coding workloads on a local GPU (tested on RTX 5070 / 12GB VRAM).

## How It Works

1. **Outer loop**: Cycles through infrastructure configs (flash attention, parallelism levels) by restarting the Ollama Docker container directly via `docker compose`
2. **Middle loop**: For each infra config, tests each candidate model — pulls it, verifies it fits fully in GPU VRAM, warms up
3. **Inner loop**: Uses coordinate descent to optimize per-request parameters (context window, temperature, top_p, top_k, repeat penalty, max tokens) one at a time
4. **Evaluation**: Each parameter setting is tested against a suite of 15 prompts (4 coding + 11 tool-calling). Outputs are scored by Claude Sonnet against pre-generated Opus reference answers
5. **Scoring**: Composite score = quality (65%) + latency — total wall-clock time per prompt (35%), which captures thinking time for reasoning models

## Key Constraints

- All models must fit entirely in GPU VRAM (no CPU offload)
- Quality matters most — a fast but wrong model is useless
- Budget-capped API calls to Anthropic

## Files

| File | Purpose |
|------|---------|
| `autotune.py` | Main loop — coordinate descent optimizer |
| `eval_harness.py` | Ollama API client — inference, metrics, GPU checks |
| `judge.py` | Claude Sonnet quality scoring |
| `generate_references.py` | One-time Opus reference answer generation |
| `eval_prompts.json` | Eval prompt suite (coding + tool-calling) |
| `config.yaml` | All configuration — models, search space, scoring weights |
| `results.tsv` | Output — one row per experiment (resume key) |
| `details.jsonl` | Per-prompt detail records for analysis |
| `configs/` | Docker Compose configs for each infra variant |

## Future Optimizations

### Sweep strategy

**Baseline-first, proportional sweep budget**
Run baseline evals across all models first, then allocate sweep budget proportionally to top performers. Currently the full coordinate descent sweep runs on every model equally regardless of early signals. Since baseline ranking closely predicts final ranking, investing sweep budget in the top 1-2 models and doing a shallow sweep (or skipping) on the rest would be more efficient.

**Binary search for max viable num_ctx**
Rather than testing fixed ctx values and hitting a cliff, run a binary search phase before the main sweep to find the highest `num_ctx` that stays above a target TPS (e.g. 25 tok/s). The transition from full-GPU to partial-offload is a step change driven by Ollama's VRAM allocation strategy — interpolation doesn't work across it, but bisection between the last passing and first failing value would find the optimal ctx ceiling per model in ~3-4 extra inference calls.

**Early stopping on poor baselines**
If a model's baseline quality is below a configurable threshold, skip the sweep entirely. codellama:13b scored 3.56 at baseline and never recovered — that budget was wasted.

### Infrastructure

**Log Ollama container output per experiment**
Currently we infer GPU vs CPU inference from TPS alone. Capturing the relevant lines from `docker logs` (layer offload count, KV cache split) at each num_ctx trial would give definitive confirmation and richer data for post-hoc analysis.

**`OLLAMA_GPU_OVERHEAD` tuning**
The `low-overhead` config sets `OLLAMA_GPU_OVERHEAD=512MB`, reducing Ollama's VRAM reservation buffer. This might push marginal ctx sizes (e.g. qwen14b at 16k) from partial-offload into full-GPU. Worth testing as a separate infra config sweep.
