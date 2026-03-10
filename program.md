# autotune-ollama: Autonomous Parameter Optimization

## What This Does

Automatically finds the best Ollama model + parameter configuration for agentic coding and chat workloads on a 12GB VRAM GPU (RTX 5070).

## How It Works

1. **Outer loop**: Cycles through infrastructure configs (flash attention, parallelism levels) by restarting the Ollama Docker container via a gatekeeper script over SSH
2. **Middle loop**: For each infra config, tests each candidate model — pulls it, verifies it fits fully in GPU VRAM, warms up
3. **Inner loop**: Uses coordinate descent to optimize per-request parameters (context window, temperature, top_p, top_k, repeat penalty, max tokens) one at a time
4. **Evaluation**: Each parameter setting is tested against a suite of coding and chat prompts. Outputs are scored by Claude Sonnet against pre-generated Opus reference answers
5. **Scoring**: Composite score = quality (60%) + speed (25%) + time-to-first-token (15%)

## Key Constraints

- All models must fit entirely in GPU VRAM (no CPU offload)
- No sudo/docker access — all container management goes through the gatekeeper script
- Quality matters most — a fast but wrong coder is useless
- Budget-capped API calls to Anthropic

## Files

| File | Purpose |
|------|---------|
| `autotune.py` | Main loop — coordinate descent optimizer |
| `eval_harness.py` | Ollama API client — inference, metrics, GPU checks |
| `judge.py` | Claude Sonnet quality scoring |
| `generate_references.py` | One-time Opus reference answer generation |
| `eval_prompts.json` | Eval prompt suite with reference answers |
| `config.yaml` | All configuration — VM, models, search space, scoring |
| `results.tsv` | Output — every experiment logged |
| `vm/` | Docker Compose configs + gatekeeper script for the VM |
