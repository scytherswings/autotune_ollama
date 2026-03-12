# autotune-ollama

Automatically finds the best Ollama model and inference parameters for agentic tool-calling and coding workloads on a local GPU.

Runs a coordinate descent sweep across models and parameters, scores each configuration against a structured eval suite using Claude as a judge, and produces a ranked leaderboard with optimal settings per model.

## Results (RTX 5070, 12GB VRAM)

| Model | Avg Quality | Best Config |
|-------|-------------|-------------|
| qwen3:14b | **8.80** | ctx=8192, temp=0.0, top_p=0.95 |
| qwen2.5-coder:14b-instruct-q4_K_M | 8.36 | ctx=4096, temp=0.3, top_p=0.95 |
| qwen3-coder:30b | 8.05 | ctx=32768, temp=0.3, top_p=0.95 |
| llama3.1:8b-instruct-q8_0 | 7.89 | — |
| gemma3:12b | 5.39 | — |
| deepseek-coder-v2:16b-lite-instruct-q4_K_M | 4.88 | — |

**Key finding:** `qwen3:14b` outperforms the 2x-larger `qwen3-coder:30b` due to significantly better tool-calling accuracy (9.54 vs 7.76). The coder model excels at coding tasks (8.32 vs 6.66) but struggles with booking, scheduling, and tool selection prompts.

## How It Works

1. **Outer loop** — cycles through infra configs (flash-attn on/off, parallelism) by restarting the Ollama Docker container
2. **Middle loop** — for each infra config, pulls and warms up each candidate model; skips any that don't fit in VRAM
3. **Inner loop** — coordinate descent over `num_ctx`, `temperature`, `top_p` (one parameter at a time, holding others fixed)
4. **Eval** — each config is tested against 15 prompts: 4 coding tasks + 11 tool-calling scenarios
5. **Scoring** — composite score = quality (65%) + latency (35%, wall-clock time per prompt including thinking time)
6. **Judging** — Claude Sonnet scores outputs against pre-generated Opus reference answers

## Requirements

- Linux with NVIDIA GPU (tested on RTX 5070 / Blackwell)
- Docker + Docker Compose v2
- NVIDIA Container Toolkit
- Python 3.10+ and [uv](https://github.com/astral-sh/uv)
- Anthropic API key

## Quick Start

```bash
# 1. Clone and install dependencies
git clone <repo-url> autotune-ollama
cd autotune-ollama
uv venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Configure
cp .env.example .env          # add your ANTHROPIC_API_KEY
cp config.yaml.example config.yaml  # edit models/search space as needed

# 3. Generate reference answers (one-time, ~15 API calls)
python generate_references.py

# 4. Run
python autotune.py
```

See [setup.md](setup.md) for full installation instructions including Docker and NVIDIA Container Toolkit setup.

## Configuration

All tuning knobs are in `config.yaml` (gitignored; copy from `config.yaml.example`):

- **`models`** — which Ollama models to evaluate
- **`skip_models`** — models to exclude (e.g. eliminated in a prior run)
- **`infra_configs`** — which Docker Compose variants to test (`baseline`, `flash-attn`, etc.)
- **`search_space`** — parameter values to sweep per model
- **`eval.type_weights`** — relative weight of coding vs tool-calling in the composite score
- **`scoring`** — quality vs latency trade-off weights
- **`budget.max_api_calls`** — cap on Anthropic API calls

## Eval Suite

15 prompts across two categories:

**Coding (4):** trie implementation, bug fix, refactor strategy, async rate limiter

**Tool-calling (11):** appointment booking (explicit, informal, sore-muscles), cancel with reason, reschedule (complete and missing-info), availability check, client lookup, wrong-tool trap, ambiguous check-vs-book, no-tool (incomplete info)

Tool-calling prompts test a realistic scheduling assistant with 5 tools: `check_availability`, `book_appointment`, `cancel_appointment`, `get_client_appointments`, `get_client_info`.

## Monitoring a Run

```bash
./status.py   # live leaderboard + current sweep activity
./report.py   # full findings: rankings, param sensitivity, prompt difficulty
```

## Files

| File | Purpose |
|------|---------|
| `autotune.py` | Main loop — coordinate descent optimizer |
| `eval_harness.py` | Ollama API client — inference, metrics, GPU checks |
| `judge.py` | Claude Sonnet quality scoring |
| `generate_references.py` | One-time Opus reference answer generation |
| `score_reference_baseline.py` | Score reference answers against themselves (sanity check) |
| `status.py` | Live status report for in-progress runs |
| `report.py` | Full analytical summary of completed runs |
| `eval_prompts.json` | Eval prompt suite |
| `config.yaml.example` | Configuration template |
| `configs/` | Docker Compose variants for each infra config |
