#!/usr/bin/env python3
"""Quick status report for an autotune-ollama run."""

import json
import sys
from collections import defaultdict
from pathlib import Path

DETAILS = Path("details.jsonl")
LOG = Path("run.log")


def load_data():
    if not DETAILS.exists():
        print("No details.jsonl found.")
        sys.exit(1)
    return [json.loads(l) for l in DETAILS.open()]


def model_summary(data):
    """Per-model record counts, avg quality, and phase breakdown."""
    models = defaultdict(lambda: {"quality": [], "phases": defaultdict(int)})
    for r in data:
        m = models[r["model"]]
        m["quality"].append(r["quality"])
        m["phases"][r["phase"]] += 1
    return models


def best_configs(data, top_n=5):
    """Best (model, num_ctx, temperature, top_p) combos by avg quality, min 4 samples."""
    configs = defaultdict(list)
    for r in data:
        key = (r["model"], r["num_ctx"], r["temperature"], r["top_p"], r["top_k"], r["repeat_penalty"])
        configs[key].append(r["quality"])
    ranked = [
        (sum(v) / len(v), len(v), k)
        for k, v in configs.items()
        if len(v) >= 4
    ]
    ranked.sort(reverse=True)
    return ranked[:top_n]


def current_activity(data):
    """What the run is currently doing, based on the last few records."""
    last = data[-1]
    return last


def prompt_breakdown(data, model=None):
    """Per-prompt avg quality across all (or a specific) model."""
    records = [r for r in data if model is None or r["model"] == model]
    prompts = defaultdict(list)
    for r in records:
        prompts[r["prompt_id"]].append(r["quality"])
    return {k: sum(v) / len(v) for k, v in sorted(prompts.items())}


def infer_run_state(data):
    """Guess if a run is in-progress or complete."""
    if LOG.exists():
        content = LOG.read_text()
        if content.rstrip().endswith("autotune-ollama complete\n" + "=" * 60):
            return "complete"
        # Check last line pattern
        lines = [l for l in content.splitlines() if l.strip()]
        if lines and "autotune-ollama complete" in lines[-1]:
            return "complete"
    return "in-progress"


def print_bar(value, max_value=10, width=20):
    filled = int(round(value / max_value * width))
    return "[" + "#" * filled + "." * (width - filled) + f"] {value:.2f}"


def main():
    data = load_data()
    last = current_activity(data)
    models = model_summary(data)
    state = infer_run_state(data)

    print(f"\n=== autotune-ollama status ({state}) ===")
    print(f"Total eval records: {len(data)}")
    print()

    # Per-model summary
    print("Models:")
    model_rows = []
    for name, m in models.items():
        avg = sum(m["quality"]) / len(m["quality"])
        phases = ", ".join(f'{p}:{n}' for p, n in sorted(m["phases"].items()))
        model_rows.append((avg, name, len(m["quality"]), phases))
    model_rows.sort(reverse=True)
    for avg, name, n, phases in model_rows:
        bar = print_bar(avg)
        print(f"  {bar}  n={n:<4}  {name}")
        print(f"           {'':20}         phases: {phases}")
    print()

    # Current activity
    if state == "in-progress":
        print("Currently:")
        print(f"  model    : {last['model']}")
        print(f"  phase    : {last['phase']}")
        print(f"  sweeping : {last['param_being_optimized']}")
        print(f"  last val : {last['param_being_optimized']}={last[last['param_being_optimized']]}")
        print(f"  last prompt: {last['prompt_id']}  quality={last['quality']:.1f}")
        print(f"  timestamp: {last['timestamp']}")
        print()

    # Best configs
    print("Best configs (≥4 samples):")
    best = best_configs(data)
    if best:
        for avg, n, k in best:
            model, num_ctx, temp, top_p, top_k, rep = k
            bar = print_bar(avg)
            short_model = model.split(":")[0] + ":" + model.split(":")[-1]
            print(f"  {bar}  n={n}  {short_model}  ctx={num_ctx} temp={temp} top_p={top_p}")
    else:
        print("  (not enough data yet)")
    print()

    # Per-prompt quality for best model
    best_model = model_rows[0][1] if model_rows else None
    if best_model:
        print(f"Per-prompt quality ({best_model.split(':')[0]}):")
        prompts = prompt_breakdown(data, model=best_model)
        for prompt, avg in sorted(prompts.items(), key=lambda x: -x[1]):
            bar = print_bar(avg)
            print(f"  {bar}  {prompt}")
        print()


if __name__ == "__main__":
    main()
