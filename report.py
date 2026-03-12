#!/usr/bin/env python3
"""
Summarize what we've learned from autotune-ollama results.
Designed to give an AI assistant (or human) a quick read on findings.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

DETAILS = Path("details.jsonl")
PARAMS = ["num_ctx", "temperature", "top_p", "top_k", "repeat_penalty", "num_predict"]


def load_data():
    if not DETAILS.exists():
        print("No details.jsonl found.")
        sys.exit(1)
    return [json.loads(l) for l in DETAILS.open()]


def bar(value, max_value=10, width=16):
    filled = int(round(value / max_value * width))
    return "[" + "#" * filled + "." * (width - filled) + f"] {value:.2f}"


def model_ranking(data):
    models = defaultdict(list)
    for r in data:
        models[r["model"]].append(r["quality"])
    ranked = sorted(models.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)
    return [(name, sum(qs) / len(qs), len(qs)) for name, qs in ranked]


def category_ranking(data):
    """Per-category (prompt_type) model ranking."""
    # {category: {model: [quality scores]}}
    cats = defaultdict(lambda: defaultdict(list))
    for r in data:
        pt = r.get("prompt_type")
        if pt:
            cats[pt][r["model"]].append(r["quality"])
    result = {}
    for cat, models in cats.items():
        ranked = sorted(
            ((sum(v) / len(v), name, len(v)) for name, v in models.items()),
            reverse=True,
        )
        result[cat] = ranked
    return result


def best_overall_config(data, min_samples=4):
    configs = defaultdict(list)
    for r in data:
        key = (r["model"], r.get("num_ctx"), r.get("temperature"), r.get("top_p"),
               r.get("top_k"), r.get("repeat_penalty"), r.get("num_predict"))
        configs[key].append(r["quality"])
    ranked = [
        (sum(v) / len(v), len(v), k)
        for k, v in configs.items()
        if len(v) >= min_samples
    ]
    ranked.sort(reverse=True)
    return ranked


def param_effects(data):
    """For each model x param, show avg quality at each tested value."""
    result = {}
    models = sorted(set(r["model"] for r in data))
    for model in models:
        result[model] = {}
        for param in PARAMS:
            sweep = [r for r in data
                     if r["model"] == model and r["param_being_optimized"] == param]
            if not sweep:
                continue
            by_val = defaultdict(list)
            for r in sweep:
                by_val[r.get(param)].append(r["quality"])
            result[model][param] = {v: sum(q) / len(q) for v, q in sorted(by_val.items())}
    return result


def param_sensitivity(effects_by_model):
    """For each model x param, compute (max - min) quality spread."""
    rows = []
    for model, params in effects_by_model.items():
        for param, val_avgs in params.items():
            if len(val_avgs) < 2:
                continue
            vals = list(val_avgs.values())
            spread = max(vals) - min(vals)
            best_val = max(val_avgs, key=val_avgs.__getitem__)
            rows.append((spread, model, param, best_val, val_avgs))
    rows.sort(reverse=True)
    return rows


def prompt_difficulty(data):
    """Avg quality per prompt across all models."""
    prompts = defaultdict(list)
    for r in data:
        prompts[r["prompt_id"]].append(r["quality"])
    return sorted(
        ((sum(v) / len(v), k, len(v)) for k, v in prompts.items()),
        reverse=True,
    )


def failure_summary(data):
    """Count inference failures from log (approximated by missing prompts in sweep)."""
    # Infer from log if available
    log = Path("run.log")
    if not log.exists():
        return []
    failures = []
    for line in log.read_text().splitlines():
        if "Inference failed" in line:
            failures.append(line.strip())
    return failures


def completed_models(data):
    """Models where sweep appears done (has baseline + multiple param sweeps)."""
    model_params = defaultdict(set)
    model_phases = defaultdict(set)
    for r in data:
        model_phases[r["model"]].add(r["phase"])
        if r["phase"] == "sweep":
            model_params[r["model"]].add(r["param_being_optimized"])
    return model_params, model_phases


def main():
    data = load_data()
    n = len(data)
    models_ranked = model_ranking(data)
    effects = param_effects(data)
    sensitivity = param_sensitivity(effects)
    prompts = prompt_difficulty(data)
    failures = failure_summary(data)
    model_params, model_phases = completed_models(data)

    cat_ranked = category_ranking(data)

    print(f"\n{'=' * 60}")
    print(f"  autotune-ollama findings  ({n} eval records)")
    print(f"{'=' * 60}\n")

    # --- Overall model ranking ---
    print("MODEL RANKING (avg quality across all evals)")
    print("-" * 50)
    for name, avg, count in models_ranked:
        swept = sorted(model_params.get(name, []))
        status = "sweep done" if len(swept) >= 4 else f"swept: {', '.join(swept) or 'none'}"
        print(f"  {bar(avg)}  n={count:<4}  {name}")
        print(f"  {'':18}         {status}")
    print()

    # --- Per-category rankings ---
    cat_order = ["chat", "tool_call", "coding"]
    cat_labels = {"chat": "CHAT", "tool_call": "TOOL CALLING", "coding": "CODING"}
    for cat in cat_order:
        if cat not in cat_ranked:
            continue
        print(f"CATEGORY: {cat_labels[cat]}")
        print("-" * 50)
        for avg, name, count in cat_ranked[cat]:
            print(f"  {bar(avg)}  n={count:<4}  {name}")
        print()
    print()

    # --- Best config ---
    print("BEST CONFIGS (≥4 samples, ranked by avg quality)")
    print("-" * 50)
    best = best_overall_config(data)[:8]
    if best:
        for avg, n_samples, k in best:
            model, num_ctx, temp, top_p, top_k, rep, num_pred = k
            short = model  # keep full name for clarity
            print(f"  {bar(avg)}  n={n_samples}  {short}")
            print(f"  {'':18}         ctx={num_ctx} temp={temp} top_p={top_p} "
                  f"top_k={top_k} rep={rep} pred={num_pred}")
    else:
        print("  (not enough data yet)")
    print()

    # --- Param sensitivity ---
    print("PARAMETER SENSITIVITY (largest quality spread → most impactful)")
    print("-" * 50)
    for spread, model, param, best_val, val_avgs in sensitivity[:15]:
        short_model = model.split(":")[0]
        vals_str = "  ".join(f"{v}→{a:.2f}" for v, a in sorted(val_avgs.items()))
        print(f"  Δ{spread:.2f}  {short_model:<30}  {param}  [best={best_val}]")
        print(f"       {vals_str}")
    print()

    # --- Prompt difficulty ---
    print("PROMPT DIFFICULTY (avg quality across all models, best→worst)")
    print("-" * 50)
    for avg, prompt, count in prompts:
        print(f"  {bar(avg)}  n={count:<4}  {prompt}")
    print()

    # --- Failures ---
    if failures:
        print(f"INFERENCE FAILURES ({len(failures)} total)")
        print("-" * 50)
        # Summarize by context
        ctx_fails = defaultdict(int)
        for f in failures:
            ctx_fails[f] += 1
        for msg, count in sorted(ctx_fails.items(), key=lambda x: -x[1])[:10]:
            print(f"  x{count}  {msg}")
        print()


if __name__ == "__main__":
    main()
