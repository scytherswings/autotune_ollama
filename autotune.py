"""Main orchestration loop: coordinate descent parameter optimization for Ollama models."""

import csv
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()  # Load .env before anything touches os.environ

sys.stdout.reconfigure(line_buffering=True)  # Flush after each line when piped to tee/log

from eval_harness import (
    check_gpu_fit,
    ollama_url,
    pull_model,
    run_inference,
    wait_for_api,
    warmup,
)
from judge import judge_output
from preflight import preflight_check


TSV_COLUMNS = [
    "timestamp",
    "infra_config",
    "model",
    "phase",
    "param_being_optimized",
    "num_ctx",
    "temperature",
    "top_p",
    "top_k",
    "repeat_penalty",
    "num_predict",
    "tokens_per_sec",
    "ttft_ms",
    "quality_score",
    "composite_score",
    "is_best",
    "notes",
]


@dataclass
class EvalResult:
    """Aggregated result from evaluating a full prompt suite."""
    avg_quality: float
    avg_tokens_per_sec: float
    avg_ttft_ms: float
    per_prompt: list  # List of dicts with per-prompt details


def load_config(path: str = "config.yaml") -> dict:
    if not Path(path).exists():
        print(f"ERROR: {path} not found.")
        print(f"  Copy config.yaml.example to config.yaml and fill in your settings.")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def load_eval_prompts(
    prompts_path: str = "eval_prompts.json",
    references_path: str = "references.json",
) -> list[dict]:
    """Load eval prompts and merge in reference answers from references.json."""
    if not Path(prompts_path).exists():
        print(f"ERROR: {prompts_path} not found.")
        print(f"  Make sure eval_prompts.json is present in the project directory.")
        sys.exit(1)

    with open(prompts_path) as f:
        data = json.load(f)

    refs: dict[str, str] = {}
    if Path(references_path).exists():
        with open(references_path) as f:
            refs = json.load(f)

    prompts = data.get("coding_prompts", []) + data.get("chat_prompts", [])

    # Merge references in without mutating the source data
    merged = []
    for p in prompts:
        if p["id"] in refs:
            merged.append({**p, "reference": refs[p["id"]]})

    skipped = len(prompts) - len(merged)

    if not merged:
        print("ERROR: No eval prompts have reference answers. Run generate_references.py first.")
        sys.exit(1)

    if skipped:
        print(f"  WARNING: {skipped} prompt(s) skipped (no reference answer yet).")
    print(f"Loaded {len(merged)} eval prompts with references.")
    return merged


def load_completed_experiments(tsv_path: str) -> set[str]:
    """Load completed experiment keys from existing results.tsv for resume capability."""
    completed = set()
    if not Path(tsv_path).exists():
        return completed

    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            key = _experiment_key(
                row.get("infra_config", ""),
                row.get("model", ""),
                row.get("phase", ""),
                row.get("param_being_optimized", ""),
                row.get("num_ctx", ""),
                row.get("temperature", ""),
                row.get("top_p", ""),
                row.get("top_k", ""),
                row.get("repeat_penalty", ""),
                row.get("num_predict", ""),
            )
            completed.add(key)

    return completed


def _experiment_key(*args) -> str:
    return "|".join(str(a) for a in args)


def init_tsv(tsv_path: str) -> None:
    """Create TSV file with headers if it doesn't exist."""
    if not Path(tsv_path).exists():
        with open(tsv_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(TSV_COLUMNS)


def append_tsv(tsv_path: str, row: dict) -> None:
    """Append a single result row to the TSV."""
    with open(tsv_path, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([row.get(col, "") for col in TSV_COLUMNS])


def switch_infra_config(
    config_name: str,
    compose_dir: str,
    project_name: str,
    ollama_volume: str,
) -> bool:
    """Switch Ollama infra config by restarting the Docker container locally."""
    compose_file = Path(compose_dir) / f"docker-compose.{config_name}.yml"
    if not compose_file.exists():
        print(f"ERROR: Compose file not found: {compose_file}")
        return False

    print(f"\n{'='*60}")
    print(f"Switching infra config: {config_name}")
    print(f"{'='*60}")

    # Stop any stale container on port 11434 (handles migration from other setups)
    result = subprocess.run(
        ["docker", "ps", "-q", "--filter", "publish=11434"],
        capture_output=True, text=True,
    )
    stale = [c for c in result.stdout.strip().split() if c]
    if stale:
        print(f"  Stopping {len(stale)} stale container(s)...")
        subprocess.run(["docker", "stop"] + stale, capture_output=True)

    # Pass the configured volume name so compose can substitute ${OLLAMA_VOLUME}
    env = os.environ.copy()
    env["OLLAMA_VOLUME"] = ollama_volume

    # Start new config
    result = subprocess.run(
        ["docker", "compose", "-p", project_name, "-f", str(compose_file),
         "up", "-d", "--force-recreate"],
        capture_output=True, text=True, timeout=120,
        env=env,
    )
    if result.returncode != 0:
        print(f"ERROR: docker compose failed:\n{result.stderr}")
        return False
    return True


def compute_quality(scores: dict, judge_weights: dict) -> float:
    """Compute quality score as weighted mean of judge sub-scores."""
    return sum(scores[k] * judge_weights[k] for k in judge_weights)


def append_details(
    details_path: str,
    infra_config: str,
    model: str,
    phase: str,
    param_being_optimized: str,
    params: dict,
    prompt_id: str,
    scores: dict,
    quality: float,
    tokens_per_sec: float,
    ttft_ms: float,
) -> None:
    """Append one per-prompt detail record to details.jsonl."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "infra_config": infra_config,
        "model": model,
        "phase": phase,
        "param_being_optimized": param_being_optimized,
        **params,
        "prompt_id": prompt_id,
        "correctness": scores["correctness"],
        "completeness": scores["completeness"],
        "clarity": scores["clarity"],
        "agent_utility": scores["agent_utility"],
        "quality": round(quality, 4),
        "brief_rationale": scores.get("brief_rationale", ""),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "ttft_ms": round(ttft_ms, 1),
    }
    with open(details_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def evaluate_params(
    model: str,
    infra_config: str,
    phase: str,
    param_being_optimized: str,
    params: dict,
    eval_prompts: list[dict],
    base_url: str,
    judge_model: str,
    judge_weights: dict,
    details_path: str,
) -> EvalResult:
    """Run all eval prompts with given params, judge quality, return aggregated scores."""
    quality_scores = []
    tps_values = []
    ttft_values = []
    per_prompt = []

    for prompt_entry in eval_prompts:
        prompt_id = prompt_entry["id"]
        prompt_text = prompt_entry["prompt"]
        reference = prompt_entry["reference"]

        try:
            result = run_inference(model, prompt_text, params, base_url)
        except Exception as e:
            print(f"    Inference failed for {prompt_id}: {e}")
            per_prompt.append({"id": prompt_id, "error": str(e)})
            continue

        # Judge quality — let fatal errors (billing, auth) propagate and abort the run
        scores = judge_output(
            prompt=prompt_text,
            candidate=result.response_text,
            reference=reference,
            model=judge_model,
        )
        quality = compute_quality(scores, judge_weights)

        append_details(
            details_path=details_path,
            infra_config=infra_config,
            model=model,
            phase=phase,
            param_being_optimized=param_being_optimized,
            params=params,
            prompt_id=prompt_id,
            scores=scores,
            quality=quality,
            tokens_per_sec=result.tokens_per_sec,
            ttft_ms=result.ttft_ms,
        )

        quality_scores.append(quality)
        tps_values.append(result.tokens_per_sec)
        ttft_values.append(result.ttft_ms)

        per_prompt.append({
            "id": prompt_id,
            "quality": quality,
            "tokens_per_sec": result.tokens_per_sec,
            "ttft_ms": result.ttft_ms,
            "rationale": scores.get("brief_rationale", ""),
        })

        print(f"    {prompt_id}: quality={quality:.2f} (c={scores['correctness']:.0f} co={scores['completeness']:.0f} cl={scores['clarity']:.0f} a={scores['agent_utility']:.0f}) tps={result.tokens_per_sec:.1f} ttft={result.ttft_ms:.0f}ms")

    if not quality_scores:
        return EvalResult(avg_quality=0, avg_tokens_per_sec=0, avg_ttft_ms=float("inf"), per_prompt=per_prompt)

    return EvalResult(
        avg_quality=sum(quality_scores) / len(quality_scores),
        avg_tokens_per_sec=sum(tps_values) / len(tps_values),
        avg_ttft_ms=sum(ttft_values) / len(ttft_values),
        per_prompt=per_prompt,
    )


def compute_composite(
    quality: float,
    tps: float,
    ttft: float,
    tps_range: tuple[float, float],
    ttft_range: tuple[float, float],
    weights: dict,
) -> float:
    """Compute composite score with min-max normalization for speed metrics."""
    # Normalize tokens/sec (higher is better)
    tps_min, tps_max = tps_range
    if tps_max > tps_min:
        tps_norm = (tps - tps_min) / (tps_max - tps_min)
    else:
        tps_norm = 0.5

    # Normalize TTFT (lower is better, so invert)
    ttft_min, ttft_max = ttft_range
    if ttft_max > ttft_min:
        ttft_norm = 1.0 - (ttft - ttft_min) / (ttft_max - ttft_min)
    else:
        ttft_norm = 0.5

    # Clamp to [0, 1]
    tps_norm = max(0, min(1, tps_norm))
    ttft_norm = max(0, min(1, ttft_norm))

    # Quality is on 1-10 scale, normalize to 0-1
    quality_norm = (quality - 1) / 9

    composite = (
        quality_norm * weights["quality_weight"]
        + tps_norm * weights["speed_weight"]
        + ttft_norm * weights["ttft_weight"]
    )

    # Scale back to 0-10 for readability
    return composite * 10


def coordinate_descent(
    model: str,
    infra_config: str,
    config: dict,
    eval_prompts: list[dict],
    base_url: str,
    tsv_path: str,
    details_path: str,
    completed: set[str],
    api_call_count: list[int],
) -> dict:
    """Run coordinate descent optimization for a single model+infra combo.

    Optimizes one parameter at a time, keeping others at their current best.
    Returns the best parameter set found.
    """
    defaults = deepcopy(config["defaults"])
    search_space = config["search_space"]
    judge_model = config["judge"]["model"]
    judge_weights = config["judge"]["quality_weights"]
    weights = config["scoring"]
    budget = config["budget"]["max_api_calls"]
    param_order = list(search_space.keys())

    best_params = deepcopy(defaults)

    # Track all tps/ttft values for normalization
    all_tps = []
    all_ttft = []

    # Phase 1: Evaluate baseline (defaults)
    print(f"\n  --- Baseline evaluation ---")
    phase = "baseline"
    key = _experiment_key(infra_config, model, phase, "none", *[defaults[p] for p in param_order])

    if key not in completed:
        if api_call_count[0] >= budget:
            print("  Budget exhausted!")
            return best_params

        baseline_result = evaluate_params(
            model=model, infra_config=infra_config, phase="baseline",
            param_being_optimized="none", params=defaults,
            eval_prompts=eval_prompts, base_url=base_url,
            judge_model=judge_model, judge_weights=judge_weights,
            details_path=details_path,
        )
        api_call_count[0] += len(eval_prompts)

        all_tps.append(baseline_result.avg_tokens_per_sec)
        all_ttft.append(baseline_result.avg_ttft_ms)

        # For baseline, use raw quality as composite (no normalization data yet)
        baseline_composite = baseline_result.avg_quality

        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "infra_config": infra_config,
            "model": model,
            "phase": phase,
            "param_being_optimized": "none",
            "tokens_per_sec": f"{baseline_result.avg_tokens_per_sec:.2f}",
            "ttft_ms": f"{baseline_result.avg_ttft_ms:.0f}",
            "quality_score": f"{baseline_result.avg_quality:.2f}",
            "composite_score": f"{baseline_composite:.2f}",
            "is_best": "true",
            "notes": "baseline",
        }
        for p in param_order:
            row[p] = defaults[p]
        append_tsv(tsv_path, row)
    else:
        print("  Baseline already evaluated, skipping.")
        baseline_composite = 5.0  # Placeholder for resume
        baseline_result = EvalResult(avg_quality=5.0, avg_tokens_per_sec=30.0, avg_ttft_ms=500.0, per_prompt=[])
        all_tps.append(baseline_result.avg_tokens_per_sec)
        all_ttft.append(baseline_result.avg_ttft_ms)

    best_composite = baseline_composite

    # Phase 2: Coordinate descent — optimize one parameter at a time
    for param_name in param_order:
        print(f"\n  --- Optimizing: {param_name} ---")
        phase = "sweep"

        best_value_for_param = best_params[param_name]
        best_composite_for_param = best_composite

        for value in search_space[param_name]:
            if value == best_params[param_name]:
                # Already evaluated at this value (it's the current best/default)
                continue

            trial_params = deepcopy(best_params)
            trial_params[param_name] = value

            key = _experiment_key(infra_config, model, phase, param_name, *[trial_params[p] for p in param_order])
            if key in completed:
                print(f"    {param_name}={value}: already evaluated, skipping")
                continue

            if api_call_count[0] >= budget:
                print("  Budget exhausted!")
                return best_params

            print(f"    Trying {param_name}={value}")
            result = evaluate_params(
                model=model, infra_config=infra_config, phase="sweep",
                param_being_optimized=param_name, params=trial_params,
                eval_prompts=eval_prompts, base_url=base_url,
                judge_model=judge_model, judge_weights=judge_weights,
                details_path=details_path,
            )
            api_call_count[0] += len(eval_prompts)

            all_tps.append(result.avg_tokens_per_sec)
            all_ttft.append(result.avg_ttft_ms)

            # Compute composite with current normalization ranges
            tps_range = (min(all_tps), max(all_tps))
            ttft_range = (min(all_ttft), max(all_ttft))
            composite = compute_composite(
                result.avg_quality, result.avg_tokens_per_sec, result.avg_ttft_ms,
                tps_range, ttft_range, weights,
            )

            is_best = composite > best_composite_for_param

            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "infra_config": infra_config,
                "model": model,
                "phase": phase,
                "param_being_optimized": param_name,
                "tokens_per_sec": f"{result.avg_tokens_per_sec:.2f}",
                "ttft_ms": f"{result.avg_ttft_ms:.0f}",
                "quality_score": f"{result.avg_quality:.2f}",
                "composite_score": f"{composite:.2f}",
                "is_best": str(is_best).lower(),
                "notes": "",
            }
            for p in param_order:
                row[p] = trial_params[p]
            append_tsv(tsv_path, row)

            if is_best:
                print(f"    NEW BEST: {param_name}={value} (composite {composite:.2f} > {best_composite_for_param:.2f})")
                best_value_for_param = value
                best_composite_for_param = composite

        # Lock in best value for this parameter before moving to next
        best_params[param_name] = best_value_for_param
        best_composite = best_composite_for_param
        print(f"  Best {param_name}: {best_value_for_param} (composite: {best_composite:.2f})")

    return best_params


def validate_config(config: dict) -> None:
    """Validate config.yaml has all required keys. Exits with a clear message on failure."""
    errors = []

    # infra section
    infra = config.get("infra", {})
    for key in ("ollama_host", "ollama_port", "compose_dir", "compose_project", "ollama_volume"):
        if key not in infra:
            errors.append(f"  missing: infra.{key}")

    # models
    if not config.get("models"):
        errors.append("  missing or empty: models")

    # search_space and defaults must have the same keys
    search_space = config.get("search_space", {})
    defaults = config.get("defaults", {})
    if not search_space:
        errors.append("  missing or empty: search_space")
    if not defaults:
        errors.append("  missing or empty: defaults")
    for key in search_space:
        if key not in defaults:
            errors.append(f"  search_space key '{key}' has no entry in defaults")
    for key in defaults:
        if key not in search_space:
            errors.append(f"  defaults key '{key}' has no entry in search_space")

    # scoring weights
    scoring = config.get("scoring", {})
    for key in ("quality_weight", "speed_weight", "ttft_weight"):
        if key not in scoring:
            errors.append(f"  missing: scoring.{key}")
    if not errors:
        total = sum(scoring.get(k, 0) for k in ("quality_weight", "speed_weight", "ttft_weight"))
        if abs(total - 1.0) > 0.01:
            print(f"  WARNING: scoring weights sum to {total:.3f}, expected 1.0")

    # judge quality_weights
    quality_weights = config.get("judge", {}).get("quality_weights", {})
    for key in ("correctness", "completeness", "clarity", "agent_utility"):
        if key not in quality_weights:
            errors.append(f"  missing: judge.quality_weights.{key}")
    if quality_weights:
        total = sum(quality_weights.get(k, 0) for k in ("correctness", "completeness", "clarity", "agent_utility"))
        if abs(total - 1.0) > 0.01:
            print(f"  WARNING: judge.quality_weights sum to {total:.3f}, expected 1.0")

    # budget
    if "max_api_calls" not in config.get("budget", {}):
        errors.append("  missing: budget.max_api_calls")

    if errors:
        print("ERROR: config.yaml is invalid:")
        for e in errors:
            print(e)
        sys.exit(1)



def main():
    config = load_config()
    validate_config(config)
    eval_prompts = load_eval_prompts()
    tsv_path = "results.tsv"
    details_path = "details.jsonl"
    init_tsv(tsv_path)

    infra = config["infra"]
    base_url = ollama_url(infra["ollama_host"], infra["ollama_port"])
    compose_dir = infra["compose_dir"]
    compose_project = infra["compose_project"]
    ollama_volume = infra["ollama_volume"]
    models = config["models"]
    infra_configs = config["infra_configs"]

    preflight_check(config)

    completed = load_completed_experiments(tsv_path)
    api_call_count = [0]  # Mutable counter shared across calls

    print("autotune-ollama starting")
    print(f"  Ollama: {base_url}")
    print(f"  Models: {len(models)}")
    print(f"  Infra configs: {len(infra_configs)}")
    print(f"  Eval prompts: {len(eval_prompts)}")
    print(f"  Completed experiments (resume): {len(completed)}")
    print(f"  API budget: {config['budget']['max_api_calls']}")
    print()

    for infra_config in infra_configs:
        # Switch infra config by restarting the Docker container
        if not switch_infra_config(infra_config, compose_dir, compose_project, ollama_volume):
            print(f"  SKIPPING infra config: {infra_config} (switch failed)")
            continue

        # Wait for API to come up after container restart
        if not wait_for_api(base_url, timeout=90):
            print(f"  SKIPPING infra config: {infra_config} (API not ready after 90s)")
            continue

        for model in models:
            print(f"\n{'='*60}")
            print(f"Model: {model} | Infra: {infra_config}")
            print(f"{'='*60}")

            # Pull model
            try:
                pull_model(model, base_url)
            except Exception as e:
                print(f"  SKIPPING model {model}: pull failed ({e})")
                continue

            # Check GPU fit
            if not check_gpu_fit(model, base_url):
                print(f"  SKIPPING model {model}: does not fit in GPU")
                continue

            # Warmup
            warmup(model, base_url)

            # Run coordinate descent
            best_params = coordinate_descent(
                model=model,
                infra_config=infra_config,
                config=config,
                eval_prompts=eval_prompts,
                base_url=base_url,
                tsv_path=tsv_path,
                details_path=details_path,
                completed=completed,
                api_call_count=api_call_count,
            )

            print(f"\n  Best params for {model} @ {infra_config}: {best_params}")

            if api_call_count[0] >= config["budget"]["max_api_calls"]:
                print("\nAPI budget exhausted. Stopping.")
                break

        if api_call_count[0] >= config["budget"]["max_api_calls"]:
            break

    print(f"\n{'='*60}")
    print(f"autotune-ollama complete")
    print(f"  Total API calls: {api_call_count[0]}")
    print(f"  Results: {tsv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
