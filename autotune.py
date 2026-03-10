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

from eval_harness import (
    InferenceResult,
    check_gpu_fit,
    ollama_url,
    pull_model,
    run_inference,
    wait_for_api,
    warmup,
)
from judge import judge_output


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
    with open(path) as f:
        return yaml.safe_load(f)


def load_eval_prompts(path: str = "eval_prompts.json") -> list[dict]:
    """Load eval prompts, filtering out any without reference answers."""
    with open(path) as f:
        data = json.load(f)

    prompts = data.get("coding_prompts", []) + data.get("chat_prompts", [])
    valid = [p for p in prompts if p.get("reference")]

    if not valid:
        print("ERROR: No eval prompts have reference answers. Run generate_references.py first.")
        sys.exit(1)

    print(f"Loaded {len(valid)} eval prompts with references.")
    return valid


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


def switch_infra_config(config_name: str, vm_host: str, vm_user: str, ssh_key: str) -> bool:
    """SSH to VM and run the gatekeeper script to switch infra config."""
    print(f"\n{'='*60}")
    print(f"Switching infra config: {config_name}")
    print(f"{'='*60}")

    cmd = [
        "ssh",
        "-i", os.path.expanduser(ssh_key),
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=10",
        f"{vm_user}@{vm_host}",
        f"/home/{vm_user}/bin/ollama-reconfig {config_name}",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        print(result.stdout)
        if result.returncode != 0:
            print(f"ERROR switching config: {result.stderr}")
            return False
        return True
    except subprocess.TimeoutExpired:
        print("ERROR: SSH command timed out")
        return False
    except FileNotFoundError:
        print("ERROR: ssh command not found")
        return False


def evaluate_params(
    model: str,
    params: dict,
    eval_prompts: list[dict],
    base_url: str,
    judge_model: str,
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

        # Judge quality
        try:
            scores = judge_output(
                prompt=prompt_text,
                candidate=result.response_text,
                reference=reference,
                model=judge_model,
            )
            quality = scores["overall"]
        except Exception as e:
            print(f"    Judging failed for {prompt_id}: {e}")
            quality = 1.0
            scores = {"overall": 1.0, "brief_rationale": f"Judge error: {e}"}

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

        print(f"    {prompt_id}: quality={quality:.1f} tps={result.tokens_per_sec:.1f} ttft={result.ttft_ms:.0f}ms")

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

        baseline_result = evaluate_params(model, defaults, eval_prompts, base_url, judge_model)
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
            result = evaluate_params(model, trial_params, eval_prompts, base_url, judge_model)
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


def main():
    config = load_config()
    eval_prompts = load_eval_prompts()
    tsv_path = "results.tsv"
    init_tsv(tsv_path)

    vm = config["vm"]
    base_url = ollama_url(vm["host"], vm["ollama_port"])
    models = config["models"]
    infra_configs = config["infra_configs"]

    completed = load_completed_experiments(tsv_path)
    api_call_count = [0]  # Mutable counter shared across calls

    print(f"autotune-ollama starting")
    print(f"  VM: {vm['host']}:{vm['ollama_port']}")
    print(f"  Models: {len(models)}")
    print(f"  Infra configs: {len(infra_configs)}")
    print(f"  Eval prompts: {len(eval_prompts)}")
    print(f"  Completed experiments (resume): {len(completed)}")
    print(f"  API budget: {config['budget']['max_api_calls']}")
    print()

    # Overall best tracking
    overall_best = {"composite": 0, "model": "", "infra": "", "params": {}}

    for infra_config in infra_configs:
        # Switch infra config via gatekeeper
        if not switch_infra_config(infra_config, vm["host"], vm["user"], vm["ssh_key"]):
            print(f"  SKIPPING infra config: {infra_config} (switch failed)")
            continue

        # Wait for API
        if not wait_for_api(base_url, timeout=90):
            print(f"  SKIPPING infra config: {infra_config} (API not ready)")
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
