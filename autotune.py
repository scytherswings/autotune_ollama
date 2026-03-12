"""Main orchestration loop: coordinate descent parameter optimization for Ollama models."""

import csv
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()  # Load .env before anything touches os.environ

sys.stdout.reconfigure(line_buffering=True)  # Flush after each line when piped to tee/log

from eval_harness import (
    OllamaOomError,
    check_gpu_fit,
    check_objective_criteria,
    get_ollama_allocation,
    unload_model,
    ollama_url,
    pull_model,
    run_chat_inference,
    run_inference,
    run_tool_inference,
    wait_for_api,
    warmup,
)
from judge import (
    batch_judge,
    collect_judge_batch,
    judge_chat,
    judge_output,
    judge_tool_call,
    submit_judge_batch,
)
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
    "total_time_ms",
    "objective_score",
    "judge_score",
    "quality_score",
    "composite_score",
    "is_best",
    "notes",
]


class TpsFailure(Exception):
    """Raised when TPS drops below the minimum — signals CPU fallback for the whole run."""


@dataclass
class EvalResult:
    """Aggregated result from evaluating a full prompt suite."""
    avg_quality: float           # weighted blend across active eval types
    avg_objective_score: float   # avg for prompts with objective checks (tool_call)
    avg_judge_score: float
    avg_tokens_per_sec: float
    avg_ttft_ms: float
    avg_total_time_ms: float     # wall-clock time per prompt (includes thinking for qwen3 etc.)
    per_prompt: list
    quality_by_type: dict = field(default_factory=dict)  # {"coding": float, "tool_call": float}
    failed_count: int = 0


@dataclass
class PendingEval:
    """In-flight evaluation: inferences done, judge batch submitted, collection deferred."""
    model: str
    judge_model: str
    batch_id: str | None              # None = submission failed, collect_judge_batch will sync fallback
    required_keys_by_id: dict
    inference_data: list[dict]        # per-prompt inference results; also used as items for fallback
    tps_values: list[float]
    ttft_values: list[float]
    total_time_values: list[float]
    per_prompt: list                  # failure entries from inference phase
    failed_count: int
    load_time: str = ""               # ISO timestamp before first inference (for GPU alloc logging)


def load_config(path: str = "config.yaml") -> dict:
    if not Path(path).exists():
        print(f"ERROR: {path} not found.")
        print(f"  Copy config.yaml.example to config.yaml and fill in your settings.")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def load_eval_prompts(config: dict) -> list[dict]:
    """Load eval prompts from unified eval_prompts.json, filtered by config eval.types.

    - Coding prompts get reference answers merged from references.json.
    - Tool-call prompts get tool schemas inlined from tool_sets.
    - Only categories listed in config.eval.types are included.
    """
    eval_cfg = config.get("eval", {})
    active_types = set(eval_cfg.get("types", ["coding", "tool_call"]))
    prompts_path = eval_cfg.get("prompts_file", "eval_prompts.json")

    if not Path(prompts_path).exists():
        print(f"ERROR: {prompts_path} not found.")
        sys.exit(1)

    with open(prompts_path) as f:
        data = json.load(f)

    tool_sets = data.get("tool_sets", {})

    # Load references (coding answers + tool_call ideal args)
    refs: dict[str, str] = {}
    refs_path = "references.json"
    if Path(refs_path).exists():
        with open(refs_path) as f:
            refs = json.load(f)

    prompts = []

    if "coding" in active_types:
        skipped = 0
        for p in data.get("coding_prompts", []):
            if p["id"] not in refs:
                skipped += 1
                continue
            prompts.append({**p, "reference": refs[p["id"]]})
        if skipped:
            print(f"  WARNING: {skipped} coding prompt(s) skipped (no reference — run generate_references.py).")

    if "tool_call" in active_types:
        for p in data.get("tool_call_prompts", []):
            entry = dict(p)
            ts_key = entry.pop("tool_set", None)
            if ts_key and ts_key in tool_sets:
                entry["tools"] = tool_sets[ts_key]
            entry["reference"] = refs.get(p["id"])  # None if not yet generated
            prompts.append(entry)

    if "chat" in active_types:
        for p in data.get("chat_prompts", []):
            entry = dict(p)
            entry["reference"] = refs.get(p["id"])
            prompts.append(entry)

    if not prompts:
        print("ERROR: No prompts loaded. Check eval.types in config.yaml and run generate_references.py.")
        sys.exit(1)

    counts = {}
    for p in prompts:
        t = p.get("category", "unknown")
        counts[t] = counts.get(t, 0) + 1
    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    print(f"Loaded {len(prompts)} eval prompts ({summary}).")
    return prompts


def load_completed_experiments(tsv_path: str) -> tuple[set[str], dict[str, float]]:
    """Load completed experiment keys and composite scores from results.tsv for resume."""
    completed = set()
    scores = {}
    if not Path(tsv_path).exists():
        return completed, scores

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
            try:
                scores[key] = float(row.get("composite_score", 0))
            except (ValueError, TypeError):
                scores[key] = 0.0

    return completed, scores


def _experiment_key(*args) -> str:
    return "|".join(str(a) for a in args)


def init_tsv(tsv_path: str) -> None:
    if not Path(tsv_path).exists():
        with open(tsv_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(TSV_COLUMNS)


def append_tsv(tsv_path: str, row: dict) -> None:
    with open(tsv_path, "a", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([row.get(col, "") for col in TSV_COLUMNS])


def switch_infra_config(
    config_name: str,
    compose_dir: str,
    project_name: str,
    ollama_volume: str,
) -> bool:
    """Switch Ollama infra config by restarting the Docker container."""
    compose_file = Path(compose_dir) / f"docker-compose.{config_name}.yml"
    if not compose_file.exists():
        print(f"ERROR: Compose file not found: {compose_file}")
        return False

    print(f"\n{'='*60}")
    print(f"Switching infra config: {config_name}")
    print(f"{'='*60}")

    result = subprocess.run(
        ["docker", "ps", "-q", "--filter", "publish=11434"],
        capture_output=True, text=True,
    )
    stale = [c for c in result.stdout.strip().split() if c]
    if stale:
        print(f"  Stopping {len(stale)} stale container(s)...")
        subprocess.run(["docker", "stop"] + stale, capture_output=True)

    env = os.environ.copy()
    env["OLLAMA_VOLUME"] = ollama_volume

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


def compute_quality(
    judge_scores: dict,
    prompt_type: str,
    weights: dict,
    objective_score: float | None = None,
) -> tuple[float, float]:
    """Compute quality score (0–10) from judge sub-scores, blended with objective for tool_call.

    Returns (quality_0_to_10, judge_score_0_to_10).
    """
    if prompt_type == "coding":
        sub_weights = {
            "correctness":   weights.get("coding_correctness", 0.40),
            "completeness":  weights.get("coding_completeness", 0.30),
            "clarity":       weights.get("coding_clarity", 0.20),
            "agent_utility": weights.get("coding_agent_utility", 0.10),
        }
        judge_score = sum(judge_scores.get(k, 5.0) * w for k, w in sub_weights.items())
        quality = judge_score  # coding quality is pure semantic judge

    elif prompt_type == "tool_call":
        sub_weights = {
            "arg_correctness": weights.get("tool_arg_correctness", 0.60),
            "tool_selection":  weights.get("tool_selection", 0.40),
        }
        judge_score = sum(judge_scores.get(k, 5.0) * w for k, w in sub_weights.items())
        obj_weight = weights.get("objective_weight", 0.50)
        judge_weight = weights.get("judge_weight", 0.50)
        quality = (objective_score or 1.0) * obj_weight * 10 + judge_score * judge_weight

    elif prompt_type == "chat":
        sub_weights = {
            "instruction_following": weights.get("chat_instruction_following", 0.25),
            "content_quality":       weights.get("chat_content_quality", 0.30),
            "professionalism":       weights.get("chat_professionalism", 0.20),
            "conciseness":           weights.get("chat_conciseness", 0.15),
            "context_retention":     weights.get("chat_context_retention", 0.10),
        }
        judge_score = sum(judge_scores.get(k, 5.0) * w for k, w in sub_weights.items())
        quality = judge_score  # chat quality is pure semantic judge

    else:
        judge_score = 5.0
        quality = 5.0

    return round(quality, 4), round(judge_score, 4)


def append_details(
    details_path: str,
    infra_config: str,
    model: str,
    phase: str,
    param_being_optimized: str,
    params: dict,
    prompt_id: str,
    prompt_type: str,
    objective_score: float,
    judge_scores: dict,
    judge_score: float,
    quality: float,
    tokens_per_sec: float,
    ttft_ms: float,
    used_native_tools: bool = False,
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
        "prompt_type": prompt_type,
        "objective_score": round(objective_score, 4),
        "judge_score": round(judge_score, 4),
        **{k: v for k, v in judge_scores.items() if k != "brief_rationale"},
        "quality": round(quality, 4),
        "brief_rationale": judge_scores.get("brief_rationale", ""),
        "tokens_per_sec": round(tokens_per_sec, 2),
        "ttft_ms": round(ttft_ms, 1),
        "used_native_tools": used_native_tools,
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
    type_weights: dict,
    details_path: str,
    min_tokens_per_sec: float = 0,
) -> EvalResult:
    """Run all eval prompts, batch-judge all at once, return aggregated results.

    Phase 1: run all inferences sequentially (prints TPS/TTFT as they complete)
    Phase 2: submit all judge requests as a Message Batch (50% cheaper)
    Phase 3: process results, log details, print quality scores
    """
    per_type_quality: dict[str, list[float]] = {}
    per_type_obj: list[float] = []
    all_judge_scores: list[float] = []
    tps_values: list[float] = []
    ttft_values: list[float] = []
    total_time_values: list[float] = []
    per_prompt = []
    failed_count = 0

    # ── Phase 1: Inference ───────────────────────────────────────────────────
    inference_data = []  # one dict per successful inference

    for prompt_entry in eval_prompts:
        prompt_id = prompt_entry["id"]
        prompt_type = prompt_entry.get("category", "coding")

        try:
            if prompt_type == "coding":
                result = run_inference(
                    model=model, prompt=prompt_entry["prompt"],
                    options=params, base_url=base_url,
                )
                response_text = result.response_text
                tool_calls = []
                used_native = False

            elif prompt_type == "tool_call":
                result = run_tool_inference(
                    model=model, system_prompt=prompt_entry["system_prompt"],
                    user_message=prompt_entry["user_message"],
                    tools=prompt_entry.get("tools", []),
                    options=params, base_url=base_url,
                )
                response_text = result.response_text
                tool_calls = result.tool_calls
                used_native = result.used_native_tools

            elif prompt_type == "chat":
                result = run_chat_inference(
                    model=model, system_prompt=prompt_entry.get("system_prompt"),
                    turns=[t["content"] for t in prompt_entry["turns"]],
                    options=params, base_url=base_url,
                )
                response_text = result.response_text
                tool_calls = []
                used_native = False

            else:
                print(f"    Skipping unknown prompt type '{prompt_type}' for {prompt_id}")
                continue

            tps = result.tokens_per_sec
            ttft = result.ttft_ms
            total_time = result.total_duration_ns / 1_000_000

            if min_tokens_per_sec > 0 and tps < min_tokens_per_sec:
                raise TpsFailure(
                    f"TPS too low ({tps:.1f} < {min_tokens_per_sec} minimum) — likely CPU fallback"
                )

        except TpsFailure:
            raise
        except Exception as e:
            print(f"    Inference failed for {prompt_id}: {e}")
            per_prompt.append({"id": prompt_id, "type": prompt_type, "error": str(e)})
            failed_count += 1
            continue

        # Objective checks (free — no API call)
        if prompt_type == "tool_call":
            objective = check_objective_criteria(prompt_entry, tool_calls, response_text)
            objective_score = objective.objective_score
            per_type_obj.append(objective_score)
        else:
            objective_score = 1.0

        print(f"    {prompt_id}: tps={tps:.1f} ttft={ttft:.0f}ms", flush=True)

        inference_data.append({
            "custom_id": prompt_id,
            "prompt_type": prompt_type,
            "prompt_entry": prompt_entry,
            "response_text": response_text,
            "tool_calls": tool_calls,
            "reference": prompt_entry.get("reference"),
            "objective_score": objective_score,
            "tps": tps, "ttft": ttft, "total_time": total_time,
            "used_native": used_native,
        })

        tps_values.append(tps)
        ttft_values.append(ttft)
        total_time_values.append(total_time)

    if not inference_data:
        return EvalResult(
            avg_quality=0, avg_objective_score=0, avg_judge_score=0,
            avg_tokens_per_sec=0, avg_ttft_ms=float("inf"), avg_total_time_ms=float("inf"),
            per_prompt=per_prompt, quality_by_type={}, failed_count=failed_count,
        )

    # ── Phase 2: Batch judge ─────────────────────────────────────────────────
    scores_map = batch_judge(inference_data, judge_model)

    # ── Phase 3: Process results ─────────────────────────────────────────────
    for d in inference_data:
        prompt_id = d["custom_id"]
        prompt_type = d["prompt_type"]
        prompt_entry = d["prompt_entry"]
        scores = scores_map[prompt_id]
        objective_score = d["objective_score"]
        tps, ttft = d["tps"], d["ttft"]
        tool_calls = d["tool_calls"]
        used_native = d["used_native"]

        quality, judge_score = compute_quality(scores, prompt_type, judge_weights, objective_score)

        append_details(
            details_path=details_path, infra_config=infra_config, model=model,
            phase=phase, param_being_optimized=param_being_optimized, params=params,
            prompt_id=prompt_id, prompt_type=prompt_type, objective_score=objective_score,
            judge_scores=scores, judge_score=judge_score, quality=quality,
            tokens_per_sec=tps, ttft_ms=ttft, used_native_tools=used_native,
        )

        per_type_quality.setdefault(prompt_type, []).append(quality)
        all_judge_scores.append(judge_score)

        per_prompt.append({
            "id": prompt_id, "type": prompt_type, "quality": quality,
            "objective": objective_score, "judge": judge_score,
            "tokens_per_sec": tps, "ttft_ms": ttft,
            "rationale": scores.get("brief_rationale", ""),
        })

        if prompt_type == "coding":
            print(f"    {prompt_id}: quality={quality:.2f} "
                  f"(c={scores.get('correctness',0):.0f} co={scores.get('completeness',0):.0f} "
                  f"cl={scores.get('clarity',0):.0f} a={scores.get('agent_utility',0):.0f})")
        elif prompt_type == "chat":
            n_turns = len(prompt_entry.get("turns", []))
            print(f"    {prompt_id}: quality={quality:.2f} "
                  f"(if={scores.get('instruction_following',0):.0f} cq={scores.get('content_quality',0):.0f} "
                  f"pr={scores.get('professionalism',0):.0f} cn={scores.get('conciseness',0):.0f} "
                  f"cr={scores.get('context_retention',0):.0f}) turns={n_turns}")
        else:
            called = tool_calls[0]["function"]["name"] if tool_calls else "none"
            native = "native" if used_native else "text"
            print(f"    {prompt_id}: quality={quality:.2f} obj={objective_score:.2f} "
                  f"judge={judge_score:.1f} tool={called} ({native})")

    if not per_type_quality:
        return EvalResult(
            avg_quality=0, avg_objective_score=0, avg_judge_score=0,
            avg_tokens_per_sec=0, avg_ttft_ms=float("inf"), avg_total_time_ms=float("inf"),
            per_prompt=per_prompt, quality_by_type={}, failed_count=failed_count,
        )

    # Per-type averages
    quality_by_type = {t: sum(v) / len(v) for t, v in per_type_quality.items()}

    # Weighted blend across types
    if type_weights and len(quality_by_type) > 1:
        total_w = sum(type_weights.get(t, 0) for t in quality_by_type)
        if total_w > 0:
            blended = sum(quality_by_type[t] * type_weights.get(t, 0) for t in quality_by_type) / total_w
        else:
            blended = sum(quality_by_type.values()) / len(quality_by_type)
    else:
        blended = sum(quality_by_type.values()) / len(quality_by_type)

    # Print per-type summary when running mixed suite
    if len(quality_by_type) > 1:
        type_summary = "  ".join(f"{t}={v:.2f}" for t, v in sorted(quality_by_type.items()))
        print(f"    → per-type: {type_summary}  blended={blended:.2f}")

    avg_obj = sum(per_type_obj) / len(per_type_obj) if per_type_obj else 1.0
    avg_judge = sum(all_judge_scores) / len(all_judge_scores)

    return EvalResult(
        avg_quality=round(blended, 4),
        avg_objective_score=round(avg_obj, 4),
        avg_judge_score=round(avg_judge, 4),
        avg_tokens_per_sec=sum(tps_values) / len(tps_values),
        avg_ttft_ms=sum(ttft_values) / len(ttft_values),
        avg_total_time_ms=sum(total_time_values) / len(total_time_values),
        per_prompt=per_prompt,
        quality_by_type=quality_by_type,
        failed_count=failed_count,
    )


def start_eval(
    model: str,
    infra_config: str,
    phase: str,
    param_being_optimized: str,
    params: dict,
    eval_prompts: list[dict],
    base_url: str,
    judge_model: str,
    judge_weights: dict,
    type_weights: dict,
    details_path: str,
    min_tokens_per_sec: float = 0,
    load_time: str = "",
) -> PendingEval:
    """Run all inferences and submit judge batch. Returns immediately after batch submit.

    Raises TpsFailure or OllamaOomError if a critical inference failure occurs.
    The caller should catch these and not call finish_eval.
    """
    inference_data: list[dict] = []
    tps_values: list[float] = []
    ttft_values: list[float] = []
    total_time_values: list[float] = []
    per_prompt: list = []
    failed_count = 0

    for prompt_entry in eval_prompts:
        prompt_id = prompt_entry["id"]
        prompt_type = prompt_entry.get("category", "coding")

        try:
            if prompt_type == "coding":
                result = run_inference(
                    model=model, prompt=prompt_entry["prompt"],
                    options=params, base_url=base_url,
                )
                response_text = result.response_text
                tool_calls = []
                used_native = False

            elif prompt_type == "tool_call":
                result = run_tool_inference(
                    model=model, system_prompt=prompt_entry["system_prompt"],
                    user_message=prompt_entry["user_message"],
                    tools=prompt_entry.get("tools", []),
                    options=params, base_url=base_url,
                )
                response_text = result.response_text
                tool_calls = result.tool_calls
                used_native = result.used_native_tools

            elif prompt_type == "chat":
                result = run_chat_inference(
                    model=model, system_prompt=prompt_entry.get("system_prompt"),
                    turns=[t["content"] for t in prompt_entry["turns"]],
                    options=params, base_url=base_url,
                )
                response_text = result.response_text
                tool_calls = []
                used_native = False

            else:
                print(f"    Skipping unknown prompt type '{prompt_type}' for {prompt_id}")
                continue

            tps = result.tokens_per_sec
            ttft = result.ttft_ms
            total_time = result.total_duration_ns / 1_000_000

            if min_tokens_per_sec > 0 and tps < min_tokens_per_sec:
                raise TpsFailure(
                    f"TPS too low ({tps:.1f} < {min_tokens_per_sec} minimum) — likely CPU fallback"
                )

        except TpsFailure:
            raise
        except Exception as e:
            print(f"    Inference failed for {prompt_id}: {e}")
            per_prompt.append({"id": prompt_id, "type": prompt_type, "error": str(e)})
            failed_count += 1
            continue

        if prompt_type == "tool_call":
            objective = check_objective_criteria(prompt_entry, tool_calls, response_text)
            objective_score = objective.objective_score
        else:
            objective_score = 1.0

        print(f"    {prompt_id}: tps={tps:.1f} ttft={ttft:.0f}ms", flush=True)

        inference_data.append({
            "custom_id": prompt_id,
            "prompt_type": prompt_type,
            "prompt_entry": prompt_entry,
            "response_text": response_text,
            "tool_calls": tool_calls,
            "reference": prompt_entry.get("reference"),
            "objective_score": objective_score,
            "tps": tps, "ttft": ttft, "total_time": total_time,
            "used_native": used_native,
        })

        tps_values.append(tps)
        ttft_values.append(ttft)
        total_time_values.append(total_time)

    # Submit batch (non-blocking — inference N+1 will run while this processes)
    if inference_data:
        submit_result = submit_judge_batch(inference_data, judge_model)
        if submit_result is None:
            batch_id, required_keys_by_id = None, {}
        else:
            batch_id, required_keys_by_id = submit_result
    else:
        batch_id, required_keys_by_id = None, {}

    return PendingEval(
        model=model,
        judge_model=judge_model,
        batch_id=batch_id,
        required_keys_by_id=required_keys_by_id,
        inference_data=inference_data,
        tps_values=tps_values,
        ttft_values=ttft_values,
        total_time_values=total_time_values,
        per_prompt=per_prompt,
        failed_count=failed_count,
        load_time=load_time,
    )


def finish_eval(
    pending: PendingEval,
    judge_weights: dict,
    type_weights: dict,
    infra_config: str,
    phase: str,
    param_being_optimized: str,
    params: dict,
    details_path: str,
    timeout_s: float = 300,
    poll_interval_s: float = 10,
) -> EvalResult:
    """Collect judge batch results and process into EvalResult.

    Blocks until the batch is complete (or times out and falls back to sync).
    """
    model = pending.model
    inference_data = pending.inference_data
    per_prompt = list(pending.per_prompt)  # copy — preserve inference failure entries
    failed_count = pending.failed_count

    if not inference_data:
        return EvalResult(
            avg_quality=0, avg_objective_score=0, avg_judge_score=0,
            avg_tokens_per_sec=0, avg_ttft_ms=float("inf"), avg_total_time_ms=float("inf"),
            per_prompt=per_prompt, quality_by_type={}, failed_count=failed_count,
        )

    scores_map = collect_judge_batch(
        pending.batch_id, pending.required_keys_by_id,
        inference_data, pending.judge_model, timeout_s, poll_interval_s,
    )

    # Process results
    per_type_quality: dict[str, list[float]] = {}
    all_judge_scores: list[float] = []

    for d in inference_data:
        prompt_id = d["custom_id"]
        prompt_type = d["prompt_type"]
        prompt_entry = d["prompt_entry"]
        scores = scores_map[prompt_id]
        objective_score = d["objective_score"]
        tps, ttft = d["tps"], d["ttft"]
        tool_calls = d["tool_calls"]
        used_native = d["used_native"]

        quality, judge_score = compute_quality(scores, prompt_type, judge_weights, objective_score)

        append_details(
            details_path=details_path, infra_config=infra_config, model=model,
            phase=phase, param_being_optimized=param_being_optimized, params=params,
            prompt_id=prompt_id, prompt_type=prompt_type, objective_score=objective_score,
            judge_scores=scores, judge_score=judge_score, quality=quality,
            tokens_per_sec=tps, ttft_ms=ttft, used_native_tools=used_native,
        )

        per_type_quality.setdefault(prompt_type, []).append(quality)
        all_judge_scores.append(judge_score)

        per_prompt.append({
            "id": prompt_id, "type": prompt_type, "quality": quality,
            "objective": objective_score, "judge": judge_score,
            "tokens_per_sec": tps, "ttft_ms": ttft,
            "rationale": scores.get("brief_rationale", ""),
        })

        if prompt_type == "coding":
            print(f"    {prompt_id}: quality={quality:.2f} "
                  f"(c={scores.get('correctness',0):.0f} co={scores.get('completeness',0):.0f} "
                  f"cl={scores.get('clarity',0):.0f} a={scores.get('agent_utility',0):.0f})")
        elif prompt_type == "chat":
            n_turns = len(prompt_entry.get("turns", []))
            print(f"    {prompt_id}: quality={quality:.2f} "
                  f"(if={scores.get('instruction_following',0):.0f} cq={scores.get('content_quality',0):.0f} "
                  f"pr={scores.get('professionalism',0):.0f} cn={scores.get('conciseness',0):.0f} "
                  f"cr={scores.get('context_retention',0):.0f}) turns={n_turns}")
        else:
            called = tool_calls[0]["function"]["name"] if tool_calls else "none"
            native = "native" if used_native else "text"
            print(f"    {prompt_id}: quality={quality:.2f} obj={objective_score:.2f} "
                  f"judge={judge_score:.1f} tool={called} ({native})")

    if not per_type_quality:
        return EvalResult(
            avg_quality=0, avg_objective_score=0, avg_judge_score=0,
            avg_tokens_per_sec=0, avg_ttft_ms=float("inf"), avg_total_time_ms=float("inf"),
            per_prompt=per_prompt, quality_by_type={}, failed_count=failed_count,
        )

    quality_by_type = {t: sum(v) / len(v) for t, v in per_type_quality.items()}

    if type_weights and len(quality_by_type) > 1:
        total_w = sum(type_weights.get(t, 0) for t in quality_by_type)
        if total_w > 0:
            blended = sum(quality_by_type[t] * type_weights.get(t, 0) for t in quality_by_type) / total_w
        else:
            blended = sum(quality_by_type.values()) / len(quality_by_type)
    else:
        blended = sum(quality_by_type.values()) / len(quality_by_type)

    if len(quality_by_type) > 1:
        type_summary = "  ".join(f"{t}={v:.2f}" for t, v in sorted(quality_by_type.items()))
        print(f"    → per-type: {type_summary}  blended={blended:.2f}")

    per_type_obj = [d["objective_score"] for d in inference_data if d["prompt_type"] == "tool_call"]
    avg_obj = sum(per_type_obj) / len(per_type_obj) if per_type_obj else 1.0
    avg_judge = sum(all_judge_scores) / len(all_judge_scores)

    return EvalResult(
        avg_quality=round(blended, 4),
        avg_objective_score=round(avg_obj, 4),
        avg_judge_score=round(avg_judge, 4),
        avg_tokens_per_sec=sum(pending.tps_values) / len(pending.tps_values),
        avg_ttft_ms=sum(pending.ttft_values) / len(pending.ttft_values),
        avg_total_time_ms=sum(pending.total_time_values) / len(pending.total_time_values),
        per_prompt=per_prompt,
        quality_by_type=quality_by_type,
        failed_count=failed_count,
    )


def compute_composite(
    quality: float,
    total_time_ms: float,
    total_time_range: tuple[float, float],
    weights: dict,
) -> float:
    """Compute composite score: quality + latency (total wall-clock time per prompt).

    total_time_ms captures the full cost — thinking time, generation, and prompt
    eval — making it the right signal for thinking models like qwen3 where TTFT
    and TPS individually look fine but total time can be 10× longer.
    """
    t_min, t_max = total_time_range
    # Lower total_time is better, so invert: best time → 1.0, worst → 0.0
    latency_norm = 1.0 - (total_time_ms - t_min) / (t_max - t_min) if t_max > t_min else 0.5
    latency_norm = max(0, min(1, latency_norm))
    quality_norm = (quality - 1) / 9

    composite = (
        quality_norm * weights["quality_weight"]
        + latency_norm * weights["latency_weight"]
    )
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
    completed_scores: dict[str, float],
    api_call_count: list[int],
) -> dict:
    """Run coordinate descent optimization for a single model+infra combo."""
    defaults = deepcopy(config["defaults"])
    search_space = config["search_space"]
    judge_model = config["judge"]["model"]
    judge_weights = config["judge"]["quality_weights"]
    type_weights = config.get("eval", {}).get("type_weights", {})
    weights = config["scoring"]
    budget = config["budget"]["max_api_calls"]
    min_tps = config["budget"].get("min_tokens_per_sec", 0)
    compose_project = config["infra"].get("compose_project", "ollama-autotune")
    ollama_container = f"{compose_project}-ollama-1"
    # param_order includes ALL defaults keys (not just swept ones) so experiment keys
    # remain stable when params are removed from search_space but kept in defaults.
    param_order = list(defaults.keys())

    best_params = deepcopy(defaults)
    all_total_time = []

    # Baseline uses full-quality Sonnet; sweep uses cheaper Haiku for relative ranking
    judge_model_baseline = judge_model
    judge_model_sweep = config["judge"].get("sweep_model", judge_model)

    # Phase 1: Baseline
    print(f"\n  --- Baseline evaluation ---")
    phase = "baseline"
    key = _experiment_key(infra_config, model, phase, "none", *[defaults[p] for p in param_order])

    if key not in completed:
        if api_call_count[0] >= budget:
            print("  Budget exhausted!")
            return best_params

        try:
            baseline_result = evaluate_params(
                model=model, infra_config=infra_config, phase="baseline",
                param_being_optimized="none", params=defaults,
                eval_prompts=eval_prompts, base_url=base_url,
                judge_model=judge_model_baseline, judge_weights=judge_weights,
                type_weights=type_weights,
                details_path=details_path, min_tokens_per_sec=min_tps,
            )
        except TpsFailure as e:
            print(f"  Baseline aborted — {e}. Skipping model.")
            return best_params
        api_call_count[0] += len(eval_prompts)           # inference calls
        api_call_count[0] += len(eval_prompts)           # judge calls (batched but still N requests)

        all_total_time.append(baseline_result.avg_total_time_ms)
        baseline_composite = baseline_result.avg_quality

        type_notes = " ".join(f"{t}={v:.2f}" for t, v in sorted(baseline_result.quality_by_type.items()))
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "infra_config": infra_config,
            "model": model,
            "phase": phase,
            "param_being_optimized": "none",
            "tokens_per_sec": f"{baseline_result.avg_tokens_per_sec:.2f}",
            "ttft_ms": f"{baseline_result.avg_ttft_ms:.0f}",
            "total_time_ms": f"{baseline_result.avg_total_time_ms:.0f}",
            "objective_score": f"{baseline_result.avg_objective_score:.2f}",
            "judge_score": f"{baseline_result.avg_judge_score:.2f}",
            "quality_score": f"{baseline_result.avg_quality:.2f}",
            "composite_score": f"{baseline_composite:.2f}",
            "is_best": "true",
            "notes": f"baseline {type_notes}",
        }
        for p in param_order:
            row[p] = defaults[p]
        append_tsv(tsv_path, row)
    else:
        print("  Baseline already evaluated, skipping.")
        baseline_composite = 5.0
        baseline_result = EvalResult(
            avg_quality=5.0, avg_objective_score=0.5, avg_judge_score=5.0,
            avg_tokens_per_sec=30.0, avg_ttft_ms=500.0, avg_total_time_ms=30000.0, per_prompt=[],
        )
        all_total_time.append(baseline_result.avg_total_time_ms)

    best_composite = baseline_composite

    # Phase 2: Coordinate descent
    for param_name in param_order:
        if param_name not in search_space:
            continue  # fixed param, not swept
        print(f"\n  --- Optimizing: {param_name} ---")
        phase = "sweep"

        best_value_for_param = best_params[param_name]
        best_composite_for_param = best_composite

        # Pass 1: run inferences + submit judge batches for all values sequentially.
        # Each batch processes on Anthropic's end while the next inference runs locally.
        pending_evals: list[dict] = []
        budget_hit = False

        for value in search_space[param_name]:
            if value == best_params[param_name]:
                continue

            trial_params = deepcopy(best_params)
            trial_params[param_name] = value

            key = _experiment_key(infra_config, model, phase, param_name, *[trial_params[p] for p in param_order])
            if key in completed:
                pending_evals.append({
                    "status": "already_done", "value": value, "key": key,
                    "trial_params": trial_params, "pending": None,
                    "prior_composite": completed_scores.get(key, 0.0),
                })
                continue

            if api_call_count[0] >= budget:
                print("  Budget exhausted!")
                budget_hit = True
                break

            print(f"    Trying {param_name}={value}")
            if param_name == "num_ctx":
                unload_model(model, base_url)
            load_time = datetime.now(timezone.utc).isoformat()

            try:
                pending = start_eval(
                    model=model, infra_config=infra_config, phase="sweep",
                    param_being_optimized=param_name, params=trial_params,
                    eval_prompts=eval_prompts, base_url=base_url,
                    judge_model=judge_model_sweep, judge_weights=judge_weights,
                    type_weights=type_weights, details_path=details_path,
                    min_tokens_per_sec=min_tps, load_time=load_time,
                )
            except (TpsFailure, OllamaOomError) as e:
                label = "tps_fail" if isinstance(e, TpsFailure) else "oom"
                print(f"    {param_name}={value} aborted — {e}")
                append_tsv(tsv_path, {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "infra_config": infra_config, "model": model,
                    "phase": phase, "param_being_optimized": param_name,
                    **{p: trial_params[p] for p in param_order},
                    "tokens_per_sec": "0", "ttft_ms": "0", "total_time_ms": "0",
                    "objective_score": "0", "judge_score": "0",
                    "quality_score": "0", "composite_score": "0",
                    "is_best": "false", "notes": label,
                })
                completed.add(key)
                completed_scores[key] = 0.0
                if param_name == "num_ctx":
                    break  # skip larger ctx values
                continue

            api_call_count[0] += len(eval_prompts)   # inference calls
            pending_evals.append({
                "status": "pending", "value": value, "key": key,
                "trial_params": trial_params, "pending": pending,
                "prior_composite": None,
            })

        # Pass 2: collect judge results in order (batches are likely already done).
        for item in pending_evals:
            value = item["value"]
            key = item["key"]
            trial_params = item["trial_params"]

            if item["status"] == "already_done":
                prior_composite = item["prior_composite"]
                if prior_composite > best_composite_for_param:
                    best_value_for_param = value
                    best_composite_for_param = prior_composite
                print(f"    {param_name}={value}: already evaluated (composite={prior_composite:.2f})")
                continue

            # status == "pending"
            pending = item["pending"]

            # GPU alloc logging for num_ctx (logged here so it appears before quality scores)
            if param_name == "num_ctx":
                alloc = get_ollama_allocation(ollama_container, pending.load_time)
                if alloc:
                    gpu_layers = alloc.get("gpu_layers", "?")
                    total_layers = alloc.get("total_layers", "?")
                    kv_gpu = alloc.get("kv_gpu_mb", 0)
                    kv_cpu = alloc.get("kv_cpu_mb", 0)
                    print(f"    ollama: {gpu_layers}/{total_layers} layers on GPU  |  KV: {kv_gpu:.0f}MB GPU + {kv_cpu:.0f}MB CPU")

            result = finish_eval(
                pending=pending,
                judge_weights=judge_weights,
                type_weights=type_weights,
                infra_config=infra_config,
                phase="sweep",
                param_being_optimized=param_name,
                params=trial_params,
                details_path=details_path,
            )
            api_call_count[0] += len(eval_prompts)   # judge calls

            if param_name == "num_ctx" and result.failed_count > 0:
                # Generic (non-OOM) failures at this ctx — log but don't skip larger values
                print(f"    num_ctx={value} had {result.failed_count} non-OOM failure(s)")

            all_total_time.append(result.avg_total_time_ms)

            total_time_range = (min(all_total_time), max(all_total_time))
            composite = compute_composite(
                result.avg_quality, result.avg_total_time_ms,
                total_time_range, weights,
            )

            is_best = composite > best_composite_for_param

            type_notes = " ".join(f"{t}={v:.2f}" for t, v in sorted(result.quality_by_type.items()))
            row = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "infra_config": infra_config,
                "model": model,
                "phase": phase,
                "param_being_optimized": param_name,
                "tokens_per_sec": f"{result.avg_tokens_per_sec:.2f}",
                "ttft_ms": f"{result.avg_ttft_ms:.0f}",
                "total_time_ms": f"{result.avg_total_time_ms:.0f}",
                "objective_score": f"{result.avg_objective_score:.2f}",
                "judge_score": f"{result.avg_judge_score:.2f}",
                "quality_score": f"{result.avg_quality:.2f}",
                "composite_score": f"{composite:.2f}",
                "is_best": str(is_best).lower(),
                "notes": type_notes,
            }
            for p in param_order:
                row[p] = trial_params[p]
            append_tsv(tsv_path, row)

            completed.add(key)
            completed_scores[key] = composite

            if is_best:
                print(f"    NEW BEST: {param_name}={value} (composite {composite:.2f} > {best_composite_for_param:.2f})")
                best_value_for_param = value
                best_composite_for_param = composite

        if budget_hit:
            return best_params

        best_params[param_name] = best_value_for_param
        best_composite = best_composite_for_param
        print(f"  Best {param_name}: {best_value_for_param} (composite: {best_composite:.2f})")

    # Write a sentinel so future runs can skip this model+infra without loading it
    append_tsv(tsv_path, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "infra_config": infra_config, "model": model,
        "phase": "complete", "param_being_optimized": "none",
        **{p: best_params[p] for p in param_order},
        "tokens_per_sec": "", "ttft_ms": "", "total_time_ms": "", "objective_score": "",
        "judge_score": "", "quality_score": "", "composite_score": "",
        "is_best": "", "notes": "sweep_complete",
    })

    return best_params


def validate_config(config: dict) -> None:
    """Validate config.yaml has all required keys. Exits with a clear message on failure."""
    errors = []

    # infra
    for key in ("ollama_host", "ollama_port", "compose_dir", "compose_project", "ollama_volume"):
        if key not in config.get("infra", {}):
            errors.append(f"  missing: infra.{key}")

    if not config.get("models"):
        errors.append("  missing or empty: models")

    # search_space / defaults must match
    search_space = config.get("search_space", {})
    defaults = config.get("defaults", {})
    if not search_space:
        errors.append("  missing or empty: search_space")
    if not defaults:
        errors.append("  missing or empty: defaults")
    for key in search_space:
        if key not in defaults:
            errors.append(f"  search_space key '{key}' has no entry in defaults")
    # defaults may contain fixed params not in search_space (passed through to Ollama unchanged)

    # scoring
    scoring = config.get("scoring", {})
    for key in ("quality_weight", "latency_weight"):
        if key not in scoring:
            errors.append(f"  missing: scoring.{key}")
    if not errors:
        total = sum(scoring.get(k, 0) for k in ("quality_weight", "latency_weight"))
        if abs(total - 1.0) > 0.01:
            print(f"  WARNING: scoring weights sum to {total:.3f}, expected 1.0")

    # eval
    eval_cfg = config.get("eval", {})
    active_types = eval_cfg.get("types", [])
    if not active_types:
        errors.append("  missing or empty: eval.types")
    for t in active_types:
        if t not in ("coding", "tool_call", "chat"):
            errors.append(f"  eval.types contains unknown type: '{t}'")
    type_weights = eval_cfg.get("type_weights", {})
    if len(active_types) > 1:
        for t in active_types:
            if t not in type_weights:
                errors.append(f"  eval.type_weights missing entry for active type: '{t}'")
        if type_weights:
            total = sum(type_weights.get(t, 0) for t in active_types)
            if abs(total - 1.0) > 0.01:
                print(f"  WARNING: eval.type_weights sum to {total:.3f}, expected 1.0")

    # judge quality_weights
    quality_weights = config.get("judge", {}).get("quality_weights", {})
    if "coding" in active_types:
        for key in ("coding_correctness", "coding_completeness", "coding_clarity", "coding_agent_utility"):
            if key not in quality_weights:
                errors.append(f"  missing: judge.quality_weights.{key}")
    if "tool_call" in active_types:
        for key in ("objective_weight", "judge_weight", "tool_arg_correctness", "tool_selection"):
            if key not in quality_weights:
                errors.append(f"  missing: judge.quality_weights.{key}")
        if "objective_weight" in quality_weights and "judge_weight" in quality_weights:
            total = quality_weights["objective_weight"] + quality_weights["judge_weight"]
            if abs(total - 1.0) > 0.01:
                print(f"  WARNING: objective_weight + judge_weight sum to {total:.3f}, expected 1.0")

    if "chat" in active_types:
        for key in ("chat_instruction_following", "chat_content_quality", "chat_professionalism",
                    "chat_conciseness", "chat_context_retention"):
            if key not in quality_weights:
                errors.append(f"  missing: judge.quality_weights.{key}")

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
    eval_prompts = load_eval_prompts(config)
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

    completed, completed_scores = load_completed_experiments(tsv_path)
    api_call_count = [0]

    active_types = config.get("eval", {}).get("types", [])
    print("autotune-ollama starting")
    print(f"  Ollama: {base_url}")
    print(f"  Models: {len(models)}")
    print(f"  Infra configs: {len(infra_configs)}")
    print(f"  Eval types: {', '.join(active_types)}")
    print(f"  Eval prompts: {len(eval_prompts)}")
    print(f"  Completed experiments (resume): {len(completed)}")
    print(f"  API budget: {config['budget']['max_api_calls']}")
    print()

    for infra_config in infra_configs:
        if not switch_infra_config(infra_config, compose_dir, compose_project, ollama_volume):
            print(f"  SKIPPING infra config: {infra_config} (switch failed)")
            continue

        if not wait_for_api(base_url, timeout=90):
            print(f"  SKIPPING infra config: {infra_config} (API not ready after 90s)")
            continue

        skip_models = set(config.get("skip_models", []))

        for model in models:
            if model in skip_models:
                print(f"\n  Skipping {model} (listed in skip_models)")
                continue

            # Check for the sweep-complete sentinel written at end of coordinate_descent.
            # Avoids pulling the model into GPU memory when there's nothing left to do.
            _defaults_check = config["defaults"]
            _param_order_check = list(_defaults_check.keys())
            sentinel_key = _experiment_key(
                infra_config, model, "complete", "none",
                *[_defaults_check[p] for p in _param_order_check]
            )
            # Sentinel key uses default param values as placeholder — match on prefix instead
            sentinel_prefix = f"{infra_config}|{model}|complete|"
            if any(k.startswith(sentinel_prefix) for k in completed):
                print(f"\n  Skipping {model} @ {infra_config} — sweep already complete")
                continue

            print(f"\n{'='*60}")
            print(f"Model: {model} | Infra: {infra_config}")
            print(f"{'='*60}")

            try:
                pull_model(model, base_url)
            except Exception as e:
                print(f"  SKIPPING model {model}: pull failed ({e})")
                continue

            if not check_gpu_fit(model, base_url):
                print(f"  SKIPPING model {model}: does not fit in GPU")
                continue

            warmup(model, base_url)

            best_params = coordinate_descent(
                model=model,
                infra_config=infra_config,
                config=config,
                eval_prompts=eval_prompts,
                base_url=base_url,
                tsv_path=tsv_path,
                details_path=details_path,
                completed=completed,
                completed_scores=completed_scores,
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
