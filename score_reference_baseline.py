"""Score the Opus reference answers through our eval pipeline to establish a ceiling.

For each tool_call prompt, the Opus reference answer is treated as if it were a model
response and run through the same objective checks + Claude judge used during the sweep.
The judge evaluates cold (no reference passed in) to avoid circularity.

This answers: "what score does a perfect answer get on our scale?" — calibrating the
judge and confirming scoring weights make sense before spending GPU time on the sweep.

Usage:
    python score_reference_baseline.py
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from autotune import load_config, load_eval_prompts, compute_quality
from eval_harness import check_objective_criteria
from judge import judge_tool_call


def parse_reference_as_tool_calls(reference: str) -> tuple[list, str]:
    """Convert a stored Opus reference into (tool_calls, response_text).

    Tool references are stored as JSON: {"tool": name, "arguments": {...}}
    No-tool references are stored as plain text.
    """
    try:
        parsed = json.loads(reference)
        if isinstance(parsed, dict) and "tool" in parsed:
            tool_name = parsed.get("tool")
            arguments = parsed.get("arguments", {})
            if tool_name:
                tool_calls = [{"function": {"name": tool_name, "arguments": arguments}}]
                return tool_calls, ""
            else:
                # Opus flagged it shouldn't call a tool (e.g. note field)
                text = parsed.get("text", parsed.get("note", ""))
                return [], text
    except (json.JSONDecodeError, ValueError):
        pass
    # Plain text — no-tool response
    return [], reference


def main():
    config = load_config()
    judge_model = config["judge"]["model"]
    judge_weights = config["judge"]["quality_weights"]

    with open("references.json") as f:
        refs = json.load(f)

    # Load prompts to get tool schemas and metadata
    prompts = load_eval_prompts(config)
    tool_call_prompts = [p for p in prompts if p.get("category") == "tool_call"]

    print("=" * 65)
    print("Opus reference baseline — tool_call prompts")
    print(f"Judge model: {judge_model}")
    print("Note: judge runs cold (no reference passed) to avoid circularity")
    print("=" * 65)

    quality_scores = []
    objective_scores = []
    results = []

    for prompt_entry in tool_call_prompts:
        prompt_id = prompt_entry["id"]
        reference = refs.get(prompt_id)

        if reference is None:
            print(f"\n[{prompt_id}] SKIPPED — no reference in references.json")
            continue

        tool_calls, response_text = parse_reference_as_tool_calls(reference)

        # Objective checks
        objective = check_objective_criteria(prompt_entry, tool_calls, response_text)

        # Judge — cold evaluation, no reference passed in
        scores = judge_tool_call(
            prompt_entry, tool_calls, response_text,
            reference=None,
            model=judge_model,
        )

        quality, judge_score = compute_quality(
            scores, "tool_call", judge_weights, objective.objective_score
        )

        expected = prompt_entry.get("expected_tool", "(none)")
        called = tool_calls[0]["function"]["name"] if tool_calls else "(none)"
        correct_tool = "✓" if objective.correct_tool else "✗"
        fields_ok = "✓" if objective.fields_present else "✗"
        no_spurious = "✓" if objective.no_spurious_call else "✗"

        print(f"\n[{prompt_id}]")
        print(f"  user:      {prompt_entry['user_message']}")
        print(f"  expected:  {expected}  →  opus called: {called}")
        print(f"  objective: {objective.objective_score:.2f}  "
              f"(tool {correct_tool}  fields {fields_ok}  no-spurious {no_spurious})")
        print(f"  judge:     arg_correctness={scores.get('arg_correctness', '?'):.1f}  "
              f"tool_selection={scores.get('tool_selection', '?'):.1f}")
        print(f"  quality:   {quality:.2f}/10  — {scores.get('brief_rationale', '')}")

        quality_scores.append(quality)
        objective_scores.append(objective.objective_score)
        results.append({
            "id": prompt_id,
            "quality": quality,
            "objective": objective.objective_score,
            "judge": judge_score,
            "correct_tool": objective.correct_tool,
        })

    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)
    print(f"  Prompts scored:       {len(quality_scores)}")
    print(f"  Avg quality (ceiling): {sum(quality_scores)/len(quality_scores):.2f}/10")
    print(f"  Avg objective:         {sum(objective_scores)/len(objective_scores):.2f}")
    print(f"  Perfect objective (1.0): {sum(1 for r in results if r['objective'] == 1.0)}/{len(results)}")
    print(f"  Correct tool chosen:     {sum(1 for r in results if r['correct_tool'])}/{len(results)}")
    print()
    print("Per-prompt quality ceiling:")
    for r in sorted(results, key=lambda x: x["quality"]):
        bar = "█" * int(r["quality"])
        print(f"  {r['id']:<35} {r['quality']:.2f}  {bar}")


if __name__ == "__main__":
    main()
