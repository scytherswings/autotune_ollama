"""Score the Opus reference answers through our eval pipeline to establish a ceiling.

For each prompt, the Opus reference answer is treated as if it were a model response
and run through the same scoring path used during the sweep:

  coding:    judge_output(candidate=reference, reference=reference)
             Self-referential by design — confirms judge calibration and that
             sub-score weights produce sensible numbers on a known-good answer.

  tool_call: check_objective_criteria + judge_tool_call (cold, no reference)
             Non-circular — judge evaluates the Opus answer on its own merits,
             establishing the practical quality ceiling for each prompt.

Usage:
    python score_reference_baseline.py
"""

import json
from dotenv import load_dotenv

load_dotenv()

from autotune import load_config, load_eval_prompts, compute_quality
from eval_harness import check_objective_criteria
from judge import judge_output, judge_tool_call


def parse_reference_as_tool_calls(reference: str) -> tuple[list, str]:
    """Convert a stored Opus reference into (tool_calls, response_text)."""
    try:
        parsed = json.loads(reference)
        if isinstance(parsed, dict) and "tool" in parsed:
            tool_name = parsed.get("tool")
            arguments = parsed.get("arguments", {})
            if tool_name:
                return [{"function": {"name": tool_name, "arguments": arguments}}], ""
            else:
                return [], parsed.get("text", parsed.get("note", ""))
    except (json.JSONDecodeError, ValueError):
        pass
    return [], reference


def score_coding(prompts, refs, judge_model, judge_weights) -> list:
    print("=" * 65)
    print("CODING  (Opus reference scored against itself)")
    print("=" * 65)

    results = []
    for p in prompts:
        pid = p["id"]
        reference = refs.get(pid)
        if not reference:
            print(f"\n[{pid}] SKIPPED — no reference")
            continue

        scores = judge_output(
            prompt=p["prompt"],
            candidate=reference,
            reference=reference,
            model=judge_model,
        )
        quality, judge_score = compute_quality(scores, "coding", judge_weights)

        print(f"\n[{pid}] ({p.get('subcategory', '?')})")
        print(f"  prompt:  {p['prompt'][:90]}...")
        print(f"  judge:   correctness={scores.get('correctness',0):.0f}  "
              f"completeness={scores.get('completeness',0):.0f}  "
              f"clarity={scores.get('clarity',0):.0f}  "
              f"agent_utility={scores.get('agent_utility',0):.0f}")
        print(f"  quality: {quality:.2f}/10  — {scores.get('brief_rationale', '')}")

        results.append({"id": pid, "type": "coding", "quality": quality, "judge": judge_score})
    return results


def score_tool_call(prompts, refs, judge_model, judge_weights) -> list:
    print("\n" + "=" * 65)
    print("TOOL_CALL  (Opus reference scored cold — no reference passed to judge)")
    print("=" * 65)

    results = []
    for p in prompts:
        pid = p["id"]
        reference = refs.get(pid)
        if reference is None:
            print(f"\n[{pid}] SKIPPED — no reference")
            continue

        tool_calls, response_text = parse_reference_as_tool_calls(reference)
        objective = check_objective_criteria(p, tool_calls, response_text)

        scores = judge_tool_call(
            p, tool_calls, response_text,
            reference=None,   # cold — avoids circularity
            model=judge_model,
        )
        quality, judge_score = compute_quality(
            scores, "tool_call", judge_weights, objective.objective_score
        )

        expected = p.get("expected_tool") or "(none)"
        called = tool_calls[0]["function"]["name"] if tool_calls else "(none)"
        flags = (
            f"tool {'✓' if objective.correct_tool else '✗'}  "
            f"fields {'✓' if objective.fields_present else '✗'}  "
            f"no-spurious {'✓' if objective.no_spurious_call else '✗'}"
        )

        print(f"\n[{pid}]")
        print(f"  user:      {p['user_message']}")
        print(f"  expected:  {expected}  →  opus: {called}")
        print(f"  objective: {objective.objective_score:.2f}  ({flags})")
        print(f"  judge:     arg_correctness={scores.get('arg_correctness',0):.1f}  "
              f"tool_selection={scores.get('tool_selection',0):.1f}")
        print(f"  quality:   {quality:.2f}/10  — {scores.get('brief_rationale', '')}")

        results.append({
            "id": pid, "type": "tool_call", "quality": quality, "judge": judge_score,
            "objective": objective.objective_score, "correct_tool": objective.correct_tool,
        })
    return results


def print_summary(coding_results, tool_call_results):
    all_results = coding_results + tool_call_results

    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)

    for label, results in [("Coding", coding_results), ("Tool_call", tool_call_results), ("Overall", all_results)]:
        if not results:
            continue
        avg = sum(r["quality"] for r in results) / len(results)
        print(f"\n  {label} ({len(results)} prompts):")
        print(f"    avg ceiling quality: {avg:.2f}/10")
        if label == "Tool_call":
            correct = sum(1 for r in results if r.get("correct_tool", True))
            perfect_obj = sum(1 for r in results if r.get("objective", 1.0) == 1.0)
            print(f"    correct tool:        {correct}/{len(results)}")
            print(f"    perfect objective:   {perfect_obj}/{len(results)}")

    print("\n  Per-prompt quality ceiling (ascending):")
    for r in sorted(all_results, key=lambda x: x["quality"]):
        tag = "C" if r["type"] == "coding" else "T"
        bar = "█" * int(r["quality"])
        print(f"  [{tag}] {r['id']:<35} {r['quality']:.2f}  {bar}")


def main():
    import sys
    config = load_config()
    judge_model = sys.argv[1] if len(sys.argv) > 1 else config["judge"]["model"]
    judge_weights = config["judge"]["quality_weights"]

    with open("references.json") as f:
        refs = json.load(f)

    prompts = load_eval_prompts(config)
    coding_prompts = [p for p in prompts if p.get("category") == "coding"]
    tool_call_prompts = [p for p in prompts if p.get("category") == "tool_call"]

    print(f"Scoring {len(coding_prompts)} coding + {len(tool_call_prompts)} tool_call prompts")
    print(f"Judge model: {judge_model}\n")

    coding_results = score_coding(coding_prompts, refs, judge_model, judge_weights)
    tool_call_results = score_tool_call(tool_call_prompts, refs, judge_model, judge_weights)
    print_summary(coding_results, tool_call_results)


if __name__ == "__main__":
    main()
