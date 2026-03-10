"""One-time script: generate Opus reference answers for eval prompts."""

import json
import os
import sys

import anthropic


def generate_references(
    prompts_file: str = "eval_prompts.json",
    model: str = "claude-opus-4-20250514",
) -> None:
    """Read eval prompts, generate reference answers via Claude Opus, write back."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    with open(prompts_file) as f:
        data = json.load(f)

    client = anthropic.Anthropic(api_key=api_key)
    all_prompts = data.get("coding_prompts", []) + data.get("chat_prompts", [])

    total = len(all_prompts)
    generated = 0
    skipped = 0

    for prompt_entry in all_prompts:
        prompt_id = prompt_entry["id"]
        prompt_text = prompt_entry["prompt"]

        if prompt_entry.get("reference"):
            print(f"  [{skipped + generated + 1}/{total}] {prompt_id}: already has reference, skipping")
            skipped += 1
            continue

        print(f"  [{skipped + generated + 1}/{total}] Generating reference for: {prompt_id}")

        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt_text}],
            )
            reference_text = response.content[0].text
            prompt_entry["reference"] = reference_text
            generated += 1
            print(f"    Done ({len(reference_text)} chars)")

        except anthropic.APIError as e:
            print(f"    ERROR generating reference for {prompt_id}: {e}")
            prompt_entry["reference"] = None

    # Write back
    with open(prompts_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nDone. Generated: {generated}, Skipped: {skipped}, Total: {total}")


if __name__ == "__main__":
    prompts_file = sys.argv[1] if len(sys.argv) > 1 else "eval_prompts.json"
    generate_references(prompts_file)
