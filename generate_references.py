"""One-time script: generate Opus reference answers for eval prompts."""

import json
import os
import shutil
import sys

import anthropic
from dotenv import load_dotenv

load_dotenv()  # Load .env before anything touches os.environ

REFERENCES_FILE = "references.json"


def _write_atomic(path: str, data: dict) -> None:
    """Write JSON to path atomically via a temp file (safe against mid-write crashes)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    shutil.move(tmp, path)


def generate_references(
    prompts_file: str = "eval_prompts.json",
    references_file: str = REFERENCES_FILE,
    model: str = "claude-opus-4-20250514",
) -> None:
    """Read eval prompts, generate reference answers via Claude Opus.

    References are written to references_file (default: references.json),
    leaving eval_prompts.json unchanged and clean for version control.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    with open(prompts_file) as f:
        data = json.load(f)

    # Load existing references so we can skip already-generated ones
    refs: dict[str, str] = {}
    if os.path.exists(references_file):
        with open(references_file) as f:
            refs = json.load(f)

    client = anthropic.Anthropic(api_key=api_key)
    all_prompts = data.get("coding_prompts", []) + data.get("chat_prompts", [])

    total = len(all_prompts)
    generated = 0
    skipped = 0

    for prompt_entry in all_prompts:
        prompt_id = prompt_entry["id"]
        prompt_text = prompt_entry["prompt"]

        if prompt_id in refs:
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
            refs[prompt_id] = response.content[0].text
            generated += 1
            print(f"    Done ({len(refs[prompt_id])} chars)")

            # Write atomically after each success so a crash loses at most one entry
            _write_atomic(references_file, refs)

        except anthropic.APIError as e:
            print(f"    ERROR generating reference for {prompt_id}: {e}")

    print(f"\nDone. Generated: {generated}, Skipped: {skipped}, Total: {total}")
    print(f"References saved to: {references_file}")


if __name__ == "__main__":
    prompts_file = sys.argv[1] if len(sys.argv) > 1 else "eval_prompts.json"
    generate_references(prompts_file)
