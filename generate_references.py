"""One-time script: generate Opus reference answers for all eval prompt types.

Coding prompts: Opus answers freely — stored as text in references.json.
Tool-call prompts: Opus is called natively with the tool schemas.
  - If a tool call is expected: ideal args are stored as JSON string.
  - If no tool call is expected: Opus text response is stored.

Run with: python generate_references.py [prompts_file]
Already-generated references are skipped (safe to re-run after adding prompts).
"""

import json
import os
import shutil
import sys

import anthropic
from dotenv import load_dotenv

load_dotenv()

REFERENCES_FILE = "references.json"


def _write_atomic(path: str, data: dict) -> None:
    """Write JSON atomically — safe against mid-write crashes."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    shutil.move(tmp, path)


def _generate_coding_reference(
    client: anthropic.Anthropic,
    prompt_entry: dict,
    model: str,
) -> str:
    """Free Opus response to the coding prompt."""
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt_entry["prompt"]}],
    )
    return response.content[0].text


def _to_anthropic_tools(ollama_tools: list) -> list:
    """Convert Ollama tool schema format to Anthropic API format.

    Ollama: {"type": "function", "function": {"name": ..., "parameters": ...}}
    Anthropic: {"name": ..., "input_schema": ...}
    """
    result = []
    for t in ollama_tools:
        fn = t.get("function", t)  # handle both wrapped and unwrapped
        result.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return result


def _generate_tool_call_reference(
    client: anthropic.Anthropic,
    prompt_entry: dict,
    tools: list,
    model: str,
) -> str:
    """Call Opus natively with the tool schemas to get an ideal reference.

    For prompts with an expected tool: returns JSON string of ideal {tool, arguments}.
    For prompts with expected_tool=None: returns Opus text response (no tool should be used).
    """
    anthropic_tools = _to_anthropic_tools(tools)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=prompt_entry["system_prompt"],
        tools=anthropic_tools,
        messages=[{"role": "user", "content": prompt_entry["user_message"]}],
    )

    expected_tool = prompt_entry.get("expected_tool")
    tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
    text_blocks = [b for b in response.content if hasattr(b, "text")]
    text = text_blocks[0].text if text_blocks else ""

    if expected_tool is not None:
        if tool_use_blocks:
            block = tool_use_blocks[0]
            return json.dumps({"tool": block.name, "arguments": block.input})
        else:
            # Opus failed to call a tool — record what it said
            return json.dumps({"tool": None, "note": "Opus did not call a tool", "text": text[:500]})
    else:
        # No tool expected — want the text response
        if tool_use_blocks:
            # Opus spuriously called a tool; note it but keep text too
            return json.dumps({
                "note": f"Opus spuriously called {tool_use_blocks[0].name}",
                "text": text,
            })
        return text


def generate_references(
    prompts_file: str = "eval_prompts.json",
    references_file: str = REFERENCES_FILE,
    model: str = "claude-opus-4-20250514",
) -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    with open(prompts_file) as f:
        data = json.load(f)

    refs: dict[str, str] = {}
    if os.path.exists(references_file):
        with open(references_file) as f:
            refs = json.load(f)

    client = anthropic.Anthropic(api_key=api_key)
    tool_sets = data.get("tool_sets", {})

    # Build flat list of (entry, generation_fn) pairs
    work: list[tuple[dict, callable]] = []

    for entry in data.get("coding_prompts", []):
        work.append((entry, lambda e: _generate_coding_reference(client, e, model)))

    for entry in data.get("tool_call_prompts", []):
        ts_key = entry.get("tool_set")
        tools = tool_sets.get(ts_key, []) if ts_key else []
        # Capture tools in closure
        work.append((entry, lambda e, t=tools: _generate_tool_call_reference(client, e, t, model)))

    total = len(work)
    generated = 0
    skipped = 0

    for i, (entry, generate_fn) in enumerate(work, 1):
        prompt_id = entry["id"]
        category = entry.get("category", "?")

        if prompt_id in refs:
            print(f"  [{i}/{total}] {prompt_id} ({category}): already has reference, skipping")
            skipped += 1
            continue

        print(f"  [{i}/{total}] Generating reference for: {prompt_id} ({category})")

        try:
            refs[prompt_id] = generate_fn(entry)
            generated += 1
            print(f"    Done ({len(refs[prompt_id])} chars)")
            _write_atomic(references_file, refs)
        except anthropic.APIError as e:
            print(f"    ERROR: {e}")

    print(f"\nDone. Generated: {generated}, Skipped: {skipped}, Total: {total}")
    print(f"References saved to: {references_file}")


if __name__ == "__main__":
    prompts_file = sys.argv[1] if len(sys.argv) > 1 else "eval_prompts.json"
    generate_references(prompts_file)
