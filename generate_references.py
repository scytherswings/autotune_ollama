"""Generate Opus reference answers using the Anthropic Batch API (50% cheaper).

Coding prompts: Opus answers freely — stored as text in references.json.
Tool-call prompts: Opus is called natively with the tool schemas.
  - If a tool call is expected: ideal args are stored as JSON string.
  - If no tool call is expected: Opus text response is stored.
Chat prompts: Multi-turn prompts require 2 batch rounds (intermediate turns first).

Usage:
  python generate_references.py                        # skip already-generated
  python generate_references.py --force id1 id2 ...   # force-regenerate specific IDs
  python generate_references.py prompts.json           # alternate prompts file
"""

import json
import os
import shutil
import sys
import time

import anthropic
from dotenv import load_dotenv

load_dotenv()

REFERENCES_FILE = "references.json"
POLL_INTERVAL = 5  # seconds between batch status checks


def _write_atomic(path: str, data: dict) -> None:
    """Write JSON atomically — safe against mid-write crashes."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    shutil.move(tmp, path)


def _to_anthropic_tools(ollama_tools: list) -> list:
    """Convert Ollama tool schema format to Anthropic API format."""
    result = []
    for t in ollama_tools:
        fn = t.get("function", t)
        result.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return result


def _submit_and_wait(client: anthropic.Anthropic, requests: list, label: str) -> dict:
    """Submit requests and return {custom_id: message}.

    Falls back to realtime (non-batch) when there's only 1 request — batch API
    adds ~1-2 min of latency overhead that isn't worth it for a single call.
    """
    if len(requests) == 1:
        req = requests[0]
        print(f"\nRealtime (1 request): {label}...")
        message = client.messages.create(**req["params"])
        return {req["custom_id"]: message}

    print(f"\nSubmitting batch: {label} ({len(requests)} requests)...")
    batch = client.messages.batches.create(requests=requests)
    print(f"  Batch ID: {batch.id}")

    while True:
        time.sleep(POLL_INTERVAL)
        batch = client.messages.batches.retrieve(batch.id)
        counts = batch.request_counts
        print(f"  {batch.processing_status} — "
              f"succeeded={counts.succeeded} errored={counts.errored} "
              f"processing={counts.processing}")
        if batch.processing_status == "ended":
            break

    results = {}
    for result in client.messages.batches.results(batch.id):
        if result.result.type == "succeeded":
            results[result.custom_id] = result.result.message
        else:
            print(f"  ERROR [{result.custom_id}]: {result.result.error}")
    return results


def _extract_text(message) -> str:
    blocks = [b for b in message.content if hasattr(b, "text")]
    return blocks[0].text if blocks else ""


def _extract_tool_result(message, expected_tool) -> str:
    tool_blocks = [b for b in message.content if b.type == "tool_use"]
    text = _extract_text(message)

    if expected_tool is not None:
        if tool_blocks:
            block = tool_blocks[0]
            return json.dumps({"tool": block.name, "arguments": block.input})
        return json.dumps({"tool": None, "note": "Opus did not call a tool", "text": text[:500]})
    else:
        if tool_blocks:
            return json.dumps({
                "note": f"Opus spuriously called {tool_blocks[0].name}",
                "text": text,
            })
        return text


def generate_references(
    prompts_file: str = "eval_prompts.json",
    references_file: str = REFERENCES_FILE,
    model: str = "claude-opus-4-20250514",
    force_ids: set | None = None,
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

    def _needs_generation(entry: dict) -> bool:
        if force_ids is not None:
            return entry["id"] in force_ids
        return entry["id"] not in refs

    # Categorize work
    coding_entries = [e for e in data.get("coding_prompts", []) if _needs_generation(e)]
    tool_call_entries = [e for e in data.get("tool_call_prompts", []) if _needs_generation(e)]
    chat_entries = [e for e in data.get("chat_prompts", []) if _needs_generation(e)]
    single_turn_chat = [e for e in chat_entries if len(e.get("turns", [])) <= 1]
    multi_turn_chat = [e for e in chat_entries if len(e.get("turns", [])) > 1]

    skipped = sum(
        1 for section in (data.get("coding_prompts", []), data.get("tool_call_prompts", []), data.get("chat_prompts", []))
        for e in section if not _needs_generation(e)
    )
    total_new = len(coding_entries) + len(tool_call_entries) + len(chat_entries)

    print(f"References to generate: {total_new}  (skipping {skipped} already done)")
    if total_new == 0:
        print("Nothing to do.")
        return

    # --- Round 1: Intermediate turns for multi-turn chat ---
    # For a 2-turn prompt we need the assistant's Turn 1 response to build the
    # Turn 2 context. All current prompts are ≤2 turns so one intermediate batch
    # is sufficient; deeper chains would need additional rounds.
    intermediate_text: dict[str, str] = {}
    if multi_turn_chat:
        round1_requests = [
            {
                "custom_id": f"__intermediate__{entry['id']}",
                "params": {
                    "model": model,
                    "max_tokens": 2048,
                    "system": entry.get("system_prompt", ""),
                    "messages": [{"role": "user", "content": entry["turns"][0]["content"]}],
                },
            }
            for entry in multi_turn_chat
        ]
        r1_results = _submit_and_wait(client, round1_requests, "Round 1: multi-turn chat intermediates")
        for entry in multi_turn_chat:
            cid = f"__intermediate__{entry['id']}"
            if cid in r1_results:
                intermediate_text[entry["id"]] = _extract_text(r1_results[cid])

    # --- Round 2: All final requests ---
    round2_requests = []

    for entry in coding_entries:
        round2_requests.append({
            "custom_id": entry["id"],
            "params": {
                "model": model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": entry["prompt"]}],
            },
        })

    for entry in tool_call_entries:
        ts_key = entry.get("tool_set")
        tools = tool_sets.get(ts_key, []) if ts_key else []
        round2_requests.append({
            "custom_id": entry["id"],
            "params": {
                "model": model,
                "max_tokens": 1024,
                "system": entry["system_prompt"],
                "tools": _to_anthropic_tools(tools),
                "messages": [{"role": "user", "content": entry["user_message"]}],
            },
        })

    for entry in single_turn_chat:
        turns = entry.get("turns", [])
        round2_requests.append({
            "custom_id": entry["id"],
            "params": {
                "model": model,
                "max_tokens": 2048,
                "system": entry.get("system_prompt", ""),
                "messages": [{"role": "user", "content": turns[0]["content"]}] if turns else [],
            },
        })

    for entry in multi_turn_chat:
        turns = entry["turns"]
        messages = [
            {"role": "user", "content": turns[0]["content"]},
            {"role": "assistant", "content": intermediate_text.get(entry["id"], "")},
            {"role": "user", "content": turns[-1]["content"]},
        ]
        round2_requests.append({
            "custom_id": entry["id"],
            "params": {
                "model": model,
                "max_tokens": 2048,
                "system": entry.get("system_prompt", ""),
                "messages": messages,
            },
        })

    r2_results = _submit_and_wait(client, round2_requests, "Round 2: all final responses")

    # Extract and save
    for entry in coding_entries + single_turn_chat + multi_turn_chat:
        pid = entry["id"]
        if pid in r2_results:
            refs[pid] = _extract_text(r2_results[pid])
            print(f"  {pid}: {len(refs[pid])} chars")

    for entry in tool_call_entries:
        pid = entry["id"]
        if pid in r2_results:
            refs[pid] = _extract_tool_result(r2_results[pid], entry.get("expected_tool"))
            print(f"  {pid}: {refs[pid][:80]}...")

    _write_atomic(references_file, refs)
    print(f"\nDone. Generated {total_new} references. Saved to {references_file}")


if __name__ == "__main__":
    args = sys.argv[1:]
    force_ids = None
    prompts_file = "eval_prompts.json"

    if "--force" in args:
        idx = args.index("--force")
        force_ids = set(args[idx + 1:]) or None  # empty set → None (regenerate all)
        args = args[:idx]

    if args:
        prompts_file = args[0]

    generate_references(prompts_file, force_ids=force_ids)
