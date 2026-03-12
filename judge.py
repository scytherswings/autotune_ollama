"""Quality judging via Claude API — coding and tool-calling variants."""

import json
import os
import time

import anthropic

# Lazy singleton — instantiated once on first judge call
_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _call_judge(judge_prompt: str, model: str, required_keys: list, max_retries: int = 5) -> dict:
    """Shared retry/parse logic for all judge variants."""
    client = _get_client()
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=512,
                messages=[
                    {"role": "user", "content": judge_prompt},
                    {"role": "assistant", "content": "{"},
                ],
            )
            text = "{" + response.content[0].text.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in judge response")
            scores = json.loads(text[start:end])
            for key in required_keys:
                if key not in scores:
                    raise ValueError(f"Missing key in judge response: {key}")
                if key != "brief_rationale":
                    scores[key] = float(scores[key])
            return scores

        except anthropic.RateLimitError:
            wait = min(2 ** attempt * 5, 120)
            print(f"  Rate limited, waiting {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"  Judge parse error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
        except anthropic.APIStatusError as e:
            if e.status_code in (400, 401, 403):
                raise RuntimeError(f"Fatal API error (not retriable): {e}") from e
            wait = min(2 ** attempt * 5, 120)
            print(f"  API error: {e}, waiting {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
        except anthropic.APIError as e:
            wait = min(2 ** attempt * 5, 120)
            print(f"  API error: {e}, waiting {wait}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)

    raise RuntimeError(f"Judge failed after {max_retries} retries — results not recorded")

JUDGE_TOOL_CALL_TEMPLATE = """You are evaluating the quality of a tool call produced by a language model acting as a scheduling assistant.

Context provided to the model:
<system_prompt>{system_prompt}</system_prompt>

User request:
<user_message>{user_message}</user_message>

Available tools:
<tool_schemas>{tool_schemas}</tool_schemas>
{reference_section}
Model response:
<model_response>{model_response}</model_response>

Note: Objective checks (JSON validity, required fields) are handled separately. Focus on SEMANTIC quality.
The model had access to the system prompt above — evaluate date/time values against that context.

Score 1–10 on each criterion. Use the full range.

1. arg_correctness: Are the argument values semantically right given the user's request?
   - Names extracted correctly, dates/times parsed accurately, service types mapped appropriately
   - If no tool was called: does that make sense given the request?
   - 1 = values are wrong or fabricated; 10 = all values precisely match the user's intent

2. tool_selection: Was the right tool chosen for this request?
   - If the user asked a general question, the right answer is no tool call
   - If the user made an incomplete request, the right answer is no tool call (ask for info instead)
   - 1 = completely wrong tool or spurious call; 10 = exactly right choice

Return JSON only:
{{"arg_correctness": N, "tool_selection": N, "brief_rationale": "one sentence on the most important issue"}}"""


def judge_tool_call(
    prompt_entry: dict,
    tool_calls: list,
    response_text: str,
    reference: str | None = None,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Judge the semantic quality of a model's tool call response.

    Args:
        reference: Pre-generated Opus ideal response (JSON string of ideal args, or ideal text).
                   When provided, the judge compares against it rather than reasoning cold.

    Returns dict with keys: arg_correctness, tool_selection, brief_rationale
    """
    tools = prompt_entry.get("tools", [])
    # Include full parameter schema (required + optional) so judge doesn't penalise valid optional args
    tool_schema_text = json.dumps(
        [{"name": t["function"]["name"],
          "description": t["function"]["description"],
          "parameters": t["function"]["parameters"]}
         for t in tools],
        indent=2,
    )

    if tool_calls:
        model_response_text = json.dumps(
            [{"tool": tc.get("function", {}).get("name"),
              "arguments": tc.get("function", {}).get("arguments", {})}
             for tc in tool_calls],
            indent=2,
        )
    else:
        model_response_text = f"(No tool called)\nText response: {response_text[:500]}" if response_text else "(No tool called, no text response)"

    reference_section = (
        f"\nReference (ideal response from a top-tier model):\n<reference>{reference}</reference>\n"
        if reference else ""
    )

    judge_prompt = JUDGE_TOOL_CALL_TEMPLATE.format(
        system_prompt=prompt_entry.get("system_prompt", ""),
        user_message=prompt_entry["user_message"],
        tool_schemas=tool_schema_text,
        reference_section=reference_section,
        model_response=model_response_text,
    )
    return _call_judge(judge_prompt, model, ["arg_correctness", "tool_selection", "brief_rationale"])


JUDGE_PROMPT_TEMPLATE = """You are evaluating the quality of an LLM response for use in agentic coding workflows.

Reference answer (high quality):
<reference>{reference}</reference>

Candidate answer (from local model):
<candidate>{candidate}</candidate>

Original prompt:
<prompt>{prompt}</prompt>

Score the candidate 1-10 on each of these four criteria. Use the full range — reserve 10 for exceptional responses and 1 for completely wrong or useless ones.

1. correctness: Does the code/answer actually solve the problem correctly?
   - Code must run without errors and produce correct output for the stated problem
   - Logic must be sound with no bugs or incorrect assumptions
   - Factual claims must be accurate
   - 1 = fundamentally broken or wrong; 10 = fully correct and handles the problem well

2. completeness: Does the response address the entire prompt?
   - All requirements and sub-tasks are covered
   - Important edge cases are handled
   - Nothing critical is missing that would require follow-up
   - 1 = barely started or missing most requirements; 10 = fully addressed

3. clarity: Is the response well-structured and understandable?
   - Code is readable with appropriate naming, structure, and comments
   - Explanations are clear and well-organized
   - 1 = confusing or unreadable; 10 = exemplary clarity

4. agent_utility: Can another LLM directly consume and act on this output without clarification?
   - Code blocks are properly fenced and complete
   - Output is unambiguous and self-contained
   - 1 = requires major interpretation or cleanup; 10 = immediately usable as-is

Return JSON only:
{{"correctness": N, "completeness": N, "clarity": N, "agent_utility": N, "brief_rationale": "one sentence explaining the correctness score"}}"""


JUDGE_CHAT_TEMPLATE = """You are evaluating the quality of a business chat response from a language model.

The model was given this system prompt:
<system_prompt>{system_prompt}</system_prompt>

The conversation so far:
<conversation>
{conversation_history}
</conversation>

The model's final response:
<response>{response}</response>
{reference_section}
Score 1–10 on each criterion. Use the full range — reserve 10 for exceptional and 1 for useless.

1. instruction_following: Did the model do exactly what was asked?
   - Correct format (agenda, email, bullet list, etc.), right length, addressed the actual request
   - 1 = ignored instructions; 10 = followed every instruction precisely

2. content_quality: Is the substance genuinely useful?
   - Well-reasoned, specific, grounded in the context provided — not generic filler
   - Decisions are logical, recommendations are actionable, structure serves the purpose
   - 1 = vague or useless; 10 = would use this as-is in a real business setting

3. professionalism: Is the tone, structure, and language appropriate for business use?
   - No unnecessary hedging, no fluff, no overly casual or overly formal language
   - 1 = unprofessional or jarring; 10 = polished and appropriate

4. conciseness: Is the response the right length — no padding, no unnecessary caveats?
   - Gets to the point without being terse; doesn't repeat itself
   - 1 = bloated or padded; 10 = exactly as long as it needs to be

5. context_retention: Does the response accurately use specific details from earlier in the conversation?
   - Names, numbers, constraints, and decisions from prior turns are referenced correctly
   - For a first turn with no prior context, score this 10
   - 1 = ignores or contradicts earlier context; 10 = accurately integrates all relevant prior details

Return JSON only:
{{"instruction_following": N, "content_quality": N, "professionalism": N, "conciseness": N, "context_retention": N, "brief_rationale": "one sentence on the most important strength or weakness"}}"""


def judge_chat(
    prompt_entry: dict,
    conversation: list[dict],
    response_text: str,
    reference: str | None = None,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Judge the quality of a business chat response.

    Args:
        conversation: Full message history including the final user turn
                      (list of {role, content} dicts, without the model's final response)
        response_text: The model's final response to judge
        reference: Pre-generated Opus reference answer for this prompt

    Returns dict with keys: instruction_following, content_quality, professionalism,
                            conciseness, context_retention, brief_rationale
    """
    history_text = "\n\n".join(
        f"[{m['role'].upper()}]: {m['content']}"
        for m in conversation
        if m["role"] != "system"
    )
    reference_section = (
        f"\nReference (ideal response from a top-tier model):\n<reference>{reference}</reference>\n"
        if reference else ""
    )
    judge_prompt = JUDGE_CHAT_TEMPLATE.format(
        system_prompt=prompt_entry.get("system_prompt", ""),
        conversation_history=history_text,
        response=response_text,
        reference_section=reference_section,
    )
    return _call_judge(
        judge_prompt, model,
        ["instruction_following", "content_quality", "professionalism",
         "conciseness", "context_retention", "brief_rationale"],
    )


def judge_output(
    prompt: str,
    candidate: str,
    reference: str,
    model: str = "claude-sonnet-4-20250514",
    max_retries: int = 5,
) -> dict:
    """Judge coding response quality against a reference answer.

    Returns dict with keys: correctness, completeness, clarity, agent_utility, brief_rationale
    """
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        reference=reference,
        candidate=candidate,
        prompt=prompt,
    )
    return _call_judge(
        judge_prompt, model,
        ["correctness", "completeness", "clarity", "agent_utility", "brief_rationale"],
        max_retries=max_retries,
    )
