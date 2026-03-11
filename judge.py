"""Quality judging via Claude Sonnet API."""

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


def judge_output(
    prompt: str,
    candidate: str,
    reference: str,
    model: str = "claude-sonnet-4-20250514",
    max_retries: int = 5,
) -> dict:
    """Send candidate output to Claude for quality scoring.

    Args:
        prompt: The original eval prompt
        candidate: The local model's output
        reference: The Opus reference answer
        model: Claude model to use for judging
        max_retries: Max retries with exponential backoff

    Returns:
        Dict with keys: correctness, completeness, clarity, agent_utility, brief_rationale
    """
    client = _get_client()

    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
        reference=reference,
        candidate=candidate,
        prompt=prompt,
    )

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": judge_prompt},
                    {"role": "assistant", "content": "{"},
                ],
            )

            # Prepend the prefilled "{" and extract the first complete JSON object
            text = "{" + response.content[0].text.strip()
            start = text.find("{")
            end = text.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found in judge response")
            text = text[start:end]

            scores = json.loads(text)

            # Validate expected keys
            required = ["correctness", "completeness", "clarity", "agent_utility"]
            for key in required:
                if key not in scores:
                    raise ValueError(f"Missing key in judge response: {key}")
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
            # Fatal errors (billing, auth, permissions) should not be retried
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
