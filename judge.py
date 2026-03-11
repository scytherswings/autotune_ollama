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

Score the candidate from 1-10 on:
1. Correctness (does the code work / is the information accurate?)
2. Completeness (does it address the full prompt?)
3. Clarity (well-structured, good explanations?)
4. Usefulness for an agent workflow (could another LLM consume this output and act on it?)

Return JSON only:
{{"correctness": N, "completeness": N, "clarity": N, "agent_utility": N, "overall": N, "brief_rationale": "..."}}"""


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
        Dict with keys: correctness, completeness, clarity, agent_utility, overall, brief_rationale
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
                max_tokens=512,
                messages=[{"role": "user", "content": judge_prompt}],
            )

            # Extract text content
            text = response.content[0].text.strip()

            # Parse JSON — handle potential markdown code blocks
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                text = text.rsplit("```", 1)[0]
            text = text.strip()

            scores = json.loads(text)

            # Validate expected keys
            required = ["correctness", "completeness", "clarity", "agent_utility", "overall"]
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
