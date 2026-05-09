"""LLM-as-judge utilities for the eval suite.

Each judge defines a single rubric, calls Gemini Flash through litellm,
and parses the JSON response. Judges are deliberately tiny — one rubric
per function — so failures point at one diagnosable property of the
summary (faithfulness vs. coverage vs. length, etc.).
"""

import json
import os
import re
from dataclasses import dataclass


@dataclass
class FaithfulnessVerdict:
    score: int                       # 0 / 1 / 2
    rationale: str
    unsupported_claims: list[str]


_FAITHFULNESS_RUBRIC = """You are evaluating whether a book summary is FAITHFUL to its source notes.

NOTES (the only allowed source of facts):
---
{notes}
---

SUMMARY (to evaluate):
---
{summary}
---

Score using this rubric:
  0 = The summary contains at least one factual claim NOT supported by the notes.
      Examples: invented people, made-up dates, statistics not present in the source.
  1 = All claims in the summary are supported by the notes, but the summary misses
      key themes or substantial portions of the source content.
  2 = All claims are supported by the notes AND the summary captures the main themes.

Output ONLY a JSON object on a single line, no markdown, no preamble:
{{"score": <0|1|2>, "rationale": "<one sentence>", "unsupported_claims": ["<claim>", ...]}}

If score is 1 or 2, ``unsupported_claims`` MUST be the empty list."""


def _strip_code_fences(text: str) -> str:
    """Models occasionally wrap JSON in ```json ... ``` despite the prompt."""
    text = text.strip()
    if text.startswith("```"):
        # Drop the opening fence (with optional language) and the closing fence.
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


def faithfulness_judge(
    *,
    notes: str,
    summary: str,
    model: str = "gemini/gemini-2.5-flash",
    api_key_env: str = "GOOGLE_API_KEY",
) -> FaithfulnessVerdict:
    """Score a summary's faithfulness against its source notes.

    Raises ``ValueError`` if the judge returns malformed JSON — better to
    fail loudly than silently treat a parse error as a passing score.
    """
    from litellm import completion

    prompt = _FAITHFULNESS_RUBRIC.format(notes=notes, summary=summary)
    # max_tokens=8192 because Gemini 2.5 Flash spends invisible "thinking"
    # tokens against this budget. The visible JSON is tiny (~200 chars)
    # but a stricter budget truncates mid-string. 8192 matches the project's
    # note-taker setting for the same reason.
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=8192,
        api_key=os.getenv(api_key_env),
    )
    raw = response.choices[0].message.content or ""
    body = _strip_code_fences(raw)

    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Faithfulness judge returned non-JSON output:\n{raw!r}"
        ) from e

    if "score" not in parsed:
        raise ValueError(f"Faithfulness judge response missing 'score': {parsed}")

    return FaithfulnessVerdict(
        score=int(parsed["score"]),
        rationale=str(parsed.get("rationale", "")),
        unsupported_claims=list(parsed.get("unsupported_claims", [])),
    )
