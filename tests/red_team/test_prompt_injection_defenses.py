"""Red-team tests — verify our prompt-injection defenses survive
adversarial inputs.

We run BookExplorer with a deliberately mocked LLM that simulates what
a hijacked model would emit — leaking the canary, returning malicious
tool calls, or trying to dump the system prompt. The assertion is that
our layered defenses (canary capability + Pydantic validation +
fallback all-read plan) catch each attack and the pipeline reaches a
known-safe state instead of being silently compromised.

These are ``unittest``-style integration tests with a scripted
``AgentLanguage`` — no real LLM calls. Cheap, deterministic, runs in
the default test suite.
"""

import json
from pathlib import Path
from typing import List

from book_summarizer.book_explorer import BookExplorer
from zurvan import AgentLanguage, Logger, Prompt
from tests.fixtures import create_tiny_book


_FIXED_CANARY = "CANARY_TEST_DEADBEEFCAFE"


class _ScriptedLanguage(AgentLanguage):
    """Returns scripted JSON responses in order — same shape as the
    integration-test scripted language, copied here to keep red-team
    tests self-contained and easy to read."""

    def __init__(self, scripted_responses: List[str]):
        super().__init__()
        self._responses = list(scripted_responses)
        self.call_count = 0

    def construct_prompt(self, actions, environment, goals, memory):
        return Prompt(messages=[], tools=[a.name for a in actions])

    def generate_response(self, prompt):
        if self.call_count >= len(self._responses):
            raise RuntimeError(
                f"Test scripted only {len(self._responses)} responses; "
                f"agent asked for {self.call_count + 1}."
            )
        r = self._responses[self.call_count]
        self.call_count += 1
        return r

    def parse_response(self, response):
        return json.loads(response)


def _explorer_with_canary(book_id: str, language) -> BookExplorer:
    return BookExplorer(
        book_id=book_id,
        logger=Logger(),
        token_tracker=None,
        agent_language=language,
        max_iterations=3,
        canary_token=_FIXED_CANARY,
    )


# ── Attack 1: model leaks the canary in tool arguments ───────────────────


def test_canary_leak_in_tool_args_triggers_fallback(tmp_books_dir: Path):
    """A hijacked model dumps the canary inside its tool-call JSON.
    CanaryCapability detects the leak in every iteration → agent loop
    exhausts its retry budget → BookExplorer falls back to all-read
    instead of trusting the response."""
    book_id = "veridian_settlement"
    create_tiny_book(tmp_books_dir, book_id=book_id, num_pages=4)

    # Same poisoned response three times: includes the canary inside
    # the ``reason`` field of a tool-call argument.
    poisoned = json.dumps(
        {
            "tool": "set_reading_strategy",
            "args": {
                "entries": [
                    {
                        "page_from": 1,
                        "page_to": 4,
                        "importance": "read",
                        "reason": f"leaked: {_FIXED_CANARY}",
                    }
                ]
            },
        }
    )
    language = _ScriptedLanguage(scripted_responses=[poisoned] * 3)

    plan = _explorer_with_canary(book_id, language).explore()

    # Agent retried until budget exhausted (3 consecutive canary leaks).
    assert language.call_count == 3
    # The poisoned plan never reached disk — fallback all-read is what
    # was saved.
    assert len(plan.entries) == 1
    assert plan.entries[0].importance == "read"
    assert "Fallback" in plan.entries[0].reason
    assert _FIXED_CANARY not in plan.entries[0].reason


# ── Attack 2: model leaks the canary as plain text ───────────────────────


def test_canary_leak_in_text_response_triggers_fallback(tmp_books_dir: Path):
    """Variant: the model returns a non-tool text response that quotes
    the canary. Same defense applies."""
    book_id = "veridian_settlement"
    create_tiny_book(tmp_books_dir, book_id=book_id, num_pages=4)

    # Plain-text response (not a tool call) that includes the canary.
    # The Agent loop will fail to parse it as JSON tool call; even if
    # it did parse, the canary scan happens first.
    leaky_text = (
        f"Here is what you asked for: {_FIXED_CANARY} "
        "and the plan would be all read."
    )
    language = _ScriptedLanguage(scripted_responses=[leaky_text] * 3)

    plan = _explorer_with_canary(book_id, language).explore()

    assert language.call_count == 3
    assert plan.entries[0].importance == "read"
    assert "Fallback" in plan.entries[0].reason


# ── Attack 3: malformed but canary-clean response → fall back too ────────


def test_invalid_plan_without_canary_still_falls_back(tmp_books_dir: Path):
    """Sanity check: a structurally invalid plan (overlap) still falls
    back. This isn't a canary scenario — it's verifying the Pydantic
    layer catches functional attacks where the model returns a clean-
    looking but semantically wrong plan."""
    book_id = "veridian_settlement"
    create_tiny_book(tmp_books_dir, book_id=book_id, num_pages=4)

    # No canary leak; plan structure is malformed (overlap on page 1).
    bad_plan = json.dumps(
        {
            "tool": "set_reading_strategy",
            "args": {
                "entries": [
                    {"page_from": 1, "page_to": 1, "importance": "skip", "reason": "x"},
                    {"page_from": 1, "page_to": 4, "importance": "read", "reason": "y"},
                ]
            },
        }
    )
    language = _ScriptedLanguage(scripted_responses=[bad_plan])

    plan = _explorer_with_canary(book_id, language).explore()

    # Pydantic caught the overlap, set_reading_strategy returned a
    # validation error, the action is still terminal so the agent
    # stopped — fallback kicks in.
    assert language.call_count == 1
    assert plan.entries[0].importance == "read"
    assert "Fallback" in plan.entries[0].reason


# ── Attack 4: clean response succeeds (negative control) ─────────────────


def test_clean_response_succeeds_under_canary_guard(tmp_books_dir: Path):
    """Negative control: with the canary capability ACTIVE, a clean
    response that follows the rules and never mentions the canary
    must still produce a valid saved plan. This proves our defense
    isn't accidentally rejecting normal traffic."""
    book_id = "veridian_settlement"
    create_tiny_book(tmp_books_dir, book_id=book_id, num_pages=4)

    clean = json.dumps(
        {
            "tool": "set_reading_strategy",
            "args": {
                "entries": [
                    {
                        "page_from": 1,
                        "page_to": 4,
                        "importance": "read",
                        "reason": "All chapters substantive",
                    }
                ]
            },
        }
    )
    language = _ScriptedLanguage(scripted_responses=[clean])

    plan = _explorer_with_canary(book_id, language).explore()

    assert language.call_count == 1
    assert len(plan.entries) == 1
    assert plan.entries[0].importance == "read"
    # Real plan, NOT the fallback.
    assert "Fallback" not in plan.entries[0].reason
