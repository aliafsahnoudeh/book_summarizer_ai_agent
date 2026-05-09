"""Integration tests for BookExplorer — agent end-to-end with a scripted LLM.

We never call a real LLM here. Instead we inject a scripted
``AgentLanguage`` that returns canned tool-call JSONs in order. This
exercises the GAME loop, the tool registry, the Pydantic validation, and
the disk persistence — everything except the model itself.
"""

import json
from pathlib import Path
from typing import List

from book_summarizer.book_explorer import BookExplorer, ReadingPlan
from zurvan import AgentLanguage, Logger, Prompt
from tests.fixtures import create_tiny_book


class _ScriptedLanguage(AgentLanguage):
    """AgentLanguage that returns scripted JSON tool calls, in order."""

    def __init__(self, scripted_responses: List[str]):
        super().__init__()
        self._responses = list(scripted_responses)
        self.call_count = 0
        # Capture every prompt for diagnostics (lets a test assert what
        # tools the agent saw, what memory shape, etc.).
        self.prompts_seen: list[Prompt] = []

    def construct_prompt(self, actions, environment, goals, memory):
        prompt = Prompt(messages=[], tools=[a.name for a in actions])
        self.prompts_seen.append(prompt)
        return prompt

    def generate_response(self, prompt):
        if self.call_count >= len(self._responses):
            raise RuntimeError(
                f"Agent made more LLM calls ({self.call_count + 1}) than the "
                f"test scripted ({len(self._responses)})."
            )
        r = self._responses[self.call_count]
        self.call_count += 1
        return r

    def parse_response(self, response):
        return json.loads(response)


def _explorer(book_id: str, language: _ScriptedLanguage) -> BookExplorer:
    return BookExplorer(
        book_id=book_id,
        logger=Logger(),
        token_tracker=None,
        agent_language=language,
        max_iterations=10,
    )


# ── Happy path ────────────────────────────────────────────────────────────


def test_agent_calls_set_reading_strategy_in_one_iteration(tmp_books_dir: Path):
    """The agent has metadata + TOC pre-loaded into the goal description,
    so its only job is to call ``set_reading_strategy`` once. One LLM
    call per uncached book — no agent overhead."""
    book_id = "veridian_settlement"
    create_tiny_book(tmp_books_dir, book_id=book_id, num_pages=4)

    language = _ScriptedLanguage(
        scripted_responses=[
            json.dumps(
                {
                    "tool": "set_reading_strategy",
                    "args": {
                        "entries": [
                            {
                                "page_from": 1,
                                "page_to": 1,
                                "importance": "skim",
                                "reason": "Frontmatter / chapter 1 intro",
                            },
                            {
                                "page_from": 2,
                                "page_to": 4,
                                "importance": "read",
                                "reason": "Substantive chapters",
                            },
                        ]
                    },
                }
            ),
        ]
    )

    plan = _explorer(book_id, language).explore()

    # Single LLM call total — no metadata/TOC fetch round-trips.
    assert language.call_count == 1

    # Plan structure matches what we scripted.
    assert plan.book_id == book_id
    assert plan.total_pages == 4
    assert len(plan.entries) == 2
    assert plan.entries[0].importance == "skim"
    assert plan.entries[1].importance == "read"

    # Plan was persisted to disk in the expected location.
    plan_path = tmp_books_dir / book_id / "reading_plan.json"
    assert plan_path.exists()
    saved = json.loads(plan_path.read_text())
    assert saved["total_pages"] == 4


def test_goal_description_includes_metadata_and_toc(tmp_books_dir: Path):
    """The whole point of the refactor: the LLM sees the metadata and TOC
    in its system prompt, not via tool calls. Verify they're there."""
    from book_summarizer.book_explorer import _GOAL_TEMPLATE, _format_toc, BookExplorer

    book_id = "veridian_settlement"
    create_tiny_book(tmp_books_dir, book_id=book_id, num_pages=4)

    explorer = BookExplorer(
        book_id=book_id,
        logger=Logger(),
        token_tracker=None,
        agent_language=_ScriptedLanguage([]),
    )

    formatted_toc = _format_toc(explorer._toc)
    goal = _GOAL_TEMPLATE.format(
        title=explorer._metadata["title"],
        authors=", ".join(explorer._metadata.get("authors", [])),
        language=explorer._metadata["language"],
        total_pages=explorer._total_pages,
        toc_section=formatted_toc,
    )

    # Metadata fields land in the goal text.
    assert "Veridian Settlement" in goal
    assert "Total pages:  4" in goal
    # TOC entries land in the goal text — at least one chapter title
    # from our synthetic fixture.
    assert "The Founding" in goal


# ── Caching ───────────────────────────────────────────────────────────────


def test_cached_plan_is_reused_without_calling_the_agent(tmp_books_dir: Path):
    book_id = "veridian_settlement"
    create_tiny_book(tmp_books_dir, book_id=book_id, num_pages=4)

    # First run: agent produces a plan in a single LLM call.
    first_lang = _ScriptedLanguage(
        scripted_responses=[
            json.dumps(
                {
                    "tool": "set_reading_strategy",
                    "args": {
                        "entries": [
                            {
                                "page_from": 1,
                                "page_to": 4,
                                "importance": "read",
                                "reason": "All content",
                            }
                        ]
                    },
                }
            ),
        ]
    )
    plan1 = _explorer(book_id, first_lang).explore()
    assert first_lang.call_count == 1

    # Second run: cached plan is reused; agent never invoked at all.
    # If the loop did run, it would raise ("more calls than scripted=0").
    second_lang = _ScriptedLanguage(scripted_responses=[])
    plan2 = _explorer(book_id, second_lang).explore()
    assert second_lang.call_count == 0
    assert plan2.entries == plan1.entries


def test_stale_cache_is_regenerated_when_total_pages_changes(
    tmp_books_dir: Path,
):
    book_id = "veridian_settlement"
    create_tiny_book(tmp_books_dir, book_id=book_id, num_pages=4)

    first_lang = _ScriptedLanguage(
        scripted_responses=[
            json.dumps(
                {
                    "tool": "set_reading_strategy",
                    "args": {
                        "entries": [
                            {
                                "page_from": 1,
                                "page_to": 4,
                                "importance": "read",
                                "reason": "All content",
                            }
                        ]
                    },
                }
            ),
        ]
    )
    _explorer(book_id, first_lang).explore()
    assert first_lang.call_count == 1

    # Book grows from 4 to 6 pages — old plan total_pages=4 is stale.
    create_tiny_book(tmp_books_dir, book_id=book_id, num_pages=6)

    second_lang = _ScriptedLanguage(
        scripted_responses=[
            json.dumps(
                {
                    "tool": "set_reading_strategy",
                    "args": {
                        "entries": [
                            {
                                "page_from": 1,
                                "page_to": 6,
                                "importance": "read",
                                "reason": "All content",
                            }
                        ]
                    },
                }
            ),
        ]
    )
    plan2 = _explorer(book_id, second_lang).explore()

    # Stale cache was discarded, agent re-ran with one new LLM call.
    assert second_lang.call_count == 1
    assert plan2.total_pages == 6


# ── Failure modes ─────────────────────────────────────────────────────────


def test_invalid_plan_args_return_validation_error_to_agent(
    tmp_books_dir: Path,
):
    """If the agent calls set_reading_strategy with malformed args, the
    tool returns a structured validation error rather than crashing.
    set_reading_strategy is terminal so the loop ends, and the explorer
    falls back to an all-read plan."""
    book_id = "veridian_settlement"
    create_tiny_book(tmp_books_dir, book_id=book_id, num_pages=4)

    language = _ScriptedLanguage(
        scripted_responses=[
            json.dumps(
                {
                    "tool": "set_reading_strategy",
                    "args": {
                        "entries": [
                            # Gap: pages 3-4 are uncovered.
                            {
                                "page_from": 1,
                                "page_to": 2,
                                "importance": "read",
                                "reason": "incomplete",
                            },
                        ]
                    },
                }
            ),
        ]
    )

    plan = _explorer(book_id, language).explore()

    # Fallback: every page marked 'read'.
    assert len(plan.entries) == 1
    assert plan.entries[0].importance == "read"
    assert plan.entries[0].page_from == 1
    assert plan.entries[0].page_to == 4
    assert "Fallback" in plan.entries[0].reason


def test_agent_failure_falls_back_to_all_read_plan(tmp_books_dir: Path):
    """If the agent raises during run (e.g. real LLM unreachable) the
    explorer must still return a usable plan — never None."""
    book_id = "veridian_settlement"
    create_tiny_book(tmp_books_dir, book_id=book_id, num_pages=4)

    # Empty script → first generate_response raises RuntimeError.
    language = _ScriptedLanguage(scripted_responses=[])
    plan = _explorer(book_id, language).explore()

    # Fallback all-read plan covers every page.
    assert plan.total_pages == 4
    assert plan.pages_to_skip() == set()
    assert plan.entries[0].importance == "read"


# ── NoteTaker integration ────────────────────────────────────────────────


def test_note_taker_skips_chunks_marked_skip_in_reading_plan(
    tmp_books_dir: Path,
):
    """The whole point of the reading plan: NoteTaker should not call
    the LLM for chunks whose pages are entirely 'skip'."""
    from book_summarizer.book_explorer import (
        ReadingPlan,
        ReadingPlanEntry,
    )
    from book_summarizer.note_taker import NoteTaker

    book_id = "veridian_settlement"
    create_tiny_book(tmp_books_dir, book_id=book_id, num_pages=4)

    plan = ReadingPlan(
        book_id=book_id,
        total_pages=4,
        entries=[
            ReadingPlanEntry(
                page_from=1, page_to=2, importance="skip", reason="frontmatter"
            ),
            ReadingPlanEntry(
                page_from=3, page_to=4, importance="read", reason="content"
            ),
        ],
    )

    class _StubNoteWriter:
        def __init__(self):
            self.calls: list[dict] = []

        def write(
            self, *, page_from, page_to, page_text, book_metadata, toc_summary
        ):
            self.calls.append({"page_from": page_from, "page_to": page_to})
            return f"## Notes for {page_from}-{page_to}\n"

    writer = _StubNoteWriter()
    nt = NoteTaker(
        book_id=book_id,
        note_writer=writer,
        logger=Logger(),
        token_tracker=None,
        chunk_size=2,
        reading_plan=plan,
    )
    nt.run()

    # 4 pages / chunk_size 2 → 2 chunks: (1,2) is skip-only, (3,4) is read.
    # Only the read chunk should hit the LLM.
    assert writer.calls == [{"page_from": 3, "page_to": 4}]

    # All 4 pages are still recorded as covered (skip pages count).
    notes_dir = tmp_books_dir / book_id / "notes"
    state = json.loads((notes_dir / "_state.json").read_text())
    assert sorted(state["pages_read"]) == [1, 2, 3, 4]
    assert state["completed"] is True
    # Only one note file was written (for the read chunk).
    md_files = sorted(f.name for f in notes_dir.glob("*.md"))
    assert md_files == ["003-004.md"]
