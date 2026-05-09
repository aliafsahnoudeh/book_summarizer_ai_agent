"""Unit tests for the BookExplorer Pydantic schemas.

The tool-input model (``SetReadingStrategyArgs``) is the security
boundary between the LLM and our filesystem — it must reject any
malformed plan **before** the function body runs. These tests pin the
contract: per-entry validation, ordering/contiguity, and the JSON-Schema
shape that ships to the model as the function-calling tool description.
"""

import pytest
from pydantic import ValidationError

from book_summarizer.book_explorer import (
    ReadingPlan,
    ReadingPlanEntry,
    SetReadingStrategyArgs,
)


# ── ReadingPlanEntry ──────────────────────────────────────────────────────


def test_entry_accepts_a_valid_range():
    e = ReadingPlanEntry(
        page_from=1, page_to=5, importance="read", reason="chapter 1"
    )
    assert e.page_from == 1 and e.page_to == 5
    assert e.importance == "read"


def test_entry_rejects_zero_or_negative_pages():
    with pytest.raises(ValidationError):
        ReadingPlanEntry(page_from=0, page_to=5, importance="read")
    with pytest.raises(ValidationError):
        ReadingPlanEntry(page_from=-1, page_to=5, importance="read")


def test_entry_rejects_page_to_before_page_from():
    with pytest.raises(ValidationError) as exc_info:
        ReadingPlanEntry(page_from=10, page_to=5, importance="read")
    assert "page_to" in str(exc_info.value)


def test_entry_rejects_unknown_importance_label():
    with pytest.raises(ValidationError):
        ReadingPlanEntry(page_from=1, page_to=5, importance="ignore")


def test_entry_allows_single_page_range():
    e = ReadingPlanEntry(page_from=7, page_to=7, importance="skip")
    assert e.page_from == e.page_to == 7


# ── SetReadingStrategyArgs ────────────────────────────────────────────────


def _entries(*tuples):
    return [
        ReadingPlanEntry(page_from=f, page_to=t, importance=imp, reason="")
        for f, t, imp in tuples
    ]


def test_args_accepts_a_contiguous_plan():
    args = SetReadingStrategyArgs(
        entries=_entries(
            (1, 3, "skip"),
            (4, 10, "read"),
            (11, 12, "skip"),
        )
    )
    assert [e.page_from for e in args.entries] == [1, 4, 11]


def test_args_sorts_unsorted_input():
    """LLMs sometimes return entries out of order. The validator
    re-sorts in place so downstream code can rely on ascending order."""
    args = SetReadingStrategyArgs(
        entries=_entries(
            (4, 10, "read"),
            (1, 3, "skip"),
            (11, 12, "skim"),
        )
    )
    assert [e.page_from for e in args.entries] == [1, 4, 11]


def test_args_rejects_a_gap_between_entries():
    with pytest.raises(ValidationError) as exc_info:
        SetReadingStrategyArgs(
            entries=_entries(
                (1, 3, "skip"),
                (5, 10, "read"),  # gap: page 4 is missing
            )
        )
    assert "Gap or overlap" in str(exc_info.value)


def test_args_rejects_an_overlap_between_entries():
    with pytest.raises(ValidationError) as exc_info:
        SetReadingStrategyArgs(
            entries=_entries(
                (1, 5, "skip"),
                (4, 10, "read"),  # overlap: pages 4-5 in both
            )
        )
    assert "Gap or overlap" in str(exc_info.value)


def test_args_rejects_empty_entries_list():
    with pytest.raises(ValidationError):
        SetReadingStrategyArgs(entries=[])


def test_args_json_schema_is_well_formed():
    """The JSON Schema is what gets shipped to the model as the
    function-calling tool description. Pin its top-level shape so we
    notice if a Pydantic upgrade silently changes it."""
    schema = SetReadingStrategyArgs.model_json_schema()
    assert schema["type"] == "object"
    assert "entries" in schema["properties"]
    assert schema["properties"]["entries"]["type"] == "array"
    assert "required" in schema and "entries" in schema["required"]


def test_inlined_schema_has_no_refs_or_defs():
    """We send the inlined schema (no ``$ref``/``$defs``) to the LLM —
    Gemini's function-calling layer doesn't reliably resolve refs and
    returns an empty ``choices`` list when it can't, which surfaces as
    a generic ``list index out of range`` error two layers up. Pin the
    contract that what we ship is fully self-contained."""
    from book_summarizer.book_explorer import _inline_json_schema_refs

    raw = SetReadingStrategyArgs.model_json_schema()
    inlined = _inline_json_schema_refs(raw)

    def _has_ref_or_def(obj) -> bool:
        if isinstance(obj, dict):
            if "$ref" in obj or "$defs" in obj:
                return True
            return any(_has_ref_or_def(v) for v in obj.values())
        if isinstance(obj, list):
            return any(_has_ref_or_def(v) for v in obj)
        return False

    assert not _has_ref_or_def(inlined), (
        f"Inlined schema still contains $ref or $defs:\n{inlined}"
    )

    # The structural constraints are still expressed inline — the items
    # entry now describes ReadingPlanEntry directly rather than via a ref.
    items = inlined["properties"]["entries"]["items"]
    assert items["type"] == "object"
    assert set(items["required"]) == {"page_from", "page_to", "importance"}
    assert items["properties"]["importance"]["enum"] == ["skip", "skim", "read"]


# ── ReadingPlan ──────────────────────────────────────────────────────────


def test_plan_pages_to_skip_collects_every_skip_page():
    plan = ReadingPlan(
        book_id="x",
        total_pages=12,
        entries=_entries(
            (1, 3, "skip"),
            (4, 10, "read"),
            (11, 12, "skip"),
        ),
    )
    assert plan.pages_to_skip() == {1, 2, 3, 11, 12}


def test_plan_importance_for_page_returns_label_for_each_page():
    plan = ReadingPlan(
        book_id="x",
        total_pages=10,
        entries=_entries(
            (1, 3, "skip"),
            (4, 7, "skim"),
            (8, 10, "read"),
        ),
    )
    assert plan.importance_for_page(1) == "skip"
    assert plan.importance_for_page(5) == "skim"
    assert plan.importance_for_page(9) == "read"


def test_plan_round_trips_through_disk(tmp_books_dir):
    plan = ReadingPlan(
        book_id="round_trip_book",
        total_pages=8,
        entries=_entries((1, 4, "skip"), (5, 8, "read")),
    )
    (tmp_books_dir / "round_trip_book").mkdir()
    saved_path = plan.save()
    assert saved_path.exists()

    loaded = ReadingPlan.load("round_trip_book")
    assert loaded is not None
    assert loaded.book_id == "round_trip_book"
    assert loaded.total_pages == 8
    assert len(loaded.entries) == 2


def test_plan_load_returns_none_when_missing(tmp_books_dir):
    assert ReadingPlan.load("no_such_book") is None
