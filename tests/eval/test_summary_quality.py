"""End-to-end eval against the synthetic Veridian Settlement fixture.

Runs the full pipeline once per session (via ``short_summary_run``
fixture) and asserts properties of the produced summary. Combines two
flavours of check:

  - Deterministic regression assertions (string matching, length budget).
    These are fast, cheap, and don't depend on the judge model. They
    catch the most common breakage: pipeline produces nothing, the
    composer drops a key entity, or output ends up in the wrong file.
  - LLM-as-judge faithfulness scoring against the notes corpus. This
    catches a different failure class: the pipeline produces *something*
    that looks reasonable but invents facts not in the source.
"""

import pytest

from tests.eval.judge import faithfulness_judge


pytestmark = pytest.mark.eval


# ── Deterministic regression checks ───────────────────────────────────────


def test_summary_file_is_produced_and_non_trivial(short_summary_run):
    summary = short_summary_run["summary"]
    out_path = short_summary_run["out_path"]
    assert out_path.exists(), f"Summary file not written at {out_path}"
    assert len(summary.strip()) > 100, (
        f"Summary too short ({len(summary.strip())} chars) — "
        "probably the composer call returned empty or near-empty content."
    )


def test_summary_mentions_core_entities(short_summary_run):
    """Every reasonable short summary of this fixture must name the
    settlement, mention the founding year, and reference at least one of
    the central figures or the central technology. If none of these
    appear, the composer is dropping the subject of the book."""
    text = short_summary_run["summary"].lower()

    assert "veridian" in text, "Summary doesn't mention the settlement name"
    assert "2087" in text, "Summary doesn't mention the founding year"

    core_entities = ["tovari", "holm", "binding lattice"]
    found = [e for e in core_entities if e in text]
    assert found, (
        "Summary doesn't reference any central figure or technology "
        f"(expected at least one of {core_entities})"
    )


def test_summary_has_no_known_out_of_corpus_terms(short_summary_run):
    """Hard canaries: terms that are NOT in the source and have no
    reasonable interpretation given the topic. If any appear, the model
    is hallucinating or the test fixture has drifted."""
    text = short_summary_run["summary"].lower()

    # The fixture book is fictional and has no connection to ancient
    # Persia, the Roman Empire, or modern American politics. If any of
    # these terms appear, the composer is fabricating context.
    canaries = ["cyrus", "rome", "biden", "machine learning", "neural network"]
    found = [c for c in canaries if c in text]
    assert not found, f"Summary contains out-of-corpus content: {found}"


def test_short_summary_length_is_reasonable(short_summary_run):
    """``short`` ≈ 5% of the book per the spec; the fixture is tiny so
    the absolute floor is generous (a coherent paragraph or two), but a
    runaway summary several times the source length signals the level
    detection or composer prompt has regressed."""
    summary_chars = len(short_summary_run["summary"])
    notes_chars = len(short_summary_run["notes"])

    # Notes are typically ~1.5-2× the page text length (the LLM expands
    # with structure). A "short" summary larger than the notes corpus
    # is strictly wrong.
    assert summary_chars < notes_chars, (
        f"Short summary ({summary_chars} chars) is larger than the notes "
        f"corpus ({notes_chars} chars) it was synthesised from."
    )


# ── LLM-as-judge faithfulness ─────────────────────────────────────────────


def test_summary_is_faithful_to_notes(short_summary_run):
    verdict = faithfulness_judge(
        notes=short_summary_run["notes"],
        summary=short_summary_run["summary"],
    )
    assert verdict.score >= 1, (
        f"Faithfulness score {verdict.score}/2 — "
        f"unsupported claims: {verdict.unsupported_claims!r}\n"
        f"Judge rationale: {verdict.rationale}"
    )
