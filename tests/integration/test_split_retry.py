"""Integration tests for NoteTaker's auto-split-on-TPM-overflow recovery.

The motivating bug: on Groq the smart default of 4 pages per chunk fits
the 6 K TPM cap on average, but a denser chunk can still spike over.
Without recovery the chunk is skipped, leaving a hole in the notes
corpus and degrading the final summary. With recovery the chunk is
split in half, retried, and recursively split further if a half is
still too large — so a successful run on dense content costs more LLM
calls but produces complete notes.
"""

import json
from pathlib import Path

from book_summarizer.note_taker import ChunkTooLargeError, NoteTaker
from zurvan import Logger
from tests.fixtures import create_tiny_book


class _SizeAwareStubNoteWriter:
    """Stub that fails ``ChunkTooLargeError`` on chunks above a size threshold,
    to simulate Groq's TPM-overflow behaviour deterministically.
    """

    def __init__(self, fail_above_pages: int):
        self.calls: list[dict] = []
        self._fail_above = fail_above_pages

    def write(
        self,
        *,
        page_from: int,
        page_to: int,
        page_text: str,
        book_metadata: dict,
        toc_summary: str,
    ) -> str:
        n_pages = page_to - page_from + 1
        self.calls.append(
            {"page_from": page_from, "page_to": page_to, "n_pages": n_pages}
        )
        if n_pages > self._fail_above:
            raise ChunkTooLargeError(
                f"stub: {n_pages} pages > fail_above={self._fail_above}"
            )
        return f"## Notes for pages {page_from}-{page_to}\n\nMocked.\n"


def _build(book_id: str, writer, chunk_size: int = 4) -> NoteTaker:
    return NoteTaker(
        book_id=book_id,
        note_writer=writer,
        logger=Logger(),
        token_tracker=None,
        chunk_size=chunk_size,
    )


def test_oversized_chunk_is_split_in_half_and_retried(tmp_books_dir: Path):
    """4-page chunk overflows; auto-split should yield two 2-page chunks
    that succeed. End state: every page covered, no failed_ranges."""
    book_id = "veridian_settlement"
    create_tiny_book(tmp_books_dir, book_id=book_id, num_pages=4)

    # Stub fails on > 3 pages (so chunk_size=4 fails the first time).
    writer = _SizeAwareStubNoteWriter(fail_above_pages=3)
    nt = _build(book_id, writer, chunk_size=4)
    nt.run()

    # Three calls total: 1 failed (4 pages), 2 succeeded (2 pages each).
    assert [c["n_pages"] for c in writer.calls] == [4, 2, 2]

    # Both halves should have been saved as separate note files.
    notes_dir = tmp_books_dir / book_id / "notes"
    files = sorted(f.name for f in notes_dir.glob("*.md"))
    assert files == ["001-002.md", "003-004.md"]

    # Final state: every page covered, run completed.
    state = json.loads((notes_dir / "_state.json").read_text())
    assert sorted(state["pages_read"]) == [1, 2, 3, 4]
    assert state["completed"] is True


def test_recursive_split_until_single_pages(tmp_books_dir: Path):
    """4-page chunk → split to 2+2 → still both fail → split each to
    1+1 → all four single-page chunks succeed. Worst-case path."""
    book_id = "veridian_settlement"
    create_tiny_book(tmp_books_dir, book_id=book_id, num_pages=4)

    # Stub only accepts single pages.
    writer = _SizeAwareStubNoteWriter(fail_above_pages=1)
    nt = _build(book_id, writer, chunk_size=4)
    nt.run()

    # Call sequence: 4 pages (fail) → 2+2 (both fail) → 1+1+1+1 (all succeed).
    sizes = [c["n_pages"] for c in writer.calls]
    assert sizes == [4, 2, 1, 1, 2, 1, 1]

    notes_dir = tmp_books_dir / book_id / "notes"
    files = sorted(f.name for f in notes_dir.glob("*.md"))
    assert files == ["001-001.md", "002-002.md", "003-003.md", "004-004.md"]

    state = json.loads((notes_dir / "_state.json").read_text())
    assert sorted(state["pages_read"]) == [1, 2, 3, 4]
    assert state["completed"] is True


def test_single_page_overflow_is_logged_and_skipped(tmp_books_dir: Path):
    """If even a single-page chunk exceeds the cap, splitting can't help.
    The page is logged + skipped; the run continues with the rest."""
    book_id = "veridian_settlement"
    create_tiny_book(tmp_books_dir, book_id=book_id, num_pages=4)

    # Reject everything — every chunk size, including 1 page, fails.
    writer = _SizeAwareStubNoteWriter(fail_above_pages=0)
    nt = _build(book_id, writer, chunk_size=2)

    # MAX_CONSECUTIVE_FAILURES is 5; we have 2 chunks at chunk_size=2,
    # each splitting to 2 single-page failures = 4 total failures.
    # Below the bail-out threshold; run should complete normally
    # (with no notes saved).
    nt.run()

    # Both halves of every chunk were attempted independently — even
    # though every call fails, the right half is not abandoned just
    # because the left's base case fired. That's the auto-recovery
    # contract: try every leaf, save what we can.
    sizes = [c["n_pages"] for c in writer.calls]
    assert sizes == [2, 1, 1, 2, 1, 1]

    notes_dir = tmp_books_dir / book_id / "notes"
    md_files = list(notes_dir.glob("*.md"))
    assert md_files == []  # nothing succeeded

    # State persisted; book was attempted but did not meet coverage.
    state = json.loads((notes_dir / "_state.json").read_text())
    assert state["pages_read"] == []
    assert state["completed"] is False


def test_partial_split_saves_succeeding_half_even_when_other_fails(
    tmp_books_dir: Path,
):
    """When the left half can't be split further (single page that still
    overflows) but the right half is fine, we must still save the right
    half. The bug we're guarding against: re-raising from the left's
    base case unwinds the stack before the right half runs."""
    book_id = "veridian_settlement"
    create_tiny_book(tmp_books_dir, book_id=book_id, num_pages=4)

    # A custom stub: page 1 alone is "too dense" no matter what; pages
    # 2-4 fit fine in any chunk size.
    class _OneStubbornPage(_SizeAwareStubNoteWriter):
        def __init__(self):
            super().__init__(fail_above_pages=10)  # never fails on size
            self.failed_pages: list[int] = []

        def write(self, **kwargs):
            self.calls.append(
                {
                    "page_from": kwargs["page_from"],
                    "page_to": kwargs["page_to"],
                    "n_pages": kwargs["page_to"] - kwargs["page_from"] + 1,
                }
            )
            # Fail any chunk that includes page 1.
            if kwargs["page_from"] <= 1 <= kwargs["page_to"]:
                raise ChunkTooLargeError(
                    "stub: chunk includes 'too dense' page 1"
                )
            return f"## Notes for {kwargs['page_from']}-{kwargs['page_to']}\n"

    writer = _OneStubbornPage()
    nt = _build(book_id, writer, chunk_size=4)
    nt.run()

    # Chunk (1,4) fails → split to (1,2) and (3,4).
    # (1,2) fails (contains page 1) → split to (1,1) and (2,2).
    #   (1,1) fails (single-page base case → raises).
    #   But (2,2) MUST still be attempted independently.
    # (3,4) succeeds straight away.
    notes_dir = tmp_books_dir / book_id / "notes"
    files = sorted(f.name for f in notes_dir.glob("*.md"))
    # Page 1 lost; pages 2, 3, 4 all saved.
    assert "002-002.md" in files
    assert "003-004.md" in files
    assert "001-001.md" not in files

    state = json.loads((notes_dir / "_state.json").read_text())
    assert sorted(state["pages_read"]) == [2, 3, 4]
