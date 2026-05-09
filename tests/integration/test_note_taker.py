"""Integration tests for NoteTaker — the Python loop that drives per-chunk
note-writing across a book.

Mocks at the ``ChunkNoteWriter`` boundary (duck-typed stub with the same
``write()`` signature). This proves the loop itself — chunk iteration,
state persistence, resumability, invalidation — without burning LLM tokens.
"""

import json
from pathlib import Path

import pytest

from book_summarizer.note_taker import NoteTaker
from zurvan import Logger
from tests.fixtures import create_tiny_book


class _StubNoteWriter:
    """Drop-in duck-type for ChunkNoteWriter that records calls."""

    def __init__(self):
        self.calls: list[dict] = []

    def write(
        self,
        *,
        page_from: int,
        page_to: int,
        page_text: str,
        book_metadata: dict,
        toc_summary: str,
    ) -> str:
        self.calls.append(
            {
                "page_from": page_from,
                "page_to": page_to,
                "text_len": len(page_text),
            }
        )
        return f"## Notes for pages {page_from}-{page_to}\n\nMocked content.\n"


def _build(book_id: str, chunk_size: int = 2) -> tuple[NoteTaker, _StubNoteWriter]:
    writer = _StubNoteWriter()
    note_taker = NoteTaker(
        book_id=book_id,
        note_writer=writer,
        logger=Logger(),
        token_tracker=None,
        chunk_size=chunk_size,
    )
    return note_taker, writer


def test_writes_one_file_per_chunk_and_persists_state(tmp_books_dir: Path):
    book_id = "veridian_settlement"
    create_tiny_book(tmp_books_dir, book_id=book_id, num_pages=4)

    note_taker, writer = _build(book_id, chunk_size=2)
    note_taker.run()

    # 4 pages / chunk_size 2 → exactly 2 LLM calls.
    assert len(writer.calls) == 2
    assert writer.calls[0]["page_from"] == 1 and writer.calls[0]["page_to"] == 2
    assert writer.calls[1]["page_from"] == 3 and writer.calls[1]["page_to"] == 4

    notes_dir = tmp_books_dir / book_id / "notes"
    files = sorted(f.name for f in notes_dir.glob("*.md"))
    assert files == ["001-002.md", "003-004.md"]

    # Each note file starts with the page-range comment.
    assert (notes_dir / "001-002.md").read_text().startswith("<!-- pages 1-2 -->")

    state = json.loads((notes_dir / "_state.json").read_text())
    assert state["completed"] is True
    assert sorted(state["pages_read"]) == [1, 2, 3, 4]
    assert state["chunk_size"] == 2


def test_resumes_from_existing_state_without_redoing_work(tmp_books_dir: Path):
    book_id = "veridian_settlement"
    create_tiny_book(tmp_books_dir, book_id=book_id, num_pages=4)

    # First run completes the book.
    nt1, writer1 = _build(book_id, chunk_size=2)
    nt1.run()
    assert len(writer1.calls) == 2

    # Second run: state says completed=True → no LLM calls.
    nt2, writer2 = _build(book_id, chunk_size=2)
    nt2.run()
    assert writer2.calls == []


def test_partial_state_skips_already_processed_chunks(tmp_books_dir: Path):
    """If only some chunks were processed in a prior run, only the missing
    ones are processed on a follow-up run."""
    book_id = "veridian_settlement"
    create_tiny_book(tmp_books_dir, book_id=book_id, num_pages=6)
    notes_dir = tmp_books_dir / book_id / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    # Pre-seed state: pages 1-2 and 3-4 already done; 5-6 still pending.
    # Total pages 6 × 80% coverage = 4.8 → 5 required for completion,
    # so 4 pages_read leaves the book NOT complete and forces continuation.
    (notes_dir / "001-002.md").write_text(
        "<!-- pages 1-2 -->\n## Notes\nPrior run.\n", encoding="utf-8"
    )
    (notes_dir / "003-004.md").write_text(
        "<!-- pages 3-4 -->\n## Notes\nPrior run.\n", encoding="utf-8"
    )
    (notes_dir / "_state.json").write_text(
        json.dumps(
            {
                "book_id": book_id,
                "language": "en",
                "total_pages": 6,
                "source_files": ["fixture.pdf"],
                "pages_read": [1, 2, 3, 4],
                "chunks": [
                    {"file": "001-002.md", "page_from": 1, "page_to": 2},
                    {"file": "003-004.md", "page_from": 3, "page_to": 4},
                ],
                "completed": False,
                "chunk_size": 2,
            }
        ),
        encoding="utf-8",
    )

    nt, writer = _build(book_id, chunk_size=2)
    nt.run()

    # Only pages 5-6 should be re-processed.
    assert len(writer.calls) == 1
    assert writer.calls[0]["page_from"] == 5 and writer.calls[0]["page_to"] == 6


def test_state_invalidated_when_total_pages_changes(tmp_books_dir: Path):
    book_id = "veridian_settlement"
    create_tiny_book(tmp_books_dir, book_id=book_id, num_pages=4)

    nt1, writer1 = _build(book_id, chunk_size=2)
    nt1.run()
    assert len(writer1.calls) == 2

    # Book grows to 6 pages — old state's total_pages=4 doesn't match.
    create_tiny_book(tmp_books_dir, book_id=book_id, num_pages=6)

    nt2, writer2 = _build(book_id, chunk_size=2)
    nt2.run()

    # All 6 pages should be reprocessed (state was invalidated on construction).
    assert len(writer2.calls) == 3
    starts = sorted(c["page_from"] for c in writer2.calls)
    assert starts == [1, 3, 5]
