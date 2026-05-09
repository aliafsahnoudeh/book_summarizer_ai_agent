"""Unit tests for NoteIndexer.

Uses a deterministic stub embedder so tests don't have to download the
~25 MB sentence-transformers model. The stub produces different (but
deterministic) vectors for different inputs, which is enough to
exercise the index-build / query / dedup logic without depending on
real semantic similarity rankings.
"""

import pytest

from book_summarizer.note_indexer import NoteIndexer, _parse_note_filename
from zurvan import Logger


# ── Stub embedder ─────────────────────────────────────────────────────────


class _DeterministicEmbedder:
    """Tiny embedder: maps each text to an 8-D vector by summing
    char-codes per position-mod-8. Different texts → different vectors;
    same text → identical vector. Enough for ChromaDB's nearest-neighbour
    machinery to function without any model download.

    Implements ChromaDB's modern embedder interface: ``__call__`` for
    legacy callers, ``embed_documents`` for the index-write path, and
    ``embed_query`` for the index-read path. ``is_legacy()`` returning
    ``False`` opts into the new contract.
    """

    def __call__(self, input):
        return self._embed_batch(input)

    def embed_documents(self, input):
        return self._embed_batch(input)

    def embed_query(self, input):
        return self._embed_batch(input)

    def _embed_batch(self, texts):
        out = []
        for text in texts:
            vec = [0.0] * 8
            for i, ch in enumerate(text[:512]):
                vec[i % 8] += (ord(ch) % 64) / 64.0
            out.append(vec)
        return out

    def is_legacy(self) -> bool:
        return False

    # ChromaDB calls this introspection method on the embedding function
    # to fingerprint the collection. Returning a stable name keeps the
    # collection's "embedding_function consistency" check happy.
    def name(self) -> str:
        return "test-deterministic-8d"


# ── Helpers ───────────────────────────────────────────────────────────────


def _seed_notes(notes_dir, *files: tuple[str, str]) -> None:
    """Write each (filename, content) pair to notes_dir."""
    notes_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files:
        (notes_dir / name).write_text(content, encoding="utf-8")


def _make_indexer(book_id: str) -> NoteIndexer:
    return NoteIndexer(
        book_id=book_id,
        logger=Logger(),
        embedding_function=_DeterministicEmbedder(),
    )


# ── Tests ────────────────────────────────────────────────────────────────


def test_parse_filename_extracts_page_range():
    assert _parse_note_filename("001-004.md") == (1, 4)
    assert _parse_note_filename("100-237.md") == (100, 237)
    assert _parse_note_filename("not-a-notes-file.md") is None
    assert _parse_note_filename("001-004.txt") is None


def test_index_builds_with_one_document_per_note_file(tmp_books_dir):
    book_id = "test_book"
    notes_dir = tmp_books_dir / book_id / "notes"
    _seed_notes(
        notes_dir,
        ("001-004.md", "Notes for chapter 1: introduction and premise."),
        ("005-008.md", "Notes for chapter 2: main argument follows..."),
        ("009-012.md", "Notes for chapter 3: counterexamples and edge cases."),
    )

    indexer = _make_indexer(book_id)
    assert indexer.collection.count() == 3


def test_index_skips_files_that_dont_match_chunk_naming(tmp_books_dir):
    book_id = "test_book"
    notes_dir = tmp_books_dir / book_id / "notes"
    _seed_notes(
        notes_dir,
        ("001-004.md", "Real chunk note."),
        ("README.md", "Stray file — should not be indexed."),
        ("scratch.md", "Also not a chunk note."),
    )

    indexer = _make_indexer(book_id)
    assert indexer.collection.count() == 1


def test_query_returns_results_with_expected_shape(tmp_books_dir):
    book_id = "test_book"
    notes_dir = tmp_books_dir / book_id / "notes"
    _seed_notes(
        notes_dir,
        ("001-004.md", "Pages 1-4 cover the introduction and premise."),
        ("005-008.md", "Pages 5-8 cover the central argument."),
    )

    results = _make_indexer(book_id).query("introduction", n_results=2)
    assert len(results) == 2
    for r in results:
        assert "document" in r
        assert "metadata" in r
        assert r["metadata"]["file"] in ("001-004.md", "005-008.md")
        assert "page_from" in r["metadata"]
        assert "page_to" in r["metadata"]
        assert "distance" in r


def test_query_returns_empty_list_when_index_is_empty(tmp_books_dir):
    book_id = "empty_book"
    (tmp_books_dir / book_id / "notes").mkdir(parents=True, exist_ok=True)

    results = _make_indexer(book_id).query("anything", n_results=5)
    assert results == []


def test_query_themes_dedupes_by_file_and_sorts_by_page(tmp_books_dir):
    """If the same chunk note matches multiple theme queries, it
    appears only once in the union; the union is sorted by page order
    so the composer reads results in book sequence."""
    book_id = "test_book"
    notes_dir = tmp_books_dir / book_id / "notes"
    _seed_notes(
        notes_dir,
        ("001-004.md", "Pages 1-4: introduction and main thesis."),
        ("005-008.md", "Pages 5-8: argument and evidence."),
        ("009-012.md", "Pages 9-12: conclusions and implications."),
    )

    notes = _make_indexer(book_id).query_themes(
        queries=[
            "main thesis",
            "key conclusions",
            "argument evidence",
        ],
        n_per_query=2,
    )

    # Every result has the expected schema.
    for n in notes:
        assert {"file", "page_from", "page_to", "content"} <= n.keys()
    # Files are unique (deduped).
    files = [n["file"] for n in notes]
    assert len(files) == len(set(files))
    # Sorted by page_from ascending.
    page_starts = [n["page_from"] for n in notes]
    assert page_starts == sorted(page_starts)


def test_index_is_reused_when_files_unchanged(tmp_books_dir):
    """Building twice in a row with no changes should reuse the existing
    collection — no duplicate documents, no rebuild work."""
    book_id = "test_book"
    notes_dir = tmp_books_dir / book_id / "notes"
    _seed_notes(
        notes_dir,
        ("001-004.md", "First chunk."),
        ("005-008.md", "Second chunk."),
    )

    first = _make_indexer(book_id)
    first_count = first.collection.count()

    # Brand new instance, same notes on disk.
    second = _make_indexer(book_id)
    second_count = second.collection.count()

    assert first_count == second_count == 2


def test_index_rebuilds_when_a_note_is_added(tmp_books_dir):
    """Adding a new chunk note after the first build should be picked
    up — the reconciler detects the new on-disk file and rebuilds."""
    book_id = "test_book"
    notes_dir = tmp_books_dir / book_id / "notes"
    _seed_notes(
        notes_dir,
        ("001-004.md", "First chunk."),
        ("005-008.md", "Second chunk."),
    )

    assert _make_indexer(book_id).collection.count() == 2

    # Append a new chunk file.
    (notes_dir / "009-012.md").write_text("Third chunk.", encoding="utf-8")

    assert _make_indexer(book_id).collection.count() == 3
