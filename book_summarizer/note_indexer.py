"""Build and query a per-book vector index of chunk notes.

After NoteTaker writes the notes corpus, the SummaryComposer needs a
way to ask "give me the chunk notes most relevant to *X*" rather than
re-reading every file. This module turns ``.books/<id>/notes/*.md``
into a searchable ChromaDB collection so the composer can do focused
retrieval per theme (intro, main arguments, conclusions, etc.) instead
of stuffing the entire corpus into one prompt.

Design notes:

  * **Per-book collection.** Each book gets its own ChromaDB instance
    at ``.books/<id>/notes_index/`` — physical separation matches the
    pipeline's "everything-for-this-book lives together" contract.
    ``rm -rf .books/<id>/`` still wipes everything for that book.

  * **One document per chunk note file.** Notes are already structured
    Markdown (~1.5 K tokens each) — splitting them finer would drop
    local context that the composer benefits from. The note's
    page-range is preserved as metadata so retrieval results carry
    their provenance back.

  * **Idempotent build.** ``collection`` lazily reconciles the index
    against the on-disk note files: if every current note file is
    already indexed (matched by stable IDs), reuse. Otherwise rebuild
    cleanly. No staleness window between NoteTaker writing a new chunk
    and the next composer run.

  * **Local embeddings.** ChromaDB's ``DefaultEmbeddingFunction`` uses
    sentence-transformers MiniLM — runs on CPU, no API quota. First
    use downloads ~25 MB of model weights; cached afterwards.
"""

import re
from pathlib import Path
from typing import Any, Optional

from book_builder.paths import book_dir
from zurvan import LogLevel


# NoteTaker emits filenames like ``001-004.md`` (3-digit zero-padded).
_NOTE_FILENAME_RE = re.compile(r"^(\d{3,})-(\d{3,})\.md$")


def _parse_note_filename(name: str) -> Optional[tuple[int, int]]:
    """Extract ``(page_from, page_to)`` from a NoteTaker-emitted filename."""
    match = _NOTE_FILENAME_RE.match(name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


class NoteIndexer:
    """Build and query a vector index of a book's chunk notes.

    Lazy: ``collection`` builds (or loads) on first access so callers
    that never need retrieval never pay the indexing cost.
    """

    def __init__(
        self,
        book_id: str,
        logger,
        embedding_function: Optional[Any] = None,
    ):
        """
        Args:
            embedding_function: Optional override for ChromaDB's default
                ``DefaultEmbeddingFunction`` (sentence-transformers MiniLM).
                Tests pass a deterministic stub embedder so unit-test runs
                don't depend on downloading the ~25 MB transformer model.
                Production callers leave this ``None``.
        """
        self._book_id = book_id
        self._notes_dir = book_dir(book_id) / "notes"
        self._index_dir = book_dir(book_id) / "notes_index"
        self._logger = logger
        self._collection: Optional[Any] = None
        self._embedding_function = embedding_function

    # ── Public ────────────────────────────────────────────────────────────

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self._build_or_load()
        return self._collection

    def query(self, query_text: str, n_results: int = 5) -> list[dict]:
        """Top-K semantic-similarity hits for ``query_text``.

        Returns a list of ``{document, metadata, distance}`` dicts in
        ascending distance order (smaller = more similar).
        """
        coll = self.collection
        count = coll.count()
        if count == 0:
            return []
        results = coll.query(
            query_texts=[query_text],
            n_results=min(n_results, count),
        )
        return [
            {"document": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

    def query_themes(
        self, queries: list[str], n_per_query: int = 5
    ) -> list[dict]:
        """Retrieve notes for multiple themes; deduplicate; sort by page.

        Each query independently retrieves its top-K notes. The union
        (deduplicated by file) becomes the composer's context. We sort
        the combined hits by ``page_from`` so the composer reads them
        in book order — preserves narrative flow even though selection
        was theme-driven.

        Returns a list of ``{file, page_from, page_to, content}`` dicts.
        """
        seen: set[str] = set()
        notes: list[dict] = []
        for q in queries:
            for hit in self.query(q, n_results=n_per_query):
                file = hit["metadata"]["file"]
                if file in seen:
                    continue
                seen.add(file)
                notes.append(
                    {
                        "file": file,
                        "page_from": hit["metadata"]["page_from"],
                        "page_to": hit["metadata"]["page_to"],
                        "content": hit["document"],
                    }
                )
        notes.sort(key=lambda n: n["page_from"])
        return notes

    # ── Private ───────────────────────────────────────────────────────────

    def _build_or_load(self):
        """Reconcile the index with the current state of ``notes/``.

        If every on-disk note file already has a matching document in
        the collection (by stable ID), the existing index is reused.
        Otherwise the collection is wiped and rebuilt from the current
        files. This is cheap and correct: the alternative — checking
        timestamps or hashes — adds complexity for little gain on a
        per-book index that's at most ~few-hundred notes.
        """
        # Lazy imports — chromadb pulls in numpy etc., not all callers
        # of book_summarizer need it.
        import chromadb
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._index_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self._index_dir))
        # ChromaDB collection names are 3-63 chars; book_id is already
        # sanitized but may exceed 63 — truncate defensively.
        collection_name = f"notes_{self._book_id}"[:63]
        embedding_fn = self._embedding_function or DefaultEmbeddingFunction()
        collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
        )

        on_disk_files = sorted(
            f for f in self._notes_dir.glob("*.md")
            if _parse_note_filename(f.name) is not None
        )
        on_disk_ids = {self._stable_id(f.stem) for f in on_disk_files}
        existing = collection.get(ids=None)
        existing_ids = set(existing.get("ids") or [])

        if on_disk_ids and on_disk_ids == existing_ids:
            self._logger.log(
                f"NoteIndexer: reusing cached index "
                f"({len(existing_ids)} notes).",
                level=LogLevel.INFO,
                env=None,
            )
            return collection

        # Stale — rebuild cleanly.
        if existing_ids:
            collection.delete(ids=list(existing_ids))

        documents: list[str] = []
        metadatas: list[dict] = []
        ids: list[str] = []
        for note_file in on_disk_files:
            page_from, page_to = _parse_note_filename(note_file.name)
            documents.append(note_file.read_text(encoding="utf-8"))
            metadatas.append(
                {
                    "file": note_file.name,
                    "page_from": page_from,
                    "page_to": page_to,
                }
            )
            ids.append(self._stable_id(note_file.stem))

        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)

        self._logger.log(
            f"NoteIndexer: built index for '{self._book_id}' "
            f"({len(documents)} notes).",
            level=LogLevel.INFO,
            env=None,
        )
        return collection

    def _stable_id(self, file_stem: str) -> str:
        """Stable per-book document ID. Changing the scheme invalidates
        all existing indexes, which the rebuild path handles correctly."""
        return f"{self._book_id}::{file_stem}"


__all__ = ["NoteIndexer"]
