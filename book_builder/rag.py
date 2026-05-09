"""
Per-book ChromaDB indexing.

Each book owns its own ``.books/<book_id>/chroma_db/`` directory — collections
are never shared. Chunks are built from the already-extracted ``pages/*.txt``
files, so indexing depends on ``build_book`` having run first.

Text is embedded with ``chromadb``'s default local model (all-MiniLM-L6-v2 via
onnxruntime) — no API key required.

This module is the **writer** for the vector index and the low-level accessor
for its chromadb collection. Query semantics (top-k, citation formatting,
query expansion, missing-book handling) are the consumer's responsibility —
see ``agents/personal_librarian/retrieval.py`` for one such consumer.
"""

import json

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from book_builder.normalize import normalize_persian_text
from book_builder.paths import book_dir, chroma_dir, collection_name

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

_EF = DefaultEmbeddingFunction()


def _chunk_text(text: str) -> list[str]:
    """Split *text* into overlapping character-level chunks."""
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    return chunks


def _load_metadata(book_id: str) -> dict:
    meta_file = book_dir(book_id) / "metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(
            f"No metadata.json for book '{book_id}'. Run build_book first."
        )
    return json.loads(meta_file.read_text(encoding="utf-8"))


def _page_to_pdf(metadata: dict, page_num: int) -> str:
    """Map a global page number back to its source PDF filename."""
    for rng in metadata.get("pdf_page_ranges", []):
        if rng["start"] <= page_num <= rng["end"]:
            return rng["file"]
    return metadata.get("source_files", [""])[0]


def _client(book_id: str) -> chromadb.PersistentClient:
    path = chroma_dir(book_id)
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def get_collection(book_id: str):
    """
    Return the ChromaDB collection for a book, with the same embedding
    function that was used at index time.

    Raises whatever chromadb raises when the collection does not exist
    (caller decides how to handle that — e.g. the librarian maps it to a
    friendly "book not indexed" message).
    """
    client = _client(book_id)
    return client.get_collection(collection_name(book_id), embedding_function=_EF)


def build_vector_db(book_id: str, force: bool = False) -> int:
    """
    Index a processed book's pages into its per-book ChromaDB.

    Returns the number of chunks indexed. Assumes ``build_book`` has run
    (i.e. ``.books/<book_id>/pages/*.txt`` exist).
    """
    bdir = book_dir(book_id)
    pages_dir = bdir / "pages"
    if not pages_dir.exists():
        raise FileNotFoundError(
            f"No pages/ for book '{book_id}'. Run build_book first."
        )

    metadata = _load_metadata(book_id)
    col_name = collection_name(book_id)
    client = _client(book_id)

    if not force:
        try:
            col = client.get_collection(col_name, embedding_function=_EF)
            if col.count() > 0:
                print(
                    f"[rag] '{metadata.get('title', book_id)}' already indexed "
                    f"({col.count()} chunks). Skipping."
                )
                return col.count()
        except Exception:
            pass

    if force:
        try:
            client.delete_collection(col_name)
        except Exception:
            pass

    rtl_fix = metadata.get("rtl_word_order_fix", False)

    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_metas: list[dict] = []

    for page_file in sorted(pages_dir.glob("*.txt")):
        page_num = int(page_file.stem)
        raw = page_file.read_text(encoding="utf-8")
        # Already normalized at build time, but re-apply in case pages were
        # written by an older tool or edited by hand.
        text = normalize_persian_text(raw, fix_rtl_word_order=rtl_fix)
        source_file = _page_to_pdf(metadata, page_num)
        for chunk_idx, chunk in enumerate(_chunk_text(text)):
            all_chunks.append(chunk)
            all_ids.append(f"{book_id}_p{page_num}_c{chunk_idx}")
            all_metas.append({
                "book_id": book_id,
                "book_title": metadata.get("title", book_id),
                "source_file": source_file,
                "page": page_num,
                "chunk_index": chunk_idx,
            })

    if not all_chunks:
        print(f"[rag] No extractable text in '{metadata.get('title', book_id)}'.")
        return 0

    col = client.get_or_create_collection(
        col_name,
        embedding_function=_EF,
        metadata={
            "title": metadata.get("title", book_id),
            "authors": json.dumps(metadata.get("authors", []), ensure_ascii=False),
            "language": metadata.get("language", "unknown"),
        },
    )
    col.upsert(documents=all_chunks, ids=all_ids, metadatas=all_metas)
    print(
        f"[rag] ✓ '{metadata.get('title', book_id)}' indexed "
        f"({len(all_chunks)} chunks across {metadata.get('total_pages', '?')} pages)."
    )
    return len(all_chunks)
