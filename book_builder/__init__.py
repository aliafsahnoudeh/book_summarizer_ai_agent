"""
Reusable book-building tool.

Extracts text, TOC, images, and metadata from PDFs into
``<project_root>/.books/<book_id>/``, and optionally builds a per-book
ChromaDB index. Handles Persian/Arabic text (NFKC + character
canonicalization + optional RTL word-order fix) as well as English and
other scripts.

Scope: this package only *produces* the book artifacts and the low-level
accessor for its ChromaDB collection (``get_collection``). Application-
level retrieval (top-k, citation formatting, query expansion) belongs to
consumers.
"""

from book_builder.builder import BuildResult, build_book
from book_builder.normalize import normalize_persian_text
from book_builder.paths import (
    BOOKS_DIR,
    book_dir,
    book_id_from_source,
    chroma_dir,
    collection_name,
    sanitize,
)
from book_builder.rag import build_vector_db, get_collection
from book_builder.reader import BookReader, list_books

__all__ = [
    "BOOKS_DIR",
    "BookReader",
    "BuildResult",
    "book_dir",
    "book_id_from_source",
    "build_book",
    "build_vector_db",
    "chroma_dir",
    "collection_name",
    "get_collection",
    "list_books",
    "normalize_persian_text",
    "sanitize",
]
