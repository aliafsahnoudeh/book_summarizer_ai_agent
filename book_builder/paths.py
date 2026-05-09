"""
Shared paths and naming for the book builder tool.

All processed book data lives under ``<project_root>/.books/<book_id>/``.
A ``book_id`` is a sanitized, filesystem- and ChromaDB-safe identifier
derived from the source PDF stem or source folder name.
"""

import re
from pathlib import Path

# Project root = parent of the ``book_builder/`` package this file lives in.
# (Was ``parent.parent.parent`` when the package lived at
# ``agents/book_summarizer/tools/book_builder/`` — corrected after the
# flatten so ``BOOKS_DIR`` resolves inside the actual project tree.)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOKS_DIR = PROJECT_ROOT / ".books"


def sanitize(text: str, max_len: int = 58) -> str:
    """Return a filesystem- and ChromaDB-safe identifier."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:max_len].rstrip("_")


def book_id_from_source(source: Path) -> str:
    """Derive a stable book_id from a source file or folder path."""
    stem = source.stem if source.is_file() else source.name
    return sanitize(stem)


def book_dir(book_id: str) -> Path:
    """Directory for a processed book."""
    return BOOKS_DIR / book_id


def chroma_dir(book_id: str) -> Path:
    """Per-book ChromaDB directory."""
    return book_dir(book_id) / "chroma_db"


def collection_name(book_id: str) -> str:
    """ChromaDB collection name (max 63 chars)."""
    return f"book_{book_id}"[:63]
