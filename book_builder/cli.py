"""
CLI entry point for the book builder.

Usage::

    uv run build-book path/to/book.pdf
    uv run build-book path/to/folder-of-pdfs
    uv run build-book path/to/book.pdf --force       # overwrite existing
    uv run build-book path/to/book.pdf --no-embed    # skip chroma_db build
    uv run build-book --embed-only <book_id>         # index an already-built book

Output goes to ``<project_root>/.books/<book_id>/``.
"""

import argparse
import sys
from pathlib import Path

from book_builder.builder import build_book
from book_builder.paths import BOOKS_DIR, book_id_from_source
from book_builder.rag import build_vector_db


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract text, TOC, and images from a PDF (or folder of PDFs) "
            "into .books/<book_id>/, and optionally build a ChromaDB index."
        )
    )
    parser.add_argument(
        "source",
        help="Path to a PDF file, a folder of PDFs, or (with --embed-only) a book_id",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing book data (text + chroma_db)",
    )
    parser.add_argument(
        "--no-embed", action="store_true",
        help="Skip building the ChromaDB index",
    )
    parser.add_argument(
        "--embed-only", action="store_true",
        help="Skip text extraction; only (re)build the ChromaDB index",
    )
    args = parser.parse_args()

    if args.embed_only:
        # Treat source as either an existing book_id or a path to derive one from.
        candidate = Path(args.source)
        if (BOOKS_DIR / args.source).is_dir():
            book_id = args.source
        elif candidate.exists():
            book_id = book_id_from_source(candidate.resolve())
        else:
            print(f"[build-book] ERROR: no book found for '{args.source}'")
            sys.exit(1)
        build_vector_db(book_id, force=args.force)
        return

    source_path = Path(args.source).resolve()
    if not source_path.exists():
        print(f"[build-book] ERROR: source not found: {source_path}")
        sys.exit(1)

    try:
        result = build_book(source_path, force=args.force)
    except FileExistsError as e:
        print(f"[build-book] {e}")
        sys.exit(1)

    if not args.no_embed:
        print("[build-book] Building ChromaDB index…")
        build_vector_db(result.book_id, force=args.force)

    print(f"[build-book] Done: book_id = {result.book_id}")


if __name__ == "__main__":
    main()
