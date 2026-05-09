"""
Read-only access to a processed book at ``.books/<book_id>/``.

A ``BookReader`` is bound to one book and exposes page text, the table of
contents, keyword search, and image metadata. Module-level ``list_books()``
enumerates every processed book on disk.
"""

import json
from pathlib import Path

from book_builder.paths import BOOKS_DIR, book_dir


def list_books() -> list[dict]:
    """Enumerate every processed book under ``.books/``."""
    if not BOOKS_DIR.exists():
        return []
    out: list[dict] = []
    for item in sorted(BOOKS_DIR.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue
        meta_file = item / "metadata.json"
        if not meta_file.exists():
            continue
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({
            "id": meta.get("book_id", item.name),
            "title": meta.get("title", item.name),
            "authors": meta.get("authors", []),
            "language": meta.get("language", "unknown"),
            "total_pages": meta.get("total_pages", 0),
            "source_files": meta.get("source_files", []),
        })
    return out


class BookReader:
    """Read-only view of a processed book."""

    def __init__(self, book_id: str):
        self.book_id = book_id
        self._dir = book_dir(book_id)
        self._pages_dir = self._dir / "pages"
        self._metadata_file = self._dir / "metadata.json"
        self._toc_file = self._dir / "toc.json"
        self._visuals_index = self._dir / "visuals" / "index.json"

    @property
    def dir(self) -> Path:
        return self._dir

    def exists(self) -> bool:
        return self._metadata_file.exists() and self._pages_dir.exists()

    def metadata(self) -> dict:
        if not self._metadata_file.exists():
            return {
                "error": (
                    f"metadata.json not found for book '{self.book_id}'. "
                    f"Run `build-book` on the source PDF(s) first."
                )
            }
        meta = json.loads(self._metadata_file.read_text(encoding="utf-8"))
        if "total_pages" not in meta and self._pages_dir.exists():
            meta["total_pages"] = len(
                [p for p in self._pages_dir.iterdir() if p.suffix == ".txt"]
            )
        return meta

    def table_of_contents(self) -> list[dict] | str:
        if not self._toc_file.exists():
            return f"Table of contents not available for book '{self.book_id}'."
        return json.loads(self._toc_file.read_text(encoding="utf-8"))

    def content(self, page_from: int, page_to: int) -> str:
        """Concatenate pages *page_from..page_to* (inclusive, 1-based)."""
        if not self._pages_dir.exists():
            return f"Pages directory not found for book '{self.book_id}'."
        if page_from < 1:
            page_from = 1
        if page_to < page_from:
            return f"Invalid range: page_from ({page_from}) > page_to ({page_to})."

        parts: list[str] = []
        for page_num in range(page_from, page_to + 1):
            page_file = self._pages_dir / f"{page_num:03d}.txt"
            if page_file.exists():
                text = page_file.read_text(encoding="utf-8").strip()
                body = text if text else "(blank page)"
            else:
                body = "(page file missing)"
            parts.append(f"--- Page {page_num} ---\n{body}")
        return "\n\n".join(parts)

    def search_keywords(self, query: str) -> str:
        """Case-insensitive substring search with ±150 chars of context."""
        if not self._pages_dir.exists():
            return f"Pages directory not found for book '{self.book_id}'."
        q_lower = query.lower()
        matches: list[str] = []
        for page_file in sorted(self._pages_dir.glob("*.txt")):
            page_num = int(page_file.stem)
            text = page_file.read_text(encoding="utf-8")
            text_lower = text.lower()
            start = 0
            while True:
                idx = text_lower.find(q_lower, start)
                if idx == -1:
                    break
                ctx_start = max(0, idx - 150)
                ctx_end = min(len(text), idx + len(query) + 150)
                matches.append(
                    f"[Page {page_num}] …{text[ctx_start:ctx_end].strip()}…"
                )
                start = idx + 1
        return "\n\n".join(matches) if matches else f"No matches found for '{query}'."

    def page_visuals(self, page_number: int) -> list[dict] | str:
        if not self._visuals_index.exists():
            return f"Visual elements index not found for book '{self.book_id}'."
        index = json.loads(self._visuals_index.read_text(encoding="utf-8"))
        key = str(page_number)
        if key not in index:
            return f"No visual elements found on page {page_number}."
        return index[key]
