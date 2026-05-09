"""Shared pytest fixtures.

The integration suite needs ``.books/`` redirected to a temp directory so
each test starts from a clean state and never touches the real
``<project_root>/.books/``.

``BOOKS_DIR`` is a module-level constant set at import time, so monkey-
patching only ``book_builder.paths.BOOKS_DIR`` is not enough — modules
that did ``from book_builder.paths import BOOKS_DIR`` already hold their
own reference to the original value. Each such reference must be patched
explicitly. ``book_dir()`` looks up ``BOOKS_DIR`` from its module's
namespace at call time, so patching ``book_builder.paths.BOOKS_DIR`` is
sufficient for any code path that goes through that helper.
"""

from pathlib import Path

import pytest


@pytest.fixture
def tmp_books_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``.books/`` to a temp directory for the duration of a test."""
    fake_books = tmp_path / ".books"
    fake_books.mkdir()

    import book_builder.paths
    import book_builder.reader

    monkeypatch.setattr(book_builder.paths, "BOOKS_DIR", fake_books)
    monkeypatch.setattr(book_builder.reader, "BOOKS_DIR", fake_books)

    return fake_books
