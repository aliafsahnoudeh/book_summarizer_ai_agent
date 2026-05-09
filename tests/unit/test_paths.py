"""Regression test for ``book_builder.paths.PROJECT_ROOT``.

When the package was flattened out of ``agents/book_summarizer/tools/``,
the ``parent.parent.parent`` math in paths.py left ``PROJECT_ROOT``
pointing one directory ABOVE the actual project, so ``build-book``
silently wrote ``.books/`` outside the project tree. The bug was
invisible to the rest of the suite because conftest monkey-patches
``BOOKS_DIR`` to a tmp dir, never exercising the real path math.

This test pins the contract that ``PROJECT_ROOT`` resolves to the
directory containing ``pyproject.toml`` — the canonical project root —
so any future move of paths.py that breaks this is caught on the next
test run.
"""

from book_builder.paths import BOOKS_DIR, PROJECT_ROOT


def test_project_root_contains_pyproject_toml():
    pyproject = PROJECT_ROOT / "pyproject.toml"
    assert pyproject.exists(), (
        f"PROJECT_ROOT={PROJECT_ROOT} does not contain pyproject.toml — "
        "the path math in book_builder/paths.py is off."
    )


def test_books_dir_lives_inside_project_root():
    assert BOOKS_DIR == PROJECT_ROOT / ".books"
    assert BOOKS_DIR.parent == PROJECT_ROOT
