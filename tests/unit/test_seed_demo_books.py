"""Tests for the demo-book seeding step.

The seeding function runs on every web-app boot and decides what shows
up in the dropdown picker. Pinning its contract here means future
refactors (e.g. moving demo books into a sub-directory) don't silently
break the deployed demo's first-impression UX.
"""

from web.app import _seed_demo_books


def _make_book(parent, book_id: str) -> None:
    """Drop a minimal valid book skeleton at ``parent/<book_id>/``."""
    book = parent / book_id
    (book / "pages").mkdir(parents=True)
    (book / "metadata.json").write_text(
        f'{{"book_id": "{book_id}", "title": "{book_id}", '
        '"language": "en", "total_pages": 1, "source_files": []}}'
    )
    (book / "pages" / "001.txt").write_text("test page")


def test_seeds_books_when_dst_is_empty(tmp_path):
    src = tmp_path / "demo_books"
    src.mkdir()
    _make_book(src, "alpha")
    _make_book(src, "beta")

    dst = tmp_path / "books"

    seeded = _seed_demo_books(src=src, dst=dst)

    assert seeded == 2
    assert (dst / "alpha" / "metadata.json").exists()
    assert (dst / "beta" / "metadata.json").exists()


def test_skips_books_that_already_exist(tmp_path):
    """A user-uploaded book with the same id MUST NOT be clobbered by
    seeding. Likewise, idempotent re-seeds on container restart should
    be no-ops."""
    src = tmp_path / "demo_books"
    src.mkdir()
    _make_book(src, "alpha")

    dst = tmp_path / "books"
    dst.mkdir()
    _make_book(dst, "alpha")
    # User has overwritten the title to verify it's preserved.
    custom_meta = '{"book_id": "alpha", "title": "USER VERSION", ' \
                  '"language": "en", "total_pages": 1, "source_files": []}'
    (dst / "alpha" / "metadata.json").write_text(custom_meta)

    seeded = _seed_demo_books(src=src, dst=dst)

    assert seeded == 0
    assert "USER VERSION" in (dst / "alpha" / "metadata.json").read_text()


def test_returns_zero_when_src_is_missing(tmp_path):
    """Local development without baked demo books — seeding silently
    does nothing rather than crashing the app."""
    src = tmp_path / "missing"
    dst = tmp_path / "books"

    seeded = _seed_demo_books(src=src, dst=dst)

    assert seeded == 0
    assert not dst.exists()


def test_creates_dst_directory_if_needed(tmp_path):
    src = tmp_path / "demo_books"
    src.mkdir()
    _make_book(src, "alpha")

    dst = tmp_path / "deeply" / "nested" / "books"

    _seed_demo_books(src=src, dst=dst)

    assert dst.exists()
    assert (dst / "alpha" / "metadata.json").exists()


def test_skips_non_directory_entries_in_src(tmp_path):
    """Stray files in ``demo_books/`` (e.g. README.md) should be
    ignored — only directories represent books."""
    src = tmp_path / "demo_books"
    src.mkdir()
    _make_book(src, "alpha")
    (src / "README.md").write_text("just docs, not a book")

    dst = tmp_path / "books"

    seeded = _seed_demo_books(src=src, dst=dst)

    assert seeded == 1
    assert (dst / "alpha").exists()
    assert not (dst / "README.md").exists()
