"""Tests for ``book_summarizer.note_taker._iter_chunk_ranges``.

The chunk-iteration math is the part of the pipeline most likely to
silently lose pages — an off-by-one would not crash, just drop the last
chunk. This suite locks the page-coverage invariant: every page in
``1..total_pages`` must appear in exactly one chunk, in order, with no
gaps and no overlaps.
"""

import pytest

from book_summarizer.note_taker import _iter_chunk_ranges


# ── Known layouts: pin specific (total, size) → chunk shapes ──────────────


@pytest.mark.parametrize(
    "total_pages, chunk_size, expected",
    [
        # Smallest sensible cases.
        (1, 1, [(1, 1)]),
        (4, 1, [(1, 1), (2, 2), (3, 3), (4, 4)]),
        # Even division.
        (4, 2, [(1, 2), (3, 4)]),
        (6, 2, [(1, 2), (3, 4), (5, 6)]),
        (10, 5, [(1, 5), (6, 10)]),
        # Uneven division: last chunk is smaller.
        (4, 3, [(1, 3), (4, 4)]),
        (10, 3, [(1, 3), (4, 6), (7, 9), (10, 10)]),
        (10, 7, [(1, 7), (8, 10)]),
        # chunk_size >= total_pages → exactly one chunk.
        (4, 4, [(1, 4)]),
        (4, 5, [(1, 4)]),
        (4, 100, [(1, 4)]),
        # The smart defaults that ship today.
        (899, 4, None),  # too long to enumerate; checked via invariant
    ],
)
def test_chunk_ranges_known_layouts(total_pages, chunk_size, expected):
    chunks = list(_iter_chunk_ranges(total_pages, chunk_size))
    if expected is not None:
        assert chunks == expected


# ── Page-coverage invariant: every page exactly once, in order ────────────


@pytest.mark.parametrize(
    "total_pages, chunk_size",
    [
        (1, 1),
        (4, 1), (4, 2), (4, 3), (4, 4), (4, 5),
        (10, 1), (10, 3), (10, 7), (10, 11),
        (100, 5), (100, 7), (100, 100),
        (899, 4),  # the actual Cyropaedia × groq default
        (899, 20),
    ],
)
def test_every_page_covered_exactly_once(total_pages, chunk_size):
    chunks = list(_iter_chunk_ranges(total_pages, chunk_size))

    pages_seen: list[int] = []
    for start, end in chunks:
        assert 1 <= start <= end <= total_pages, (
            f"Chunk ({start}, {end}) out of bounds for {total_pages} pages"
        )
        pages_seen.extend(range(start, end + 1))

    assert pages_seen == list(range(1, total_pages + 1)), (
        f"Coverage gap or overlap for total={total_pages}, "
        f"size={chunk_size}: got {pages_seen}"
    )


def test_chunks_are_in_ascending_non_overlapping_order():
    chunks = list(_iter_chunk_ranges(50, 7))
    for prev, curr in zip(chunks, chunks[1:]):
        assert prev[1] + 1 == curr[0], (
            f"Gap or overlap between {prev} and {curr}"
        )


# ── Edge cases ────────────────────────────────────────────────────────────


def test_zero_or_negative_total_pages_yields_nothing():
    assert list(_iter_chunk_ranges(0, 5)) == []
    assert list(_iter_chunk_ranges(-1, 5)) == []


def test_zero_or_negative_chunk_size_yields_nothing():
    assert list(_iter_chunk_ranges(10, 0)) == []
    assert list(_iter_chunk_ranges(10, -3)) == []


# ── Smart-default math: chunk count matches what we ship ──────────────────


@pytest.mark.parametrize(
    "total_pages, chunk_size, expected_count",
    [
        (899, 4, 225),   # groq default after the lowering
        (899, 20, 45),   # cerebras qwen / gemini default
        (899, 5, 180),   # the OLD groq default; pin the count for history
        (1084, 20, 55),  # iranzamin
        (522, 20, 27),   # Quellen zur Geschichte des Partherreiches
        (653, 20, 33),   # The Archaeology of Iran
    ],
)
def test_chunk_count_for_real_world_book_sizes(
    total_pages, chunk_size, expected_count
):
    chunks = list(_iter_chunk_ranges(total_pages, chunk_size))
    assert len(chunks) == expected_count
