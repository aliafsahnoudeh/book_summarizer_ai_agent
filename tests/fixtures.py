"""Synthetic book fixtures for integration and eval tests.

The Veridian Settlement is a fully invented topic — there is no such Mars
colony — so models cannot draw on training data. Every claim a summary
makes about it must be grounded in these pages, which makes faithfulness
and hallucination tests cleanly diagnosable.
"""

import json
from pathlib import Path

# Canonical 4-page synthetic book. Each page is a self-contained chapter
# so chunked tests at chunk_size=2 produce two distinct chunks.
TINY_BOOK_PAGES = [
    # Page 1
    """## Chapter 1: The Founding

In the year 2087, a consortium of international research institutes
established the Veridian Settlement on the southern slopes of Olympus
Mons. The mission was led by Director Selaine Tovari and Chief Engineer
Marcus Holm. The initial cohort numbered 47 personnel, drawn from the
fields of agronomy, materials science, and atmospheric chemistry.

The settlement's stated objective was to demonstrate self-sustaining
food production using only locally-sourced regolith and recycled water.
A secondary objective — rarely discussed publicly — was to test long-
duration psychological cohesion in small groups under permanent low-
gravity conditions.
""",
    # Page 2
    """## Chapter 2: Early Years

The first three years saw remarkable progress. By 2090, the agricultural
domes were producing 84% of the settlement's caloric needs. Holm's team
developed the now-famous "binding lattice" technique, which used heat-
treated regolith and a kelp-derived polymer to create stable growing
medium. The technique was patented and later licensed to seven other
off-world settlements.

However, tensions emerged between the agronomy and engineering teams
over water allocation. Director Tovari brokered the so-called Tovari
Compact in late 2091, which established a rotating priority system that
remained in force until the settlement's closure.
""",
    # Page 3
    """## Chapter 3: The Resonance Crisis

In the spring of 2094, an unexpected harmonic vibration — later traced
to thermal cycling in the binding lattice itself — caused a catastrophic
failure of the primary agricultural dome. The settlement lost 37% of its
food stockpile in 11 days.

Marcus Holm publicly took responsibility for the failure. He resigned
his post and returned to Earth in mid-2094. The settlement entered an
emergency rationing protocol that would last 14 months. Three personnel
were medically evacuated during this period; none died.
""",
    # Page 4
    """## Chapter 4: Recovery and Dissolution

Recovery began in earnest under Director Tovari's continued leadership.
By 2097, food production had returned to 95% of pre-crisis levels using
a redesigned dome architecture (the so-called "second-generation
lattice"). The settlement received its first new arrivals in five years
in 2099.

In 2103, citing budget constraints and the maturation of larger nearby
settlements, the founding consortium voted to dissolve the Veridian
Settlement. The final personnel departed on March 14th, 2103. The
binding lattice technique remains in use across twelve off-world
settlements as of this writing.
""",
]

TINY_BOOK_TOC = [
    {"title": "Chapter 1: The Founding", "level": 1, "page": 1},
    {"title": "Chapter 2: Early Years", "level": 1, "page": 2},
    {"title": "Chapter 3: The Resonance Crisis", "level": 1, "page": 3},
    {"title": "Chapter 4: Recovery and Dissolution", "level": 1, "page": 4},
]


def create_tiny_book(
    books_dir: Path,
    book_id: str = "veridian_settlement",
    num_pages: int | None = None,
) -> Path:
    """Materialise a synthetic ``.books/<book_id>/`` entry on disk.

    *num_pages* lets tests vary the size — when truncated, ``total_pages``
    in metadata.json reflects the actual page count, so state-invalidation
    paths in NoteTaker can be exercised.
    """
    pages = TINY_BOOK_PAGES if num_pages is None else TINY_BOOK_PAGES[:num_pages]
    if num_pages is not None and num_pages > len(TINY_BOOK_PAGES):
        # Pad by repeating the last page so callers can ask for arbitrary sizes.
        pages = pages + [TINY_BOOK_PAGES[-1]] * (num_pages - len(TINY_BOOK_PAGES))

    book_path = books_dir / book_id
    pages_dir = book_path / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    for i, text in enumerate(pages, start=1):
        (pages_dir / f"{i:03d}.txt").write_text(text, encoding="utf-8")

    metadata = {
        "book_id": book_id,
        "title": "The Veridian Settlement: A Brief History",
        "authors": ["Test Fixture"],
        "language": "en",
        "total_pages": len(pages),
        "source_files": ["fixture.pdf"],
        "pdf_page_ranges": [
            {"file": "fixture.pdf", "start": 1, "end": len(pages)}
        ],
    }
    (book_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    toc = [e for e in TINY_BOOK_TOC if e["page"] <= len(pages)]
    (book_path / "toc.json").write_text(
        json.dumps(toc, indent=2), encoding="utf-8"
    )

    return book_path
