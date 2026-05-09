"""Eval-suite fixtures.

The eval suite runs **real LLM calls** against the synthetic Veridian
Settlement fixture. It is skipped automatically when an API key required
by the chosen backends is missing, so CI without secrets and contributors
without keys still see a clean test run for the rest of the suite.

Default invocation skips eval tests entirely; opt in with:

    uv run pytest -m eval

Backend selection (env vars override the defaults; same vars drive the
production CLI so `book-summarizer` and `pytest -m eval` share defaults):

    BOOK_SUMMARIZER_NOTE_TAKER_BACKEND   default ``groq``    (cerebras / gemini / groq)
    BOOK_SUMMARIZER_COMPOSER_BACKEND     default ``gemini``  (cerebras / gemini / groq)

The judge is always Gemini 2.5 Flash (so ``GOOGLE_API_KEY`` is always
required).

Cost shape with the defaults (~4 LLM calls per session, free tier):
  - 2 Groq note-taker calls  (4-page book at chunk_size=2; 30 RPM)
  - 1 Gemini composer call   (single-call path; corpus << safe_input)
  - 1 Gemini judge call      (LLM-as-judge faithfulness)
"""

import os

import pytest
from dotenv import load_dotenv

from tests.fixtures import create_tiny_book


# Load the project's .env so contributors don't have to export keys
# manually before running the eval suite.
load_dotenv()


# Maps every backend to the env var that holds its API key.
_BACKEND_API_KEY_ENV = {
    "cerebras": "CEREBRAS_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
}


def _eval_backends() -> dict:
    """Backends to run the eval pipeline against, with env-var overrides.

    Defaults to ``note_taker=groq, composer=gemini`` — small/fast/predictable
    note-taking on Groq's 30 RPM free tier paired with Gemini 2.5 Flash's
    1 M context window for composition. Override via the same env vars
    the CLI reads, so eval and production share defaults.
    """
    return {
        "note_taker": os.getenv("BOOK_SUMMARIZER_NOTE_TAKER_BACKEND", "groq").lower(),
        "composer": os.getenv("BOOK_SUMMARIZER_COMPOSER_BACKEND", "gemini").lower(),
    }


@pytest.fixture(scope="session")
def required_api_keys() -> dict:
    """Skip eval cleanly when any key required by the chosen backends is missing.

    Keys needed:
      - One per chosen backend (note-taker + composer; may overlap).
      - GOOGLE_API_KEY (always — the judge is Gemini Flash).
    """
    backends = _eval_backends()
    needed = {
        _BACKEND_API_KEY_ENV[backends["note_taker"]],
        _BACKEND_API_KEY_ENV[backends["composer"]],
        "GOOGLE_API_KEY",  # judge
    }
    missing = sorted(env for env in needed if not os.getenv(env))
    if missing:
        pytest.skip(
            "Eval suite skipped — missing API keys: "
            + ", ".join(missing)
            + f". Backends in use: note_taker={backends['note_taker']}, "
            f"composer={backends['composer']}, judge=gemini. "
            "Set the required keys in .env or export them, then re-run "
            "`pytest -m eval`."
        )
    return {env: os.getenv(env) for env in needed}


@pytest.fixture(scope="session")
def eval_session_dirs(tmp_path_factory):
    """Redirect ``BOOKS_DIR`` to a session-scoped temp directory.

    Summaries now live at ``.books/<book_id>/summaries/<level>.md`` —
    co-located with the book and its notes — so redirecting ``BOOKS_DIR``
    automatically captures both notes AND summaries inside the temp tree.
    No separate output redirect needed.

    pytest's built-in ``monkeypatch`` is function-scoped, so we do this
    manually with explicit save/restore to keep the patch alive for the
    whole eval session.
    """
    base = tmp_path_factory.mktemp("eval")
    fake_books = base / ".books"
    fake_books.mkdir()

    import book_builder.paths
    import book_builder.reader

    saved = {
        "paths_BOOKS_DIR": book_builder.paths.BOOKS_DIR,
        "reader_BOOKS_DIR": book_builder.reader.BOOKS_DIR,
    }
    book_builder.paths.BOOKS_DIR = fake_books
    book_builder.reader.BOOKS_DIR = fake_books

    yield {"books_dir": fake_books}

    book_builder.paths.BOOKS_DIR = saved["paths_BOOKS_DIR"]
    book_builder.reader.BOOKS_DIR = saved["reader_BOOKS_DIR"]


@pytest.fixture(scope="session")
def short_summary_run(required_api_keys, eval_session_dirs) -> dict:
    """Run the full pipeline once per session and return the artifacts.

    Returns a dict with:
        ``summary``    — the rendered summary text.
        ``notes``      — concatenated chunk-note markdown (the judge
                         scores against this — not the raw page text —
                         because the composer only sees notes).
        ``out_path``   — Path to the summary file.
        ``book_id``    — fixture book id.
        ``backends``   — backends actually used (after env-var resolution).
    """
    from pathlib import Path

    from book_summarizer.book_summarizer import BookSummarizer

    book_id = "veridian_settlement"
    create_tiny_book(eval_session_dirs["books_dir"], book_id=book_id, num_pages=4)

    backends = _eval_backends()
    summarizer = BookSummarizer(
        book_id=book_id,
        note_taker_backend=backends["note_taker"],
        composer_backend=backends["composer"],
    )
    out_path: Path = summarizer.run("short", chunk_size=2)

    summary = out_path.read_text(encoding="utf-8")

    # Concatenate the per-chunk notes — the composer's actual input —
    # so the LLM-as-judge scores faithfulness against what the composer
    # had access to, not against the raw page text.
    notes_dir = eval_session_dirs["books_dir"] / book_id / "notes"
    notes_files = sorted(p for p in notes_dir.glob("*.md"))
    notes = "\n\n".join(p.read_text(encoding="utf-8") for p in notes_files)

    return {
        "summary": summary,
        "notes": notes,
        "out_path": out_path,
        "book_id": book_id,
        "backends": backends,
    }
