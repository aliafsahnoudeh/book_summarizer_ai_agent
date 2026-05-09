#!/usr/bin/env python3
"""Book Summarizer — CLI entry point.

Two-stage pipeline (translation has been removed for now — summaries
always come out in the book's original language):

  1. NoteTaker — comprehensive markdown notes, resumable.
  2. SummaryComposer — synthesise notes at the requested length.

Usage::

    uv run book-summarizer
    uv run book-summarizer --book the_archaelogy_of_iran
    uv run book-summarizer --fresh           # discard saved notes
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

from book_summarizer.book_summarizer import BookSummarizer
from book_builder import book_dir, list_books


def _pick_book(preselected: str | None) -> str:
    books = list_books()
    if not books:
        print(
            "No processed books found in .books/. "
            "Run `uv run build-book <path>` on a PDF first."
        )
        sys.exit(1)

    by_id = {b["id"]: b for b in books}
    if preselected:
        if preselected in by_id:
            return preselected
        print(f"No book with id '{preselected}'. Available:")
        preselected = None

    if len(books) == 1:
        b = books[0]
        print(f"Using the only available book: {b['title']} ({b['id']})")
        return b["id"]

    print("Available books:")
    for i, b in enumerate(books, start=1):
        print(
            f"  {i}. {b['title']}  [{b['id']}]  "
            f"({b['language']}, {b['total_pages']} pages)"
        )
    choice = input("\nSelect a book by number or id: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(books):
        return books[int(choice) - 1]["id"]
    if choice in by_id:
        return choice
    print("Invalid selection. Exiting.")
    sys.exit(1)


def _wipe_notes(book_id: str) -> None:
    notes = book_dir(book_id) / "notes"
    if not notes.exists():
        return
    for f in notes.iterdir():
        try:
            f.unlink()
        except OSError:
            pass
    try:
        notes.rmdir()
    except OSError:
        pass


def _render_progress(
    pages: int, total: int, chunks: int, elapsed_sec: float
) -> None:
    bar_width = 30
    pct = min(1.0, pages / max(total, 1))
    filled = int(bar_width * pct)
    bar = (
        "=" * bar_width
        if pct >= 1.0
        else "=" * filled + ">" + " " * (bar_width - filled - 1)
    )
    mins, secs = divmod(int(elapsed_sec), 60)
    line = (
        f"\r[{bar}] {pages}/{total} pages ({int(pct * 100):3d}%) · "
        f"{chunks} chunk(s) · {mins:02d}:{secs:02d}"
    )
    sys.stdout.write(line.ljust(90))
    sys.stdout.flush()


_BACKENDS = ("cerebras", "gemini", "groq")

# Default chunk size by (backend, optional cerebras-model).
# Picked to fit each backend's free-tier per-call and per-minute caps:
# • cerebras + llama3.1-8b: only 8 K total context, dense pages mean
#   ~4 pages max.
# • cerebras + qwen-3-235b: 65 K context, much smarter model, 5 RPM —
#   bigger chunks → fewer calls.
# • gemini (Flash): plenty of headroom; 20 is fine.
# • groq (qwen 32B free): tight 6 K TPM. 5 pages of dense academic
#   prose can exceed the cap (~1300 tokens/page × 5 = 6500 > 6000),
#   so we use 4 (~5200 tokens) for a comfortable margin. Auto-split
#   recovery in note_taker.py handles the rare overflow.
from book_summarizer.book_summarizer import (  # noqa: E402
    CEREBRAS_MODEL_LLAMA,
    CEREBRAS_MODEL_QWEN,
    CEREBRAS_MODELS,
)

_DEFAULT_CHUNK_SIZE_BY_PROFILE: dict[tuple[str, str | None], int] = {
    ("cerebras", CEREBRAS_MODEL_LLAMA): 4,
    ("cerebras", CEREBRAS_MODEL_QWEN): 20,
    ("gemini", None): 20,
    ("groq", None): 4,
}


def _resolve_backend_choice(
    cli_value: str | None,
    role_label: str,
    env_var: str,
    default: str = "groq",
) -> str:
    """Resolve a backend choice. Precedence:

        CLI flag  >  env var  >  interactive prompt  >  built-in default.

    When the env var is set the interactive prompt is skipped entirely —
    if the user has already configured a default they shouldn't be asked
    every run. Invalid env-var values fall through to the default with a
    short warning rather than blocking the run.
    """
    if cli_value:
        return cli_value

    env_value = os.getenv(env_var, "").strip().lower()
    if env_value:
        if env_value not in _BACKENDS:
            print(
                f"Unknown backend {env_value!r} from {env_var}; "
                f"defaulting to {default}."
            )
            return default
        print(f"{role_label} backend: {env_value} (from {env_var})")
        return env_value

    answer = input(
        f"\nLLM backend for {role_label}? "
        f"({'/'.join(_BACKENDS)}, Enter for {default})\n> "
    ).strip().lower()
    if not answer:
        return default
    if answer not in _BACKENDS:
        print(f"Unknown backend {answer!r}; defaulting to {default}.")
        return default
    return answer


def _resolve_cerebras_model(cli_value: str | None) -> str:
    """Resolve the Cerebras note-taker model.

    Precedence: CLI flag > ``BOOK_SUMMARIZER_CEREBRAS_MODEL`` > default.
    Invalid values fall through to the default with a warning.
    """
    env_var = "BOOK_SUMMARIZER_CEREBRAS_MODEL"
    if cli_value:
        model, source = cli_value, "CLI"
    else:
        env_value = os.getenv(env_var, "").strip()
        if env_value:
            model, source = env_value, env_var
        else:
            return CEREBRAS_MODEL_LLAMA

    if model not in CEREBRAS_MODELS:
        print(
            f"Unknown Cerebras model {model!r} (source: {source}); "
            f"defaulting to {CEREBRAS_MODEL_LLAMA}."
        )
        return CEREBRAS_MODEL_LLAMA
    if source == env_var:
        print(f"cerebras model: {model} (from {env_var})")
    return model


def _resolve_chunk_size(
    cli_value: int | None,
    note_taker_backend: str,
    cerebras_model: str,
) -> int:
    """Resolve chunk_size (pages per note-taking call).

    Precedence: CLI flag > ``BOOK_SUMMARIZER_CHUNK_SIZE`` > smart
    per-(backend, model) default. The smart default still respects the
    backend's free-tier per-call and per-minute caps; an explicit env-var
    override is honoured even if it's a poor fit (the user asked for it).
    """
    if cli_value is not None:
        return max(1, cli_value)

    env_var = "BOOK_SUMMARIZER_CHUNK_SIZE"
    env_value = os.getenv(env_var, "").strip()
    if env_value:
        try:
            value = int(env_value)
            if value < 1:
                raise ValueError("chunk_size must be a positive integer")
            print(f"chunk size: {value} (from {env_var})")
            return value
        except ValueError:
            print(
                f"Invalid {env_var}={env_value!r}; must be a positive "
                "integer. Falling back to the per-backend default."
            )

    profile_key: tuple[str, str | None]
    if note_taker_backend == "cerebras":
        profile_key = ("cerebras", cerebras_model)
    else:
        profile_key = (note_taker_backend, None)
    return _DEFAULT_CHUNK_SIZE_BY_PROFILE.get(profile_key, 10)


def _run_with_progress(
    summarizer: BookSummarizer,
    user_input: str,
    chunk_size: int,
) -> Path | None:
    if not sys.stdout.isatty():
        return summarizer.run(user_input, chunk_size=chunk_size)

    result: dict = {}

    def worker():
        try:
            result["path"] = summarizer.run(user_input, chunk_size=chunk_size)
        except BaseException as e:
            result["error"] = e

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    start = time.time()
    while t.is_alive():
        if summarizer.total_pages > 0 and summarizer.pages_read > 0:
            _render_progress(
                summarizer.pages_read,
                summarizer.total_pages,
                summarizer.chunks_saved,
                time.time() - start,
            )
        time.sleep(0.3)
    t.join()

    if summarizer.total_pages > 0 and summarizer.pages_read > 0:
        _render_progress(
            summarizer.pages_read,
            summarizer.total_pages,
            summarizer.chunks_saved,
            time.time() - start,
        )
        sys.stdout.write("\n")
        sys.stdout.flush()

    if "error" in result:
        raise result["error"]
    return result.get("path")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Summarise a pre-processed book."
    )
    parser.add_argument(
        "--book", help="Book id (directory name under .books/)"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Discard saved notes for this book and start clean.",
    )
    parser.add_argument(
        "--note-taker-backend",
        choices=_BACKENDS,
        help=(
            "LLM backend for the note-taking agent (default: cerebras). "
            f"Free options: {', '.join(_BACKENDS)}."
        ),
    )
    parser.add_argument(
        "--composer-backend",
        choices=_BACKENDS,
        help=(
            "LLM backend for the summary composer (default: gemini). "
            "Gemini 2.5 Flash has the largest free-tier context window "
            "and works reliably for one-shot composition."
        ),
    )
    parser.add_argument(
        "--cerebras-model",
        choices=CEREBRAS_MODELS,
        help=(
            "Cerebras model for the note-taker "
            f"(default: {CEREBRAS_MODEL_LLAMA}). "
            f"{CEREBRAS_MODEL_LLAMA}: small/fast/Production, 8 K ctx. "
            f"{CEREBRAS_MODEL_QWEN}: 235 B / 65 K ctx / Preview tier."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "Pages per note-taking chunk. Defaults adapt to the "
            "(backend, cerebras-model) pair: "
            "cerebras+llama=4, cerebras+qwen=20, gemini=20, groq=4."
        ),
    )
    args = parser.parse_args()

    book_id = _pick_book(args.book)

    if args.fresh:
        _wipe_notes(book_id)
        print("(--fresh: deleted prior notes)")

    print("=" * 60)
    print("Book Summarizer AI Agent")
    print("=" * 60)
    print()
    print("Summary length options:")
    print("  • very short    — one or two paragraphs")
    print("  • short         — ~5% of the book")
    print("  • medium        — ~10-15% of the book")
    print("  • comprehensive — full detail, no fluff")
    print()
    print("Pipeline: 1) NoteTaker (resumable) → 2) Compose "
          "(chunked if needed). Summary stays in the book's language.")
    print("-" * 60)

    user_input = input(
        "\nHow would you like the book summarised?\n"
        "(e.g. 'short', 'comprehensive', 'very short', ...)\n> "
    )
    if not user_input.strip():
        print("No input provided. Exiting.")
        return

    note_taker_backend = _resolve_backend_choice(
        args.note_taker_backend,
        "note-taking",
        env_var="BOOK_SUMMARIZER_NOTE_TAKER_BACKEND",
        default="cerebras",
    )
    # Composer defaults to Gemini 2.5 Flash — its 1 M context fits the
    # entire notes corpus for any reasonable book in a single call,
    # and the free tier is much more reliable than Cerebras Preview
    # qwen for sustained workloads.
    composer_backend = _resolve_backend_choice(
        args.composer_backend,
        "summary composition",
        env_var="BOOK_SUMMARIZER_COMPOSER_BACKEND",
        default="gemini",
    )
    cerebras_model = _resolve_cerebras_model(args.cerebras_model)
    chunk_size = _resolve_chunk_size(
        args.chunk_size, note_taker_backend, cerebras_model
    )

    print("\n" + "=" * 60)
    note_taker_label = (
        f"{note_taker_backend}/{cerebras_model}"
        if note_taker_backend == "cerebras"
        else note_taker_backend
    )
    if composer_backend == "cerebras":
        composer_label = f"{composer_backend}/qwen-3-235b"
    elif composer_backend == "gemini":
        composer_label = f"{composer_backend}/2.5-flash"
    else:
        composer_label = composer_backend
    print(
        f"Processing — note_taker={note_taker_label}, "
        f"composer={composer_label}"
    )
    print(
        f"Chunk size: {chunk_size} pages per note-taking call "
        f"({'CLI flag' if args.chunk_size is not None else 'backend default'})."
        " Note-taking can take a while for large books — Ctrl+C is safe; "
        "progress is saved per chunk."
    )
    print("=" * 60 + "\n")

    summarizer = BookSummarizer(
        book_id=book_id,
        note_taker_backend=note_taker_backend,
        composer_backend=composer_backend,
        cerebras_model=cerebras_model,
    )
    out_path = _run_with_progress(
        summarizer,
        user_input.strip(),
        chunk_size=chunk_size,
    )

    if out_path:
        print(f"\nSummary saved to: {out_path}")
        try:
            print("\n" + "=" * 60)
            print("Summary")
            print("=" * 60)
            print(out_path.read_text(encoding="utf-8"))
        except OSError:
            pass
    else:
        print("\n(No summary was produced.)")


if __name__ == "__main__":
    main()
