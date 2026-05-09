"""Book Summariser orchestrator.

A three-stage pipeline (translation has been removed for now — summaries
stay in the book's original language):

  1. **BookExplorer** — a GAME-loop agent inspects the book's metadata
     and table of contents and emits a structured **reading plan**:
     a partition of pages into 'skip', 'skim', and 'read' ranges.
     Saved to ``.books/{book_id}/reading_plan.json``. Cached across
     subsequent runs (re-runs only when the plan is missing or stale).

  2. **NoteTaker** — a Python loop walks the book in fixed-size chunks
     and calls a per-chunk ``ChunkNoteWriter`` (single LLM call) that
     receives only that chunk's text. No accumulated memory between
     chunks. Chunks whose pages are entirely 'skip' in the reading
     plan are not sent to the LLM. Notes are saved to
     ``.books/{book_id}/notes/NNN-MMM.md``, in the book's original
     language. Resumable across runs via ``_state.json``.

  3. **SummaryComposer** (single LLM call) — synthesises every chunk's
     notes into a single Markdown summary at the requested length
     (``very short`` / ``short`` / ``medium`` / ``comprehensive``),
     in the book's original language.

Output: ``.books/<book_id>/summaries/<level>.md``.
"""

import os
from pathlib import Path

from book_summarizer.book_explorer import BookExplorer
from book_summarizer.note_taker import ChunkNoteWriter, NoteTaker
from book_summarizer.summary_composer import SummaryComposer
from zurvan import LogLevel, TokenTracker
from zurvan.logger.local_text_file_logger import LocalTextFileLogger
from book_builder import BookReader, book_dir
from book_builder.paths import PROJECT_ROOT


_LEVELS = ("very short", "short", "medium", "comprehensive")

_DEFAULT_CHUNK_SIZE = 10

# ── Backend / model selection ─────────────────────────────────────────────
#
# Two roles each pick their own backend:
#   * ``note_taker`` — the per-chunk note writer.
#   * ``composer``   — the final summary synthesiser.
# Translation always uses Gemini Flash and is not exposed here.

# Cerebras's free tier exposes two models with very different shapes,
# so the user can pick which one to use via ``--cerebras-model``.
CEREBRAS_MODEL_LLAMA = "llama3.1-8b"
CEREBRAS_MODEL_QWEN = "qwen-3-235b-a22b-instruct-2507"
CEREBRAS_MODELS = (CEREBRAS_MODEL_LLAMA, CEREBRAS_MODEL_QWEN)

# Per-(backend, model) profile: model string, output budget, TPM/RPM
# caps for the pre-emptive pacer, and a "safe input budget" used by
# the composer to decide when to switch into chunked-composition mode.
# Numbers reflect the providers' published free-tier limits at the
# time of writing — adjust if a provider re-tiers.
_PROFILES: dict[tuple[str, str | None], dict] = {
    ("cerebras", CEREBRAS_MODEL_LLAMA): {
        "model": f"cerebras/{CEREBRAS_MODEL_LLAMA}",
        "context_window": 8_192,
        "tpm": 60_000,
        "rpm": 30,
        "note_taker_max_tokens": 1024,
        "composer_max_tokens": 1024,
        "composer_safe_input": 6_000,
    },
    ("cerebras", CEREBRAS_MODEL_QWEN): {
        "model": f"cerebras/{CEREBRAS_MODEL_QWEN}",
        "context_window": 65_536,
        "tpm": 30_000,
        "rpm": 5,
        "note_taker_max_tokens": 2_048,
        "composer_max_tokens": 4_096,
        # Stay well under both the 30 K TPM and 65 K context to leave
        # room for output. Composer falls back to chunked composition
        # when the notes corpus exceeds this.
        "composer_safe_input": 22_000,
    },
    # Gemini 2.5 Flash works on the user's free-tier AI Studio key
    # where 2.0-flash currently 429s. Flash 2.5 spends invisible
    # "thinking" tokens against ``max_tokens``, so the budgets here
    # are ~2× the visible-output budget you'd want for a model
    # without thinking tokens.
    ("gemini", None): {
        "model": "gemini/gemini-2.5-flash",
        "context_window": 1_000_000,
        "tpm": None,
        "rpm": 10,  # 2.5 Flash free-tier RPM
        "note_taker_max_tokens": 8_192,
        # 32 K leaves room for substantial thinking tokens on
        # comprehensive summaries of long books — empirically ~24 K
        # was getting truncated mid-sentence on a 650-page book.
        "composer_max_tokens": 32_768,
        "composer_safe_input": 800_000,
    },
    ("groq", None): {
        "model": "groq/qwen/qwen3-32b",
        "context_window": 32_768,
        "tpm": 6_000,
        "rpm": 30,
        "note_taker_max_tokens": 1_500,
        "composer_max_tokens": 1_500,
        "composer_safe_input": 4_000,
    },
}

_BACKEND_CONFIG: dict[str, dict] = {
    "cerebras": {"api_key_env": "CEREBRAS_API_KEY"},
    "gemini": {"api_key_env": "GOOGLE_API_KEY"},
    "groq": {"api_key_env": "GROQ_API_KEY"},
}


def _profile(backend: str, cerebras_model: str | None) -> dict:
    """Return the profile dict for a (backend, optional model) pair."""
    if backend == "cerebras":
        model = cerebras_model or CEREBRAS_MODEL_LLAMA
        if model not in CEREBRAS_MODELS:
            raise ValueError(
                f"Unknown Cerebras model {model!r}. "
                f"Valid: {CEREBRAS_MODELS}"
            )
        return _PROFILES[("cerebras", model)]
    return _PROFILES[(backend, None)]


_DEFAULT_BACKENDS = {"note_taker": "cerebras", "composer": "gemini"}
_DEFAULT_COMPOSER_CEREBRAS_MODEL = CEREBRAS_MODEL_QWEN


def _resolve_backend(name: str | None, role: str = "note_taker") -> str:
    """Validate a backend name. Defaults are role-specific."""
    backend = (name or _DEFAULT_BACKENDS.get(role, "groq")).lower()
    if backend not in _BACKEND_CONFIG:
        raise ValueError(
            f"Unknown LLM backend {backend!r}. "
            f"Valid options: {sorted(_BACKEND_CONFIG)}"
        )
    return backend


# ISO 639-1 → human-readable name. Used so translation prompts and
# language-comparison logic work with either codes ("en") or names
# ("English") interchangeably. Unknown codes fall through unchanged.
_LANG_NAMES = {
    "en": "English",
    "fa": "Persian",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt": "Portuguese",
    "ar": "Arabic",
    "tr": "Turkish",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "sv": "Swedish",
    "pl": "Polish",
    "hi": "Hindi",
}


def _detect_level(text: str) -> str:
    text = text.lower()
    if "very short" in text:
        return "very short"
    if "comprehensive" in text:
        return "comprehensive"
    if "medium" in text:
        return "medium"
    if "short" in text:
        return "short"
    return "short"


def _slug(name: str) -> str:
    """Map a level like 'very short' to 'very_short' for filenames."""
    return name.replace(" ", "_")


def _language_name(code_or_name: str) -> str:
    if not code_or_name:
        return "the source language"
    return _LANG_NAMES.get(code_or_name.lower(), code_or_name)


class BookSummarizer:
    """Top-level orchestrator for the book summarisation pipeline.

    Usage::

        summarizer = BookSummarizer(
            book_id="the_archaelogy_of_iran",
            note_taker_backend="groq",     # default
            composer_backend="gemini",     # default
        )
        out_path = summarizer.run("comprehensive")
    """

    def __init__(
        self,
        book_id: str,
        note_taker_backend: str | None = None,
        composer_backend: str | None = None,
        cerebras_model: str | None = None,
        logger=None,
    ):
        """
        Args:
            logger: Optional override for the default file logger. The
                web UI passes a ``MultiLogger`` here to tee normal file
                logging to a queue read by the live-activity browser
                stream. ``None`` (the default — i.e. CLI use) keeps
                writing to ``<project_root>/.logs/``.
        """
        self._book_id = book_id
        self._reader = BookReader(book_id)
        if not self._reader.exists():
            raise FileNotFoundError(
                f"Book '{book_id}' not found. Run `build-book` first."
            )

        # Logs go to ``<project_root>/.logs/`` — keeps runtime state out
        # of the package directory so a future ``pip install`` of this
        # tool doesn't try to write inside read-only ``site-packages/``.
        self._logger = logger or LocalTextFileLogger(
            "book_summarizer_agent.log",
            level=LogLevel.DEBUG,
            path=str(PROJECT_ROOT),
        )
        self._token_tracker = TokenTracker()
        self._metadata = self._reader.metadata()
        self._book_language_name = _language_name(
            self._metadata.get("language", "")
        )

        self._backends = {
            "note_taker": _resolve_backend(note_taker_backend, "note_taker"),
            "composer": _resolve_backend(composer_backend, "composer"),
        }
        # Cerebras has two free-tier models with very different shapes.
        # ``cerebras_model`` only affects the note-taker; the composer
        # always picks qwen on Cerebras (its 65 K context dwarfs llama's
        # 8 K and is essential for fitting a long book's notes).
        self._cerebras_note_taker_model = cerebras_model or CEREBRAS_MODEL_LLAMA
        if self._cerebras_note_taker_model not in CEREBRAS_MODELS:
            raise ValueError(
                f"Unknown Cerebras model {self._cerebras_note_taker_model!r}. "
                f"Valid: {CEREBRAS_MODELS}"
            )

        # Warn once per unique backend whose API key is missing. Keys are
        # checked but never logged — only their presence/absence.
        for backend in set(self._backends.values()):
            env_key = _BACKEND_CONFIG[backend]["api_key_env"]
            if not os.getenv(env_key):
                self._logger.log(
                    f"{env_key} is not set — {backend} LLM calls will fail.",
                    level=LogLevel.WARNING,
                    env=None,
                )

        # Lazily constructed during run() so the CLI progress bar can
        # observe the live state.
        self._note_taker: NoteTaker | None = None

    def _log_run_header(self, level: str, chunk_size: int) -> None:
        """Emit a single structured 'run start' block at the top of the log.

        Captures every input that affects what gets produced (book, level,
        chunk size, both backends + models, key presence) so any log file
        is reproducible without cross-referencing other context. Keys are
        reported as 'set' / 'MISSING' — never logged in plaintext.
        """
        keys_to_report = sorted(
            {_BACKEND_CONFIG[b]["api_key_env"] for b in self._backends.values()}
        )

        bar = "═" * 70
        lines = [
            "",
            bar,
            "BookSummarizer run",
            bar,
            f"  book_id            : {self._book_id}",
            f"  title              : {self._metadata.get('title', '?')}",
            f"  total_pages        : {self._metadata.get('total_pages', '?')}",
            f"  language           : {self._book_language_name}",
            f"  level              : {level}",
            f"  chunk_size         : {chunk_size}",
            "",
            f"  note_taker backend : {self._backends['note_taker']} "
            f"({self._profile_for('note_taker')['model']})",
            f"  composer backend   : {self._backends['composer']} "
            f"({self._profile_for('composer')['model']})",
            "",
            "  API keys           : "
            + ", ".join(
                f"{env}={'set' if os.getenv(env) else 'MISSING'}"
                for env in keys_to_report
            ),
            bar,
        ]
        for line in lines:
            self._logger.log(line, level=LogLevel.INFO, env=None)

    def _profile_for(self, role: str) -> dict:
        """Profile dict for the role's chosen (backend, model)."""
        backend = self._backends[role]
        if backend == "cerebras":
            # Note-taker uses the user-selected Cerebras model; composer
            # always uses qwen (only model with enough context).
            model = (
                self._cerebras_note_taker_model
                if role == "note_taker"
                else _DEFAULT_COMPOSER_CEREBRAS_MODEL
            )
            return _PROFILES[("cerebras", model)]
        return _PROFILES[(backend, None)]

    def _build_chunk_note_writer(self) -> ChunkNoteWriter:
        backend = self._backends["note_taker"]
        prof = self._profile_for("note_taker")
        return ChunkNoteWriter(
            model=prof["model"],
            api_key_env=_BACKEND_CONFIG[backend]["api_key_env"],
            max_tokens=prof["note_taker_max_tokens"],
            logger=self._logger,
            token_tracker=self._token_tracker,
            tpm_limit=prof["tpm"],
            rpm_limit=prof["rpm"],
        )

    # ── Public read-only state for progress reporting ──────────────────────

    @property
    def total_pages(self) -> int:
        return self._metadata.get("total_pages", 0)

    @property
    def pages_read(self) -> int:
        return len(self._note_taker.pages_read) if self._note_taker else 0

    @property
    def chunks_saved(self) -> int:
        return len(self._note_taker.chunks) if self._note_taker else 0

    # ── Pipeline ────────────────────────────────────────────────────────────

    def run(
        self,
        user_input: str,
        chunk_size: int = _DEFAULT_CHUNK_SIZE,
    ) -> Path:
        """Execute the three-stage pipeline (BookExplorer → NoteTaker →
        SummaryComposer) and return the output file path. Summaries are
        produced in the book's original language; translation has been
        removed for now.
        """
        level = _detect_level(user_input)
        self._log_run_header(level, chunk_size)
        # Co-locate summaries with the book they came from:
        # ``.books/<book_id>/summaries/<level>.md`` mirrors the existing
        # ``notes/`` layout and means ``rm -rf .books/<book_id>/`` wipes
        # every artefact derived from that book in one step.
        out_dir = book_dir(self._book_id) / "summaries"
        out_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. BookExplorer: produce / load the reading plan ────────────
        # Always uses Gemini 2.5 Flash (1 M context, generous free RPM).
        # Cached across runs; falls back to all-read if anything goes
        # wrong, so the pipeline is never blocked by a flaky agent call.
        self._logger.log(
            f"Step 1/3: BookExplorer — building reading plan for "
            f"'{self._book_id}'.",
            level=LogLevel.INFO,
            env=None,
        )
        explorer = BookExplorer(
            book_id=self._book_id,
            logger=self._logger,
            token_tracker=self._token_tracker,
        )
        reading_plan = explorer.explore()
        n_skip = sum(
            1 for e in reading_plan.entries if e.importance == "skip"
        )
        n_skim = sum(
            1 for e in reading_plan.entries if e.importance == "skim"
        )
        n_read = sum(
            1 for e in reading_plan.entries if e.importance == "read"
        )
        self._logger.log(
            f"Reading plan: {len(reading_plan.entries)} entries "
            f"(skip={n_skip}, skim={n_skim}, read={n_read}); "
            f"{len(reading_plan.pages_to_skip())}/{self.total_pages} "
            "pages will be skipped by the note-taker.",
            level=LogLevel.INFO,
            env=None,
        )

        # ── 2. Note-taking (resumable) ──────────────────────────────────
        self._note_taker = NoteTaker(
            book_id=self._book_id,
            note_writer=self._build_chunk_note_writer(),
            logger=self._logger,
            token_tracker=self._token_tracker,
            chunk_size=chunk_size,
            reading_plan=reading_plan,
        )
        if not self._note_taker.is_completed:
            self._logger.log(
                f"Step 2/3: NoteTaker — book={self._book_id} "
                f"already_covered={len(self._note_taker.pages_read)}/"
                f"{self.total_pages} "
                f"chunk_size={chunk_size}",
                level=LogLevel.INFO,
                env=None,
            )
            self._note_taker.run()
        else:
            self._logger.log(
                "Step 2/3: NoteTaker — already complete, skipping.",
                level=LogLevel.INFO,
                env=None,
            )

        # If note-taking failed on the very first chunk, ``notes_dir``
        # holds no ``.md`` files and there is nothing for the composer
        # to work with. Surface this clearly with actionable guidance
        # (the underlying error — e.g. Groq TPM-too-large — is in the
        # log).
        if not self._note_taker.chunks:
            note_backend = self._backends["note_taker"]
            hint = (
                f"--chunk-size {max(1, chunk_size // 2)} "
                f"(or smaller)"
                if note_backend == "groq"
                else "--note-taker-backend gemini"
            )
            raise RuntimeError(
                "Note-taking produced no notes — every chunk failed. "
                f"See log for the underlying error. Try `{hint}` "
                "and re-run."
            )

        # ── 3. Compose summary in the book's original language ──────────
        self._logger.log(
            f"Step 3/3: SummaryComposer — level={level}, "
            f"language={self._book_language_name}",
            level=LogLevel.INFO,
            env=None,
        )
        composer_backend = self._backends["composer"]
        composer_profile = self._profile_for("composer")
        composer = SummaryComposer(
            notes_dir=self._note_taker.notes_dir,
            book_metadata=self._metadata,
            logger=self._logger,
            token_tracker=self._token_tracker,
            model=composer_profile["model"],
            api_key_env=_BACKEND_CONFIG[composer_backend]["api_key_env"],
            max_tokens=composer_profile["composer_max_tokens"],
            safe_input_tokens=composer_profile["composer_safe_input"],
            tpm_limit=composer_profile["tpm"],
            rpm_limit=composer_profile["rpm"],
            book_id=self._book_id,
        )
        summary = composer.compose(level)

        out_file = out_dir / f"{_slug(level)}.md"
        out_file.write_text(summary, encoding="utf-8")

        self._logger.log(
            self._token_tracker.report(),
            level=LogLevel.INFO,
            env=None,
        )

        return out_file


__all__ = ["BookSummarizer"]
