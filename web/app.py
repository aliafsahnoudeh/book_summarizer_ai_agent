"""Gradio app — public interactive demo of the Book Summarizer pipeline.

Run locally::

    uv run python -m web.app

Run in Docker (production)::

    docker run -p 7860:7860 \\
        -e DEMO_PASSWORD=<pwd> \\
        -e GOOGLE_API_KEY=... \\
        -e GROQ_API_KEY=... \\
        ghcr.io/aliafsahnoudeh/book_summarizer_ai_agent:latest

The UI is a single Gradio Blocks page. The full pipeline runs server-side
on a background thread; structured log lines stream live to the
"Activity log" textbox via a queue (see ``web/web_ui_logger.py``). When
the pipeline finishes the final summary appears in the Markdown panel
and the token report below it.

Auth: single shared password (``DEMO_PASSWORD`` env var). When unset
(local dev only), the app runs ungated.

Cost: ``DailyRunBudget`` (``DEMO_DAILY_RUN_LIMIT`` env var, default 20)
caps full-pipeline runs per UTC day so a leaked password can't drain
the LLM provider's free-tier quota.
"""

import os
import queue
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

# Load .env for local dev — production uses HF Space secrets directly.
load_dotenv()

from book_builder import BookReader, build_book, list_books
from book_builder.paths import BOOKS_DIR, PROJECT_ROOT
from book_summarizer.book_summarizer import BookSummarizer
from book_summarizer.main import _resolve_cerebras_model, _resolve_chunk_size
from zurvan import LogLevel
from zurvan.logger.local_text_file_logger import LocalTextFileLogger
from web.budget import DailyRunBudget
from web.web_ui_logger import MultiLogger, WebUILogger


_LEVELS = ("very short", "short", "medium", "comprehensive")
_LOG_TAIL_LINES = 100
_POLL_INTERVAL_S = 0.3

# Smart auto-scroll for the live activity-log textarea.
#
# Behaviour: when a new log line arrives, scroll to the bottom if the
# user was already near the bottom — otherwise leave their scroll
# position alone (so they can scroll up to read older lines without
# being yanked back). When the user scrolls back to the bottom on
# their own, auto-scrolling resumes.
#
# Gradio's gr.Textbox doesn't ship this behaviour. We inject a vanilla
# JS via ``gr.Blocks(head=...)`` so it runs reliably on page load.
# The MutationObserver re-attaches if Gradio re-renders the component
# (which happens when the value bounces between empty and populated).
_AUTOSCROLL_HEAD_HTML = """
<script>
(() => {
    const TOLERANCE = 30;       // pixels — "near bottom" threshold
    const POLL_MS = 150;        // how often to check for new content

    const setupAutoscroll = (el) => {
        if (el.dataset.autoscrollSetup) return;
        el.dataset.autoscrollSetup = '1';
        console.log('[autoscroll] attached to', el.tagName, el);

        let wasNearBottom = true;
        let isProgrammatic = false;
        let lastValue = el.value !== undefined ? el.value : el.textContent;

        const currentValue = () =>
            el.value !== undefined ? el.value : el.textContent;

        const isNearBottom = () =>
            el.scrollTop + el.clientHeight >= el.scrollHeight - TOLERANCE;

        el.addEventListener('scroll', () => {
            if (!isProgrammatic) wasNearBottom = isNearBottom();
        });

        setInterval(() => {
            const v = currentValue();
            if (v === lastValue) return;
            lastValue = v;
            if (wasNearBottom) {
                isProgrammatic = true;
                el.scrollTop = el.scrollHeight;
                setTimeout(() => { isProgrammatic = false; }, 0);
            }
        }, POLL_MS);
    };

    const tryAttach = () => {
        const wrapper = document.getElementById('activity-log');
        if (!wrapper) return;
        // Gradio's interactive=False Textbox usually renders a textarea
        // with readonly, but versions vary — fall back to any
        // overflow-scrollable descendant if no textarea exists.
        const target =
            wrapper.querySelector('textarea')
            || wrapper.querySelector('[class*="scroll"]')
            || wrapper;
        if (target && !target.dataset.autoscrollSetup) {
            setupAutoscroll(target);
        }
    };

    const start = () => {
        console.log('[autoscroll] script ready');
        tryAttach();
        new MutationObserver(tryAttach).observe(document.body, {
            childList: true,
            subtree: true,
        });
    };

    if (document.body) start();
    else document.addEventListener('DOMContentLoaded', start);
})();
</script>
"""

# Two verbosity modes. "Verbose" is the default because the agent's
# internal trace (BookExplorer's tool calls, NoteTaker's per-chunk
# output, the canary being injected and not leaked) is the most
# compelling part of the demo to watch.
_VERBOSITY_VERBOSE = "Verbose (debug)"
_VERBOSITY_STANDARD = "Standard (info)"
_VERBOSITY_TO_LEVEL = {
    _VERBOSITY_VERBOSE: LogLevel.DEBUG,
    _VERBOSITY_STANDARD: LogLevel.INFO,
}

_BUDGET = DailyRunBudget(
    max_runs_per_day=int(os.getenv("DEMO_DAILY_RUN_LIMIT", "20"))
)


# ── Demo-book seeding ─────────────────────────────────────────────────────
#
# HF Spaces' free tier wipes runtime disk on every restart, but the
# Docker image's layers are permanent. We bake demo books into the
# image at ``demo_books/<id>/`` and copy them into the writable
# ``.books/`` on app start so the picker is always populated, even on
# a fresh container.

_DEMO_BOOKS_DIR = PROJECT_ROOT / "demo_books"


def _seed_demo_books(
    src: Path | None = None,
    dst: Path | None = None,
) -> int:
    """Copy any books in ``demo_books/`` into ``.books/`` if not present.

    Idempotent: skips books that are already in ``.books/`` (so a
    second invocation, or a user upload that happens to share an id,
    won't clobber existing state). Returns the number of books seeded.
    """
    src = src if src is not None else _DEMO_BOOKS_DIR
    dst = dst if dst is not None else BOOKS_DIR
    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    seeded = 0
    for book in sorted(src.iterdir()):
        if not book.is_dir():
            continue
        target = dst / book.name
        if target.exists():
            continue
        shutil.copytree(book, target)
        seeded += 1
    return seeded


# ── Helpers ───────────────────────────────────────────────────────────────


def _book_choices() -> list[tuple[str, str]]:
    """Build the book-picker dropdown choices.

    Each entry is ``(display_label, book_id)``. Display labels include
    title and page count so visitors know what they're picking.
    """
    out: list[tuple[str, str]] = []
    for book in list_books():
        bid = book["id"]
        title = book.get("title", bid)
        pages = book.get("total_pages", "?")
        lang = book.get("language", "?")
        out.append((f"{title} — {pages} pages ({lang})", bid))
    return out


def _ingest_uploaded_pdf(uploaded_path: str) -> str:
    """Run ``build-book`` on a user-uploaded PDF and return its book_id.

    Gradio's File component gives us a temp-file path; we wrap it as
    a ``Path`` and call ``build_book`` directly (the same entry point
    the CLI uses).
    """
    src = Path(uploaded_path)
    result = build_book(src, force=False)
    return result.book_id


def _resolve_runtime_settings() -> dict:
    """Pick backend + chunk_size from env vars, same precedence as CLI.

    The deployed Space's secrets become env vars, so we honour the
    project-wide ``BOOK_SUMMARIZER_*`` knobs without re-implementing
    resolution logic.
    """
    note_taker_backend = (
        os.getenv("BOOK_SUMMARIZER_NOTE_TAKER_BACKEND", "groq").strip().lower()
    )
    composer_backend = (
        os.getenv("BOOK_SUMMARIZER_COMPOSER_BACKEND", "gemini").strip().lower()
    )
    cerebras_model = _resolve_cerebras_model(None)
    chunk_size = _resolve_chunk_size(None, note_taker_backend, cerebras_model)
    return {
        "note_taker_backend": note_taker_backend,
        "composer_backend": composer_backend,
        "cerebras_model": cerebras_model,
        "chunk_size": chunk_size,
    }


# ── Main run handler ──────────────────────────────────────────────────────


def run_summary(
    book_choice: str | None,
    uploaded_pdf,
    level: str,
    verbosity: str = _VERBOSITY_VERBOSE,
):
    """Streaming generator that runs the pipeline and yields UI updates.

    Outputs (in order, matching ``run_btn.click(... outputs=[...])``):
      1. Activity log textarea
      2. Final summary markdown (empty until the run finishes)
      3. Token report textarea (empty until the run finishes)

    On every ``yield`` Gradio re-renders the corresponding components.
    The function yields tuples of length 3; trailing empty strings keep
    the summary/token panels blank while the run is in progress.
    """

    def _emit(log_text: str, summary: str = "", token_text: str = ""):
        return log_text, summary, token_text

    # ── Budget gate ──────────────────────────────────────────────────
    ok, budget_msg = _BUDGET.try_consume()
    if not ok:
        yield _emit(f"⚠ {budget_msg}\n\n{_BUDGET.status()}")
        return

    # ── Resolve which book to summarise ──────────────────────────────
    log_lines: list[str] = [
        f"{datetime.now():%H:%M:%S}  {budget_msg}",
    ]

    try:
        if uploaded_pdf is not None:
            log_lines.append(
                f"{datetime.now():%H:%M:%S}  Uploaded PDF detected → "
                f"running build-book to extract text + TOC..."
            )
            yield _emit("\n".join(log_lines))
            book_id = _ingest_uploaded_pdf(uploaded_pdf)
            log_lines.append(
                f"{datetime.now():%H:%M:%S}  build-book complete → "
                f"book_id={book_id}"
            )
        elif book_choice:
            book_id = book_choice
        else:
            yield _emit(
                "⚠ Pick a book from the dropdown OR upload a PDF, then click "
                "Summarise."
            )
            return
    except Exception as e:
        log_lines.append(
            f"{datetime.now():%H:%M:%S}  [ERROR] build-book failed: {e}"
        )
        yield _emit(
            "\n".join(log_lines),
            f"## Build failed\n\n```\n{e}\n```",
        )
        return

    settings = _resolve_runtime_settings()
    log_lines.append(
        f"{datetime.now():%H:%M:%S}  Settings — "
        f"note_taker={settings['note_taker_backend']}, "
        f"composer={settings['composer_backend']}, "
        f"chunk_size={settings['chunk_size']}, "
        f"level={level}"
    )
    yield _emit("\n".join(log_lines))

    # ── Wire the pipeline's logger to a streaming queue ──────────────
    log_queue: "queue.Queue[str]" = queue.Queue(maxsize=500)
    file_logger = LocalTextFileLogger(
        "book_summarizer_agent.log",
        level=LogLevel.DEBUG,
        path=str(PROJECT_ROOT),
    )
    ui_min_level = _VERBOSITY_TO_LEVEL.get(verbosity, LogLevel.DEBUG)
    ui_logger = WebUILogger(log_queue, min_level=ui_min_level)
    pipeline_logger = MultiLogger([file_logger, ui_logger])

    try:
        summarizer = BookSummarizer(
            book_id=book_id,
            note_taker_backend=settings["note_taker_backend"],
            composer_backend=settings["composer_backend"],
            cerebras_model=settings["cerebras_model"],
            logger=pipeline_logger,
        )
    except FileNotFoundError as e:
        log_lines.append(
            f"{datetime.now():%H:%M:%S}  [ERROR] {e}"
        )
        yield _emit(
            "\n".join(log_lines),
            f"## Book not found\n\n```\n{e}\n```",
        )
        return

    # ── Background pipeline thread + streaming poll loop ─────────────
    result: dict = {}

    def worker():
        try:
            result["path"] = summarizer.run(
                level, chunk_size=settings["chunk_size"]
            )
        except BaseException as e:
            result["error"] = e

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    last_yielded_count = 0
    while thread.is_alive() or not log_queue.empty():
        # Drain queued log lines without blocking.
        while True:
            try:
                log_lines.append(log_queue.get_nowait())
            except queue.Empty:
                break
        # Only push an update if new lines arrived — saves WebSocket
        # bandwidth on quiet ticks (e.g. while the LLM is thinking).
        if len(log_lines) > last_yielded_count:
            yield _emit("\n".join(log_lines[-_LOG_TAIL_LINES:]))
            last_yielded_count = len(log_lines)
        time.sleep(_POLL_INTERVAL_S)

    thread.join()

    # Final drain, in case any lines were enqueued between our last
    # poll and the thread exiting.
    while True:
        try:
            log_lines.append(log_queue.get_nowait())
        except queue.Empty:
            break

    if "error" in result:
        log_lines.append(
            f"{datetime.now():%H:%M:%S}  [ERROR] Pipeline raised: "
            f"{type(result['error']).__name__}: {result['error']}"
        )
        yield _emit(
            "\n".join(log_lines[-_LOG_TAIL_LINES:]),
            f"## Run failed\n\n```\n{result['error']}\n```",
        )
        return

    out_path: Path = result["path"]
    try:
        summary_text = out_path.read_text(encoding="utf-8")
    except OSError as e:
        summary_text = f"_(could not read summary file: {e})_"

    token_report = summarizer._token_tracker.report()

    log_lines.append(
        f"{datetime.now():%H:%M:%S}  ✓ Done. Summary saved to "
        f"{out_path.name}."
    )
    yield _emit(
        "\n".join(log_lines[-_LOG_TAIL_LINES:]),
        summary_text,
        token_report,
    )


# ── UI definition ─────────────────────────────────────────────────────────


_APP_CSS = """
#activity-log textarea {
    font-family: ui-monospace, Menlo, Monaco, monospace;
    font-size: 12px;
}
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Book Summarizer Agent — Demo") as app:
        gr.Markdown(
            "# Book Summarizer Agent — Live Demo\n"
            "An agentic pipeline that reads a book and produces a summary "
            "at four levels of detail. Pick a pre-built book OR upload a "
            "new PDF, choose a level, and watch the pipeline work.\n\n"
            "**Pipeline**: `BookExplorer` (reads TOC → reading plan) → "
            "`NoteTaker` (per-chunk notes) → `SummaryComposer` "
            "(retrieval-driven for short levels, corpus-stuffing for "
            "comprehensive). All free-tier LLMs (Groq + Gemini Flash)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                book_picker = gr.Dropdown(
                    choices=_book_choices(),
                    label="Pre-built book",
                    info="Or upload a PDF below.",
                    allow_custom_value=False,
                )
                uploaded_pdf = gr.File(
                    label="Upload a PDF (optional — slow, runs full pipeline)",
                    file_types=[".pdf"],
                    type="filepath",
                )
                level = gr.Radio(
                    choices=list(_LEVELS),
                    value="short",
                    label="Summary level",
                    info=(
                        "very short ≈ 1-2 paragraphs · short ≈ 5% · "
                        "medium ≈ 10-15% · comprehensive ≈ full detail"
                    ),
                )
                verbosity = gr.Radio(
                    choices=[_VERBOSITY_VERBOSE, _VERBOSITY_STANDARD],
                    value=_VERBOSITY_VERBOSE,
                    label="Activity log verbosity",
                    info=(
                        "Verbose: every iteration, every prompt, every "
                        "tool call. Standard: just the milestones."
                    ),
                )
                run_btn = gr.Button(
                    "Summarise", variant="primary", size="lg"
                )
                budget_status = gr.Markdown(
                    f"_{_BUDGET.status()}_",
                    visible=True,
                )

            with gr.Column(scale=2):
                activity_log = gr.Textbox(
                    label="Activity log (live)",
                    lines=30,
                    max_lines=30,
                    interactive=False,
                    elem_id="activity-log",
                    placeholder=(
                        "Click Summarise — the pipeline's structured logs "
                        "stream here as it runs.\n\n"
                        "Verbose mode shows BookExplorer's tool calls, "
                        "NoteTaker's per-chunk progress, the canary "
                        "token being injected (and not leaked), "
                        "retry-and-recover behaviour — the full agent "
                        "trace."
                    ),
                )

        with gr.Row():
            summary_md = gr.Markdown(
                value="_The summary will appear here._",
                label="Summary",
            )

        with gr.Row():
            token_report = gr.Textbox(
                label="Token usage & cost",
                lines=10,
                interactive=False,
                placeholder="Per-model token counts and estimated cost will appear here after the run.",
            )

        run_btn.click(
            fn=run_summary,
            inputs=[book_picker, uploaded_pdf, level, verbosity],
            outputs=[activity_log, summary_md, token_report],
            concurrency_limit=1,  # one pipeline run at a time
        )

    return app


def main() -> None:
    seeded = _seed_demo_books()
    if seeded:
        print(f"✓ Seeded {seeded} demo book(s) into {BOOKS_DIR}", flush=True)

    app = build_app()
    demo_password = os.getenv("DEMO_PASSWORD")
    auth = ("demo", demo_password) if demo_password else None
    if auth is None:
        print(
            "⚠ DEMO_PASSWORD not set — running ungated. "
            "Set it as a HuggingFace Space secret for production.",
            flush=True,
        )
    # Gradio 6 moved ``css`` and ``head`` from the Blocks constructor
    # to launch(). ``head`` injects literal HTML into the page <head>
    # — that's where our autoscroll <script> lives, more reliable than
    # ``launch(js=...)`` for one-off page-load behaviour.
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        auth=auth,
        share=False,
        css=_APP_CSS,
        head=_AUTOSCROLL_HEAD_HTML,
    )


if __name__ == "__main__":
    main()
