# Book Summarizer Agent

A three-stage pipeline that summarises a pre-processed book at four levels of detail, in the book's original language. Built on [`zurvan`](https://pypi.org/project/zurvan/) — the GAME (Goals, Actions, Memory, Environment) agent framework that was extracted from this project into a standalone PyPI package.

## Cost Preference

Default to free or very cheap LLMs, tools, and platforms. Future deployment also targets free tiers where viable.

When a free option doesn't make sense — quality, reliability, or feature gap that materially hurts the project — flag it and explain the trade-off rather than silently accepting it. Don't paper over real limitations.

## Pipeline

```
.books/<book_id>/  →  BookExplorer  →  NoteTaker  →  SummaryComposer
                      reading plan       comprehensive    retrieval-driven
                      (skip/skim/read)   notes per chunk  for short levels;
                      cached             resumable        single-or-chunked
                      Gemini agent       skips skip-pages otherwise
```

1. **BookExplorer** — a **GAME-loop agent** with **one tool** (terminal `set_reading_strategy`). Book metadata and table of contents are pre-loaded into the agent's goal description rather than fetched via tools — fetching data we already hold on disk would burn LLM round-trips for no decision value. The agent runs in a single iteration: see the data in the goal, emit the plan, terminate. Same cost as a direct LLM call; same shape as every future agent we'll add. Tool input is validated by a **Pydantic schema** (`SetReadingStrategyArgs`) before the function body runs — invalid plans are returned as structured errors. The plan is persisted to `.books/<book_id>/reading_plan.json` and cached across runs (re-runs only when missing or stale). On any failure (missing API key, malformed plan, etc.) the explorer falls back to an all-`read` plan so the rest of the pipeline is never blocked.
2. **NoteTaker** — a Python loop walks the book in fixed-size chunks. For each unread chunk it issues one focused **`ChunkNoteWriter`** LLM call that receives only that chunk's pages plus a tiny static header (book title, language, top-level TOC). No GAME loop, no accumulated memory, no tool calls. Chunks whose pages are entirely marked `skip` in the reading plan are not sent to the LLM. Notes (one `.md` file per chunk) are written to `.books/<book_id>/notes/` in the book's original language. State is persisted to `notes/_state.json` so an interrupted run resumes from the failed chunk.
3. **SummaryComposer** — synthesises the chunk notes into a Markdown summary at the requested level (`very short` / `short` / `medium` / `comprehensive`). Picks one of three paths: **retrieval** for short / very-short levels (vector-queries the notes index for theme-driven chunks via [`NoteIndexer`](book_summarizer/note_indexer.py), composes from focused selection — avoids stuffing the full corpus and the "telephone game" of summarising summaries on long books); **single-call** for medium / comprehensive when the corpus fits the backend's safe input budget; **chunked-merge** as a fallback when even the corpus path overflows. Output is in the book's original language.

## Setup

```bash
uv run build-book path/to/book.pdf                          # one-time book preprocessing
uv run book-summarizer                                      # interactive (recommended)
uv run book-summarizer --book <book_id>                     # specific book
uv run book-summarizer --fresh                              # discard saved notes & start clean
uv run book-summarizer --note-taker-backend gemini          # override note-taker backend
uv run book-summarizer --composer-backend gemini            # override composer backend
uv run book-summarizer --cerebras-model qwen-3-235b-...     # use qwen on Cerebras for notes
uv run book-summarizer --chunk-size 5                       # override chunk size
```

Required environment variables (set in `.env`) — only the ones for the backends you use:

- `CEREBRAS_API_KEY` — Cerebras (default backend for both note-taker and composer).
- `GOOGLE_API_KEY` — Gemini (any role using it; default composer).
- `GROQ_API_KEY` — Groq (any role using it).

## Backend Selection

Each role picks its own backend independently. CLI prompts interactively if not given via flag.

| Role              | Default (CLI)              | Default (eval)             | Choices                        |
| ----------------- | -------------------------- | -------------------------- | ------------------------------ |
| explorer          | `gemini-2.5-flash` (fixed) | `gemini-2.5-flash` (fixed) | —                              |
| note_taker        | `cerebras`                 | `groq`                     | `cerebras` / `gemini` / `groq` |
| composer          | `gemini`                   | `gemini`                   | `cerebras` / `gemini` / `groq` |
| judge (eval only) | —                          | `gemini-2.5-flash` (fixed) | —                              |

**Backend selection precedence** (12-factor):

```
CLI flag  >  env var  >  interactive prompt (backends only)  >  built-in default
```

The same env vars drive both the CLI and the eval — set once in `.env` (or shell), they apply to both:

| Env var                              | Drives     | Default                          |
| ------------------------------------ | ---------- | -------------------------------- |
| `BOOK_SUMMARIZER_NOTE_TAKER_BACKEND` | CLI + eval | `cerebras` (CLI) / `groq` (eval) |
| `BOOK_SUMMARIZER_COMPOSER_BACKEND`   | CLI + eval | `gemini`                         |
| `BOOK_SUMMARIZER_CEREBRAS_MODEL`     | CLI        | `llama3.1-8b`                    |
| `BOOK_SUMMARIZER_CHUNK_SIZE`         | CLI        | per-backend smart default        |

When the two backend vars are set, the CLI **skips its interactive backend prompt** entirely — no point asking a question the user has already answered in their environment. The other two are CLI-flag-driven; the env var is a convenience for users who run the same configuration repeatedly. Override for one run via either a CLI flag or a shell-exported variable:

```bash
book-summarizer --note-taker-backend cerebras       # one-off CLI override
BOOK_SUMMARIZER_NOTE_TAKER_BACKEND=cerebras book-summarizer   # shell override
```

The eval suite reads the same vars (see [tests/eval/conftest.py](tests/eval/conftest.py)) and skips cleanly with an actionable message when the key for any chosen backend is missing.

**Every run logs its full config in a header at the top of the log file** ([book_summarizer/book_summarizer.py](book_summarizer/book_summarizer.py) → `_log_run_header`): book_id, level, language, chunk_size, both backends + models, and which API keys are set. Keys are reported as `set` / `MISSING` only — never logged in plaintext.

On Cerebras, the **note-taker model is selectable** (`--cerebras-model`):

| Cerebras model                   | Tier       | Context | RPM | TPM  | Best for                                                       |
| -------------------------------- | ---------- | ------- | --- | ---- | -------------------------------------------------------------- |
| `llama3.1-8b` (default)          | Production | 8 K     | 30  | 60 K | Many small chunks; reliable.                                   |
| `qwen-3-235b-a22b-instruct-2507` | Preview    | 65 K    | 5   | 30 K | Bigger chunks, better quality, occasional "high traffic" 429s. |

The **composer defaults to Gemini 2.5 Flash** — its 1 M context fits the entire notes corpus for any reasonable book in a single call, and the free-tier RPD (1500/day historically) is generous enough for many runs. If you pick Cerebras for the composer, it always uses `qwen-3-235b` (the 65 K-context model), but Cerebras Preview tier 429s frequently — Gemini Flash is the more reliable choice.

## Chunked Composition

For a 650-page book the note corpus can be 100 K+ tokens — way more than any free-tier model's per-call window. `SummaryComposer` handles this automatically:

1. Estimate the total token size of the notes corpus.
2. If it fits the backend's `safe_input_tokens` budget → single composition call.
3. If not → partition chunk files into groups that each fit, compose a partial summary per group, then merge the partials into one final summary.

Tuning lives in `_PROFILES` in [book_summarizer/book_summarizer.py](book_summarizer/book_summarizer.py): `composer_safe_input` per backend/model. Cerebras qwen is conservatively set to 22 K so each call stays well under both the 30 K TPM cap and the 65 K context window.

## Per-backend models and limits

Verified against each provider's live `/v1/models` endpoint at the time of writing:

| Backend / model            | Context | RPM | TPM   | RPD    | Notes                                                                                                     |
| -------------------------- | ------- | --- | ----- | ------ | --------------------------------------------------------------------------------------------------------- |
| `cerebras/llama3.1-8b`     | 8 K     | 30  | 60 K  | 14.4 K | Note-taker default. Tiny context dictates `--chunk-size 4`.                                               |
| `cerebras/qwen-3-235b-...` | 65 K    | 5   | 30 K  | 14.4 K | Composer default. 235 B model. Preview tier — needs retry-on-429.                                         |
| `gemini/gemini-2.5-flash`  | 1 M     | ~10 | high  | high   | **Composer default.** Thinking-token model — needs ≥8 K max_tokens.                                       |
| `gemini/gemini-2.0-flash`  | 1 M     | 15  | 1 M   | 1500   | Use only if your key has fresh quota — sometimes exhausted.                                               |
| `gemini/gemini-2.5-pro`    | 1 M     | 5   | 250 K | 100    | Tight RPD; one-shot composer only.                                                                        |
| `groq/qwen/qwen3-32b`      | 32 K    | 30  | 6 K   | ~14 K  | TPM is the bottleneck; default is `--chunk-size 4`. Auto-split-on-overflow handles the rare denser chunk. |

Free-tier limits change frequently. To check what's currently available:

```bash
curl -s -H "Authorization: Bearer $CEREBRAS_API_KEY" \
     https://api.cerebras.ai/v1/models | jq '.data[].id'
```

## Default chunk sizes

CLI picks chunk_size based on (note_taker_backend, cerebras_model) unless `--chunk-size N` is passed:

| (backend, model)            | Default `chunk_size`                                                                                                    |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| (cerebras, llama3.1-8b)     | **4** — fits 8 K context                                                                                                |
| (cerebras, qwen-3-235b-...) | **20** — bigger context, smarter model                                                                                  |
| (gemini, \*)                | 20                                                                                                                      |
| (groq, \*)                  | **4** — keeps under 6 K TPM cap on dense pages (was 5; lowered after observing 45% TPM-overflow rate on academic prose) |

## Pre-emptive rate limiting

Both `ChunkNoteWriter` and `SummaryComposer` use a TPM- and RPM-aware pacer (`_RateLimiter`). Between calls it sleeps `max(tpm_wait, rpm_wait)` to keep the rolling 60-second window under the backend's caps. Plus retry-with-backoff on transient 429s (Cerebras Preview qwen often emits "high traffic" 429s that resolve after a 30-second wait).

If a request size already exceeds the per-minute quota on its own (request > limit), no waiting helps — both layers fail fast with `ChunkTooLargeError` (note-taker) or a `RuntimeError` (composer) carrying an actionable suggestion.

## Per-Book Layout

Every artefact derived from a book lives under that book's directory in `.books/`. `rm -rf .books/<book_id>/` wipes everything for that book in one step.

```
.books/<book_id>/
├── metadata.json        # title, authors, language, total_pages, source_files
├── toc.json             # extracted table of contents
├── pages/               # one .txt file per page
│   ├── 001.txt
│   └── ...
├── visuals/             # extracted images (when Pillow is available)
├── chroma_db/           # per-book raw-pages ChromaDB (built by build-book)
├── notes_index/         # per-book notes ChromaDB (built lazily by NoteIndexer)
├── reading_plan.json    # BookExplorer's per-page importance plan (cached)
├── notes/               # resumable note cache
│   ├── _state.json      # pages_read, chunks index, completed flag, chunk_size
│   ├── 001-004.md       # comprehensive markdown notes for pages 1-4
│   ├── 005-008.md
│   └── ...
└── summaries/           # final summaries, one file per level
    ├── very_short.md
    ├── short.md
    ├── medium.md
    └── comprehensive.md
```

Each summary file is overwritten on subsequent runs at the same level for the same book. Note state is invalidated automatically if `source_files` or `total_pages` changes. Use `--fresh` to wipe notes for a book and start over.

Logs go to `<project_root>/.logs/book_summarizer_agent.log_<timestamp>.txt` — outside the package directory so future `pip install`s don't try to write inside read-only `site-packages/`.

## Summary Levels

| Level             | Output Size         | Notes Coverage           |
| ----------------- | ------------------- | ------------------------ |
| **very short**    | 1-2 paragraphs      | Same comprehensive notes |
| **short**         | ~5% of the book     | Same comprehensive notes |
| **medium**        | ~10-15% of the book | Same comprehensive notes |
| **comprehensive** | Full detail         | Same comprehensive notes |

Note-taking aims for ≥80% page coverage — the level only affects composition, not reading. Frontmatter, bibliography, and index pages may be skimmed or skipped.

## Architecture

| File                                                                         | Role                                                                                                                                                                                                                                          |
| ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [book_summarizer/main.py](book_summarizer/main.py)                           | CLI entry point; book selection, prompts, progress bar.                                                                                                                                                                                       |
| [book_summarizer/book_summarizer.py](book_summarizer/book_summarizer.py)     | `BookSummarizer` orchestrator: per-(backend, model) profiles, wires BookExplorer → NoteTaker → Composer.                                                                                                                                      |
| [book_summarizer/book_explorer.py](book_summarizer/book_explorer.py)         | `BookExplorer` — first GAME-loop agent. Emits a per-page reading plan via Pydantic-validated tool calls. Always uses Gemini 2.5 Flash.                                                                                                        |
| [book_summarizer/note_taker.py](book_summarizer/note_taker.py)               | Python loop driving per-chunk note-writing. Hosts `NoteTaker` (the loop) and `ChunkNoteWriter` (single-LLM-call sub-agent) plus the shared `_RateLimiter`.                                                                                    |
| [book_summarizer/summary_composer.py](book_summarizer/summary_composer.py)   | `SummaryComposer` — retrieval / single-call / chunked-merge paths with retry-on-429 + connection-error backoff.                                                                                                                               |
| [book_summarizer/note_indexer.py](book_summarizer/note_indexer.py)           | `NoteIndexer` — per-book ChromaDB collection of chunk notes; theme-driven retrieval for the composer.                                                                                                                                         |
| [`zurvan`](https://pypi.org/project/zurvan/) (PyPI dep)                      | Reusable GAME-loop agent infrastructure (Goals, Actions, Memory, Environment, plus `Capability` hook system and `CanaryCapability`). Originally lived as a bundled `framework/` directory in this project; extracted to a standalone package. |
| [book_builder/](book_builder/)                                               | One-time PDF → `.books/<id>/` extractor (text per page, TOC, visuals, metadata).                                                                                                                                                              |
| [web/](web/)                                                                 | Gradio app for the public demo (Phase 5). Live log streaming, password-gated, daily-run budget.                                                                                                                                               |
| [Dockerfile](Dockerfile)                                                     | Single-stage image used by the demo deployment. Built by GitHub Actions, pushed to `ghcr.io`, pulled by HuggingFace Spaces.                                                                                                                   |
| [.github/workflows/build-and-push.yml](.github/workflows/build-and-push.yml) | Phase 5 CI: builds the demo image on every `main` push and pushes to `ghcr.io`. See [DEPLOY.md](DEPLOY.md).                                                                                                                                   |

## Prompt-injection defenses

Books are user-uploaded PDFs — anything from the PDF (title, authors, TOC entries, page text) is **untrusted content** and could contain attempted prompt injections. We defend in three layers:

1. **Untrusted-data tagging.** Every prompt that contains book-derived content wraps it in `<UNTRUSTED_BOOK_CONTENT>...</UNTRUSTED_BOOK_CONTENT>` tags, with an explicit instruction in the system prompt that content inside is _data, not instructions_. Applies to BookExplorer (TOC + metadata), NoteTaker (page text), and SummaryComposer (notes corpus + partial summaries).

2. **Canary token tripwire.** Every BookExplorer agent run injects a unique secret string into its prompt and tells the model to never repeat it. Every LLM response is scanned for the canary; if it ever appears, the response is discarded and the agent's retry budget treats it as a normal failure. Implemented as `CanaryCapability` in [zurvan](https://pypi.org/project/zurvan/) (`from zurvan import CanaryCapability`) — it composes with any future agent via the framework's existing `Capability` hook system.

3. **Pydantic-validated tool inputs.** Every tool action takes a Pydantic model as input (validated before the function body runs); structurally invalid plans (wrong types, gaps, overlaps, out-of-range pages) are rejected before they can affect state. Combined with `set_reading_strategy` being the only filesystem-affecting action and being terminal, the blast radius of a hijacked agent is bounded to "produce a malformed plan that gets rejected".

When all three layers fail (model produces a structurally valid but functionally bad plan), the orchestrator's all-read fallback ensures the pipeline still completes — just without the explorer's optimization.

The red-team test suite at [tests/red_team/](tests/red_team/) runs deterministic attack scenarios (canary leak in tool args, canary leak in text response, malformed plan, clean control) on every push. Future attack patterns can be added as new test cases in the same file.

## Operational Notes

- Note-taking issues exactly one LLM call per unread chunk. For a 650-page book at `--chunk-size 4` (Cerebras llama default), that's ~165 calls; at `--chunk-size 20` (Cerebras qwen) it's ~33.
- The same comprehensive notes are reused across summary levels — re-running with a different level only re-runs the Composer.
- Token usage and cost are logged to `book_summarizer_agent.log` at the end of each run via `TokenTracker`.
- All page numbers are 1-based throughout the system.
