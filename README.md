# Book Summarizer AI Agent

[![Tests](https://github.com/aliafsahnoudeh/book_summarizer_ai_agent/actions/workflows/test.yml/badge.svg)](https://github.com/aliafsahnoudeh/book_summarizer_ai_agent/actions/workflows/test.yml)
[![Eval](https://github.com/aliafsahnoudeh/book_summarizer_ai_agent/actions/workflows/eval.yml/badge.svg)](https://github.com/aliafsahnoudeh/book_summarizer_ai_agent/actions/workflows/eval.yml)

Summarise a book at four levels of detail, in any target language, using free-tier LLMs.

Built on [`zurvan`](https://pypi.org/project/zurvan/) — a small reusable GAME (Goals, Actions, Memory, Environment) agent framework, extracted from this project as a standalone Python package.

## Install

```bash
uv sync
```

Set whichever provider keys you intend to use in a `.env` at the project root:

```
CEREBRAS_API_KEY=...    # Cerebras (default note-taker / composer fallback)
GOOGLE_API_KEY=...      # Gemini (default composer)
GROQ_API_KEY=...        # Groq (alternative)
```

## Quickstart

```bash
uv run build-book path/to/book.pdf      # one-time PDF preprocessing
uv run book-summarizer                  # interactive summary
```

See [CLAUDE.md](CLAUDE.md) for the full pipeline, backend selection, chunked composition, and tuning details.

## Layout

| Directory | Role |
|---|---|
| [`zurvan`](https://pypi.org/project/zurvan/) (PyPI dep) | Reusable GAME-loop agent infrastructure (one `Agent` class, composable `Capability` hooks, provider-agnostic LLM bindings). Extracted from this project. |
| [book_summarizer/](book_summarizer/) | NoteTaker → SummaryComposer pipeline (translation removed for now). CLI entry point: `book-summarizer`. |
| [book_builder/](book_builder/) | One-time PDF → `.books/<id>/` extractor (text per page, TOC, visuals, ChromaDB index). CLI entry point: `build-book`. |

## Engineering highlights

A few design decisions worth calling out:

- **GAME-loop agent infrastructure extracted as a reusable package.** What started as an internal `framework/` directory grew up into [`zurvan`](https://pypi.org/project/zurvan/) on PyPI — Goals/Actions/Memory/Environment with composable `Capability` hooks. This project consumes it as a normal dependency, demonstrating that the abstraction generalises beyond its original use case.

- **Multi-stage pipeline with role-specific backends.** A GAME-loop reading planner ([`BookExplorer`](book_summarizer/book_explorer.py)) decides skip / skim / read per page, a Python-loop note-taker drives per-chunk writers ([`note_taker.py`](book_summarizer/note_taker.py)), and a composer ([`SummaryComposer`](book_summarizer/summary_composer.py)) picks one of three paths (retrieval / single-call / chunked-merge) based on corpus size. Each role selects its backend independently — explorer fixed to Gemini for tool-calling reliability, the others swappable via flag or env var.

- **Pydantic-validated tool inputs.** Every agent action takes a Pydantic model — structurally invalid plans (gaps, overlaps, out-of-range pages, wrong types) are rejected before they touch state. JSON-schema `$ref`s are inlined for Gemini compatibility.

- **Layered prompt-injection defenses.** Books are user-uploaded PDFs — anything inside is untrusted. Three layers: `<UNTRUSTED_BOOK_CONTENT>` tagging on every prompt, a per-run canary tripwire (`CanaryCapability`) that scans every model response and discards leaks, and the Pydantic validation above bounding the blast radius. A [red-team test suite](tests/red_team/) runs deterministic attack scenarios on every push.

- **Retrieval-driven composition for long books.** For short / very-short summary levels, a per-book ChromaDB index of chunk notes ([`NoteIndexer`](book_summarizer/note_indexer.py)) is theme-queried by the composer — avoiding both context-window stuffing and the "telephone game" of summarising summaries. The composer falls back to a single-call path when the full notes corpus fits, or a chunked-merge path when even that overflows.

- **LLM-as-judge eval, by explicit human action only.** [tests/eval/](tests/eval/) runs the full pipeline against a tiny synthetic fixture ("Veridian Settlement") and uses Gemini 2.5 Flash as a faithfulness judge. The eval workflow is manual-trigger only — no LLM call ever happens without an explicit click on the demo's Summarise button or the workflow's Run button.

- **Failure-tolerant by design.** Pre-emptive RPM/TPM rate limiting (sleep before the call, not after the 429); retry-with-backoff on 429s and connection errors; auto-split-on-overflow when a single chunk would exceed the per-minute quota; resumable state so a crashed run picks up where it left off.

- **Cost and quota visibility as a first-class concern.** Token usage is tracked per call and logged at run completion. A per-UTC-day run budget caps the public demo. The full config — backends, models, which API keys are set (`set` / `MISSING`, never plaintext) — is logged in a header at the top of every run.

## Tests

```bash
uv run pytest                     # unit + integration (no LLM calls; ~2s)
uv run pytest -m eval             # eval suite — real Gemini calls; needs GOOGLE_API_KEY; ~30s
```

CI runs unit + integration on every push ([test.yml](.github/workflows/test.yml)). The eval suite ([eval.yml](.github/workflows/eval.yml)) is **manual-trigger only** — by design, no LLM call happens without an explicit human action.

## Cost preference

This project defaults to free / very cheap LLMs and platforms. When a free option doesn't make sense for a real reason — quality, reliability, or feature gap — we flag it and explain rather than silently accepting it. See [CLAUDE.md](CLAUDE.md#cost-preference).

## Deployment

Live demo: **https://huggingface.co/spaces/aliafsah1988/book_summarizer_ai_agent**

End-to-end free-tier pipeline:

```
git push main
   ↓
GitHub Actions  ──  builds Docker image
   ↓
ghcr.io  ──────────  public image registry
   ↓
HuggingFace Space  ─  free CPU runtime, public URL
```

- **Build & push** ([build-and-push.yml](.github/workflows/build-and-push.yml)): on every push to `main` that touches image-relevant paths, a Docker image is built and pushed to GitHub Container Registry. Test-only changes don't trigger a redeploy.
- **Image**: single-stage `python:3.12-slim`, deps installed via `uv sync --frozen`, source + a synthetic demo book baked in so the app boots with at least one runnable example.
- **Runtime**: HuggingFace Space (Docker SDK, free CPU tier) pulls the image. A shared-password gate plus a per-UTC-day run budget keep the public demo within free-tier LLM quotas even if the password leaks.
- **Secrets**: provider API keys + demo password live as Space secrets — never in the image, never in git.
