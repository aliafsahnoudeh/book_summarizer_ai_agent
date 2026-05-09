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

## Tests

```bash
uv run pytest                     # unit + integration (no LLM calls; ~2s)
uv run pytest -m eval             # eval suite — real Gemini calls; needs GOOGLE_API_KEY; ~30s
```

CI runs unit + integration on every push ([test.yml](.github/workflows/test.yml)) and the eval suite nightly ([eval.yml](.github/workflows/eval.yml) — requires a `GOOGLE_API_KEY` repo secret).

## Cost preference

This project defaults to free / very cheap LLMs and platforms. When a free option doesn't make sense for a real reason — quality, reliability, or feature gap — we flag it and explain rather than silently accepting it. See [CLAUDE.md](CLAUDE.md#cost-preference).
