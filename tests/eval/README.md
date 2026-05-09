# Eval tests

Real-LLM tests that score the **quality** of summaries against a golden fixture book. Slow, costs free-tier tokens. **Opt-in** — run with `pytest -m eval`.

Skipped by default in `pytest` and CI on every push. Intended to run nightly or before a release.

## Why it's separate

Unit + integration tests catch wiring regressions. Eval tests catch **quality** regressions: a prompt change that produces a faithful but useless summary, or a model swap that introduces hallucinations.

## Planned coverage (Phase 1b)

| Test | Metric | Threshold |
|---|---|---|
| `test_summary_is_faithful` | LLM-as-judge: every claim in the summary appears in the source notes | Mean ≥ 1.5 / 2 across N=3 runs |
| `test_summary_covers_main_themes` | LLM-as-judge: required themes from the fixture's `expected_themes.json` are present | All themes present in ≥ 2 / 3 runs |
| `test_short_summary_size_budget` | Char count of `short` summary is within ±30 % of 5 % of book size | Hard threshold |

## Judge model

Free-tier Gemini 2.5 Flash. Cheap, generous RPD. If judge consistency becomes a problem we'll upgrade to a paid Sonnet/4o-class judge — flagged in `CLAUDE.md` as a known free-vs-quality trade-off.

## Fixture book

`tests/fixtures/golden_book/` — a 12-page synthetic book on a fictional topic (so models can't pull answers from training data). Contains:

- `pages/001.txt` … `012.txt`
- `metadata.json`
- `toc.json`
- `expected_themes.json` — ground-truth themes the eval asserts coverage of
- `forbidden_claims.json` — claims that must NOT appear (faithfulness negative tests)

## Required env

- `GOOGLE_API_KEY` — judge model + composer
- `CEREBRAS_API_KEY` — note-taker

CI workflow `eval.yml` (Phase 1c) runs nightly and pulls these from repo secrets.
