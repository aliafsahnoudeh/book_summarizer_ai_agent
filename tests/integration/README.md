# Integration tests

Exercises the book pipeline end-to-end **with `litellm.completion` mocked**, on a tiny synthetic book fixture. No real LLM calls; runs in CI on every push.

Planned coverage (Phase 1b):

| Test | What it proves |
|---|---|
| `test_note_taker_writes_one_file_per_chunk` | NoteTaker walks pages in chunks of N, writes `NNN-MMM.md` per chunk, persists `_state.json` |
| `test_note_taker_resumes_from_state` | Second run with existing `_state.json` skips already-processed chunks (no extra LLM calls) |
| `test_state_invalidated_when_total_pages_changes` | State is regenerated if the book changed underneath |
| `test_composer_single_call_path` | When notes corpus < safe_input → one composition call |
| `test_composer_chunked_path` | When notes corpus > safe_input → partition + per-group + merge |

Fixture book lives in `tests/fixtures/tiny_book/` — a 4-page synthetic `.books/<id>/` entry on a fictional topic so models can't draw on training data.
