"""Gradio web UI for the Book Summarizer demo.

Designed to be deployed to HuggingFace Spaces (free tier) via a Docker
image built by GitHub Actions and pushed to ghcr.io. See ``DEPLOY.md``
for the one-time HF Space wiring.

Goals:
- Showcase the full pipeline end-to-end with live log streaming so
  viewers can see BookExplorer, NoteTaker, and SummaryComposer doing
  real work (the agentic part is the most interesting bit to watch).
- Password-gate (single shared secret) to keep anonymous traffic out.
- Daily-run limit so even a leaked password can't drain the LLM quota
  for the day.
"""
