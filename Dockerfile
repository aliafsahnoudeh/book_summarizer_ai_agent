# Single-stage Dockerfile for the public Gradio demo.
#
# Built by .github/workflows/build-and-push.yml on every push to main,
# pushed to ghcr.io/aliafsahnoudeh/book_summarizer_ai_agent:latest.
# HuggingFace Space pulls and runs from there — see DEPLOY.md.

FROM python:3.12-slim

# System packages for PDF + image extraction. Slim image keeps the
# attack surface small; we add only what pypdf[image] / chromadb need.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (used both for sync and as the runtime entry).
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy lockfiles first so deps layer is cached when only source changes.
# README.md is referenced by pyproject.toml's `readme = "README.md"`, so
# hatchling needs it present at sync time when building the local package.
COPY pyproject.toml uv.lock README.md ./

# Install runtime deps. ``--frozen`` ensures we use exactly what's in
# uv.lock — reproducible builds.
RUN uv sync --frozen --no-dev

# Copy source.
COPY book_builder/ book_builder/
COPY book_summarizer/ book_summarizer/
COPY web/ web/

# Demo books baked into the image. HF Spaces' free tier wipes runtime
# disk on every restart, but image layers persist — so the app's
# `_seed_demo_books()` re-populates `.books/` from this read-only
# source on every boot. Kept tiny (currently just the synthetic
# Veridian fixture, ~10 KB) so it doesn't bloat the image.
COPY demo_books/ demo_books/

# Pre-warm: pre-create runtime directories so the first request doesn't
# pay the cost of mkdir on cold disk.
RUN mkdir -p .books .logs

# Gradio's default port — also what HuggingFace Spaces routes to.
EXPOSE 7860

# HF Spaces convention: HOME=/data persistent storage; we don't rely on
# it (everything runs from the read-only image), but if the platform
# sets HOME it shouldn't break us.
ENV PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

CMD ["uv", "run", "python", "-m", "web.app"]
