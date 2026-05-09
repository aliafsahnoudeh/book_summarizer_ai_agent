"""Unit tests for the WebUILogger.

The web demo's most-watched surface is the live activity log — this
logger is what fills it. Tests pin: the default-show-everything
behaviour, the level filter, and the long-line truncation that
prevents the textbox from rendering 10 KB prompt-dump lines.
"""

import queue

import pytest

from zurvan import LogLevel
from web.web_ui_logger import MultiLogger, WebUILogger


def _drain(q: "queue.Queue[str]") -> list[str]:
    out: list[str] = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_default_min_level_is_debug_so_everything_streams():
    """User asked for the agent's full internal trace in the UI by
    default. Every level (DEBUG, INFO, WARNING, ERROR) must reach the
    queue without an explicit min_level override."""
    sink: "queue.Queue[str]" = queue.Queue()
    logger = WebUILogger(sink)

    logger.log("debug detail", LogLevel.DEBUG)
    logger.log("info milestone", LogLevel.INFO)
    logger.log("warning condition", LogLevel.WARNING)
    logger.log("error event", LogLevel.ERROR)

    lines = _drain(sink)
    assert len(lines) == 4
    assert any("debug detail" in line for line in lines)
    assert any("info milestone" in line for line in lines)


def test_min_level_filters_below_threshold():
    sink: "queue.Queue[str]" = queue.Queue()
    logger = WebUILogger(sink, min_level=LogLevel.WARNING)

    logger.log("debug detail", LogLevel.DEBUG)
    logger.log("info milestone", LogLevel.INFO)
    logger.log("warning condition", LogLevel.WARNING)
    logger.log("error event", LogLevel.ERROR)

    lines = _drain(sink)
    assert len(lines) == 2  # only WARNING + ERROR
    assert all("warning" in line.lower() or "error" in line.lower() for line in lines)


def test_long_lines_are_truncated_with_marker():
    """DEBUG-level prompt dumps can be 10 KB+; without truncation the
    browser's textbox stalls under the rerender. Truncation keeps a
    visible marker so viewers know the line was cut, not just blank."""
    sink: "queue.Queue[str]" = queue.Queue()
    logger = WebUILogger(sink, max_line_chars=100)

    long_message = "x" * 5000
    logger.log(long_message, LogLevel.INFO)

    lines = _drain(sink)
    assert len(lines) == 1
    assert len(lines[0]) <= 100
    assert "[truncated]" in lines[0]


def test_short_lines_are_unchanged():
    sink: "queue.Queue[str]" = queue.Queue()
    logger = WebUILogger(sink, max_line_chars=1500)

    logger.log("a brief message", LogLevel.INFO)

    lines = _drain(sink)
    assert "a brief message" in lines[0]
    assert "[truncated]" not in lines[0]


def test_log_format_includes_timestamp_and_level():
    sink: "queue.Queue[str]" = queue.Queue()
    logger = WebUILogger(sink)

    logger.log("hello", LogLevel.INFO)

    lines = _drain(sink)
    # Format: "HH:MM:SS  [INFO]  hello"
    assert "[INFO]" in lines[0]
    assert "hello" in lines[0]
    # Timestamp is HH:MM:SS — has colons in the right places.
    assert lines[0].count(":") >= 2


def test_full_queue_drops_oldest_to_make_room():
    """Queue overflow must not block the pipeline — old lines get
    dropped so the most recent ones survive."""
    sink: "queue.Queue[str]" = queue.Queue(maxsize=3)
    logger = WebUILogger(sink)

    for i in range(5):
        logger.log(f"line-{i}", LogLevel.INFO)

    lines = _drain(sink)
    # Queue was capped at 3; only the last few survive.
    assert len(lines) <= 3
    # The very newest definitely makes it.
    assert any("line-4" in line for line in lines)


# ── MultiLogger ─────────────────────────────────────────────────────────


def test_multi_logger_fans_out_to_every_target():
    sink_a: "queue.Queue[str]" = queue.Queue()
    sink_b: "queue.Queue[str]" = queue.Queue()
    logger = MultiLogger([WebUILogger(sink_a), WebUILogger(sink_b)])

    logger.log("broadcast", LogLevel.INFO)

    assert "broadcast" in _drain(sink_a)[0]
    assert "broadcast" in _drain(sink_b)[0]


def test_multi_logger_swallows_per_target_failures():
    """If one target raises, the others must still receive the line —
    a single broken file logger should not silence the live UI."""

    class _BrokenLogger:
        def log(self, message, level=LogLevel.INFO, env=None):
            raise RuntimeError("disk full")

    sink: "queue.Queue[str]" = queue.Queue()
    logger = MultiLogger([_BrokenLogger(), WebUILogger(sink)])

    logger.log("survive", LogLevel.INFO)  # must not raise

    lines = _drain(sink)
    assert any("survive" in line for line in lines)
