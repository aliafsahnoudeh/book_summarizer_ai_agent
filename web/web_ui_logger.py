"""Logger adaptors for the Gradio UI.

The pipeline's structured log lines are the most interesting part of the
demo to watch — viewers see BookExplorer pick a reading plan, NoteTaker
walk chunks, the auto-split path engage on TPM-overflow, the canary
token being injected (and not leaked), etc. We want those lines streamed
live to the browser, not just buried in a file.

Two pieces:

* ``WebUILogger`` — implements the framework ``Logger`` contract by
  pushing every formatted line into a thread-safe queue. The Gradio
  generator drains the queue on a poll loop and yields concatenated
  text to the live-activity textbox. By default it streams **every**
  log level (including DEBUG) since the agent's internal trace is the
  point of the demo.

* ``MultiLogger`` — fans out a single ``log()`` call to N loggers.
  We tee the WebUILogger to ``LocalTextFileLogger`` so the persistent
  ``.logs/`` history is preserved alongside the live stream.
"""

import queue
from datetime import datetime

from zurvan import Logger, LogLevel


# Numeric ranking so we can compare levels — the framework's LogLevel is
# a StrEnum so we can't compare directly.
_LEVEL_RANK = {
    LogLevel.DEBUG: 0,
    LogLevel.INFO: 1,
    LogLevel.WARNING: 2,
    LogLevel.ERROR: 3,
}


class WebUILogger(Logger):
    """Push every log line to a thread-safe queue.

    The Gradio handler runs on the request thread; the pipeline runs on
    a background thread. The queue is the synchronisation point — the
    pipeline's logger threads emit lines, the handler drains them on
    each poll tick and updates the visible textbox.

    Args:
        sink: Thread-safe queue. The Gradio handler drains it on a
            poll loop.
        min_level: Minimum log level to forward. Defaults to ``DEBUG``
            (i.e. show everything) — the agent's internal trace is the
            most interesting part of the demo to watch. Pass
            ``LogLevel.INFO`` for a cleaner stream.
        max_line_chars: Truncate any line longer than this. DEBUG lines
            occasionally contain whole serialised prompts (~10 K chars)
            which would push the browser's textbox into rerender hell.
            Default 1500 = keeps the full system message visible while
            capping pathological cases.
    """

    def __init__(
        self,
        sink: "queue.Queue[str]",
        min_level: LogLevel = LogLevel.DEBUG,
        max_line_chars: int = 1500,
    ) -> None:
        self._sink = sink
        self._min_rank = _LEVEL_RANK[min_level]
        self._max_line_chars = max_line_chars

    def log(self, message: str, level: LogLevel = LogLevel.INFO, env=None) -> None:
        if _LEVEL_RANK.get(level, 0) < self._min_rank:
            return
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"{ts}  [{level}]  {message}"
        if len(line) > self._max_line_chars:
            line = line[: self._max_line_chars - 16] + " …[truncated]"
        try:
            self._sink.put_nowait(line)
        except queue.Full:
            # Queue capped — drop the oldest line, retry. Keeps the
            # browser stream live under sustained log bursts without
            # blocking the pipeline.
            try:
                self._sink.get_nowait()
            except queue.Empty:
                pass
            try:
                self._sink.put_nowait(line)
            except queue.Full:
                pass  # give up; the line is lost (rare)


class MultiLogger(Logger):
    """Broadcast every ``log()`` call to N underlying loggers.

    Used to tee the WebUILogger (live stream) and the LocalTextFileLogger
    (persistent ``.logs/`` history) so the demo run shows up in both
    places.
    """

    def __init__(self, loggers: list[Logger]) -> None:
        self._loggers = list(loggers)

    def log(self, message: str, level: LogLevel = LogLevel.INFO, env=None) -> None:
        for lg in self._loggers:
            try:
                lg.log(message, level, env)
            except Exception:
                # A failing tee target must not break the others.
                pass


__all__ = ["WebUILogger", "MultiLogger"]
