"""Daily-run budget for the demo deployment.

Even with a password gate, anyone with the secret can hammer the demo
and burn through the LLM provider's daily quota. This module caps the
number of full-pipeline runs accepted per UTC day so the demo stays up
even under accidental abuse.

Counter is in-memory (resets on container restart). For HF Spaces' free
tier that's an automatic safety net — restarts happen routinely, so the
budget effectively rolls forward. We don't need a database.
"""

import threading
from datetime import datetime, timezone


class DailyRunBudget:
    """Thread-safe counter of pipeline runs per UTC day.

    ``try_consume()`` either reserves one run (returns True) or refuses
    (returns False with a reason). The handler should call it before
    starting a pipeline run and surface the refusal reason in the UI.
    """

    def __init__(self, max_runs_per_day: int = 20) -> None:
        self._max = max_runs_per_day
        self._lock = threading.Lock()
        self._date: str = self._today()
        self._count: int = 0

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def try_consume(self) -> tuple[bool, str]:
        """Reserve one run if today's budget isn't exhausted.

        Returns ``(ok, message)``. On refusal, ``message`` is a
        user-facing string explaining the cap and when it resets.
        """
        with self._lock:
            today = self._today()
            if today != self._date:
                # New UTC day — counter rolls over.
                self._date = today
                self._count = 0
            if self._count >= self._max:
                return (
                    False,
                    f"Daily demo limit reached ({self._max} runs per UTC day). "
                    f"Resets at 00:00 UTC. Come back tomorrow.",
                )
            self._count += 1
            return (True, f"Run {self._count}/{self._max} today.")

    def status(self) -> str:
        with self._lock:
            return f"{self._count}/{self._max} runs used today (UTC {self._date})."


__all__ = ["DailyRunBudget"]
