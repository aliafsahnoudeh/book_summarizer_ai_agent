"""Unit tests for the demo's DailyRunBudget.

The budget is the only line of defence between a leaked demo password
and a drained LLM provider quota — worth having tests that pin its
contract: it counts up, refuses past the cap, rolls over at UTC
midnight, and is thread-safe.
"""

from unittest.mock import patch

from web.budget import DailyRunBudget


def test_consume_succeeds_until_cap():
    budget = DailyRunBudget(max_runs_per_day=3)

    for _ in range(3):
        ok, _ = budget.try_consume()
        assert ok is True

    ok, msg = budget.try_consume()
    assert ok is False
    assert "Daily demo limit reached" in msg
    assert "3 runs" in msg


def test_status_reports_count_and_date():
    budget = DailyRunBudget(max_runs_per_day=10)
    budget.try_consume()
    budget.try_consume()

    status = budget.status()
    assert "2/10 runs used today" in status
    assert "UTC " in status


def test_counter_rolls_over_at_utc_day_change():
    """Rolling over isn't tied to wall-clock — it's tied to whatever
    ``_today()`` returns. Patching that lets us simulate a date change
    deterministically without waiting for real midnight."""
    budget = DailyRunBudget(max_runs_per_day=2)

    with patch.object(DailyRunBudget, "_today", staticmethod(lambda: "2026-05-08")):
        ok, _ = budget.try_consume()
        assert ok is True
        ok, _ = budget.try_consume()
        assert ok is True
        ok, msg = budget.try_consume()
        assert ok is False  # cap hit on day 1

    # New UTC day → counter rolls over.
    with patch.object(DailyRunBudget, "_today", staticmethod(lambda: "2026-05-09")):
        ok, msg = budget.try_consume()
        assert ok is True
        assert "Run 1/2" in msg


def test_refusal_message_explains_when_it_resets():
    budget = DailyRunBudget(max_runs_per_day=1)
    budget.try_consume()
    ok, msg = budget.try_consume()

    assert ok is False
    assert "00:00 UTC" in msg


def test_concurrent_consumes_dont_overshoot_the_cap():
    """The lock should serialise increments — N threads each calling
    once should never reserve more than max_runs_per_day in total."""
    import threading

    budget = DailyRunBudget(max_runs_per_day=10)
    results = []

    def worker():
        ok, _ = budget.try_consume()
        results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successes = sum(1 for ok in results if ok)
    assert successes == 10  # exactly the cap, not 11 or more
