"""Bounded Stop observation helper (M5 Round 02 skeleton).

Python cannot interrupt a blocking SDK ``Stop()``.  This helper observes the
call in a controlled thread and reports ``timed_out``/``thread_alive`` honestly;
the caller must escalate to STOP_FAILED/E-stop instructions, not claim the robot
is stopped.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class StopOutcome:
    confirmed: bool
    timed_out: bool
    thread_alive: bool
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "confirmed": self.confirmed,
            "timed_out": self.timed_out,
            "thread_alive": self.thread_alive,
            "error": self.error,
        }


def run_stop_with_deadline(
    stop_callable: Callable[[], object],
    timeout_s: float,
    *,
    name: str = "m5-stop",
) -> StopOutcome:
    if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
        raise ValueError("stop timeout must be positive")
    result: dict = {}

    def invoke() -> None:
        try:
            result["value"] = stop_callable()
        except Exception as exc:  # noqa: BLE001
            result["error"] = type(exc).__name__

    thread = threading.Thread(target=invoke, name=name, daemon=True)
    thread.start()
    thread.join(float(timeout_s))
    if thread.is_alive():
        return StopOutcome(confirmed=False, timed_out=True, thread_alive=True, error="stop_timeout")
    if "error" in result:
        return StopOutcome(confirmed=False, timed_out=False, thread_alive=False, error=str(result["error"]))
    return StopOutcome(
        confirmed=result.get("value") is True,
        timed_out=False,
        thread_alive=False,
    )


__all__ = ["StopOutcome", "run_stop_with_deadline"]


def stop_pair_bounded(
    arm_stop: Callable[[], object],
    gn_stop: Callable[[], object],
    timeout_s: float,
    *,
    arm_name: str = "arm",
    gn_name: str = "gn01",
) -> dict:
    """Start both Stop calls in parallel and observe one shared deadline."""
    results: dict = {}
    lock = threading.Lock()

    def run(name: str, callable_) -> None:
        outcome = run_stop_with_deadline(callable_, timeout_s, name=f"m5-stop-{name}")
        with lock:
            results[name] = outcome

    threads = [
        threading.Thread(target=run, args=(arm_name, arm_stop), daemon=True),
        threading.Thread(target=run, args=(gn_name, gn_stop), daemon=True),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(float(timeout_s) + 0.5)
    with lock:
        snapshot = dict(results)
    for name in (arm_name, gn_name):
        if name not in snapshot:
            snapshot[name] = StopOutcome(
                confirmed=False, timed_out=True, thread_alive=True, error="stop_thread_missing"
            )
    return snapshot
