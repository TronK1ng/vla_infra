"""Execution layer (M1 provisional)."""

from __future__ import annotations

from .authority import AuthoritySnapshot, ControlAuthority
from .executor import ExecutionReport, Executor, ExecutorEvent, ExecutorEventKind, ExecutorStats
from .inbox import ActionInbox, InboxEntry, InboxStats
from .interpolation import CartesianPlan, PlanSample, build_plan
from .safety import CommandTarget, SafetyGate, SafetyProfile
from .supervisor import (
    Supervisor,
    SupervisorConfig,
    SystemState,
    TransitionRecord,
    WatchdogConfig,
    WatchdogStatus,
)

__all__ = [
    "ActionInbox",
    "AuthoritySnapshot",
    "CartesianPlan",
    "CommandTarget",
    "ControlAuthority",
    "ExecutionReport",
    "Executor",
    "ExecutorEvent",
    "ExecutorEventKind",
    "ExecutorStats",
    "InboxEntry",
    "InboxStats",
    "PlanSample",
    "SafetyGate",
    "SafetyProfile",
    "Supervisor",
    "SupervisorConfig",
    "SystemState",
    "TransitionRecord",
    "WatchdogConfig",
    "WatchdogStatus",
    "build_plan",
]
