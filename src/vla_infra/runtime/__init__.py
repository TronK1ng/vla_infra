"""Runtime harnesses (M1 shadow, M2 collection)."""

from __future__ import annotations

from .collection import CollectionHarness, CollectionResult, build_collection_metadata
from .execution_log import ExecutionLog
from .m3 import M3Config, M3Result, M3Runtime
from .shadow import HarnessResult, ShadowHarness, build_adapter

__all__ = [
    "CollectionHarness",
    "CollectionResult",
    "ExecutionLog",
    "M3Config",
    "M3Result",
    "M3Runtime",
    "HarnessResult",
    "ShadowHarness",
    "build_adapter",
    "build_collection_metadata",
]
