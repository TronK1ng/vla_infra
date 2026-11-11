"""M5 preflight report schema/runner (Round 01; injected probes only).

Fail-closed rules:

* a probe must return ``ProbeResult`` or a mapping with exactly ``status``
  ``PASS``/``FAIL`` plus optional JSON-serializable details;
* missing, unknown, malformed, non-serializable, or FAIL probe results block;
* every required enable probe must be present and explicitly PASS;
* an enable verdict also requires an explicit operator ACK.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from types import MappingProxyType
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Tuple


class PreflightVerdict(str, Enum):
    PASS_READONLY = "PASS_READONLY"
    PASS_ENABLE_ALLOWED = "PASS_ENABLE_ALLOWED"
    FAIL = "FAIL"


class ProbeStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


REQUIRED_SECTIONS = (
    "software",
    "identity",
    "network",
    "robot",
    "tool",
    "calibration",
    "state",
    "safety",
    "storage",
    "ownership",
)

# Fixed by design; callers may not shrink this list.
REQUIRED_ENABLE_PROBES = (
    "operator_ack",
    "estop_released",
    "writer_lock",
    "config_hash_match",
)


@dataclass(frozen=True)
class ProbeResult:
    status: ProbeStatus
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, ProbeStatus):
            raise ValueError("status must be ProbeStatus")
        if not isinstance(self.details, Mapping):
            raise ValueError("details must be a mapping")

    def as_dict(self) -> dict:
        return {"status": self.status.value, "details": dict(self.details)}


def _normalise_probe_result(value: Any) -> dict:
    if isinstance(value, ProbeResult):
        result = value.as_dict()
    elif isinstance(value, Mapping):
        if "status" not in value:
            raise ValueError("probe result has no status")
        status = value.get("status")
        if isinstance(status, ProbeStatus):
            status = status.value
        if status not in (ProbeStatus.PASS.value, ProbeStatus.FAIL.value):
            raise ValueError("probe status must be PASS or FAIL")
        details = value.get("details", {})
        if not isinstance(details, Mapping):
            raise ValueError("probe details must be a mapping")
        result = {"status": status, "details": dict(details)}
    else:
        raise ValueError("probe result must be ProbeResult or a mapping")
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("probe result is not JSON-serializable") from exc
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


@dataclass(frozen=True)
class M5PreflightReport:
    schema_version: str
    generated_at_ns: int
    expires_at_ns: int
    station_id: str
    config_hash: str
    sections: Mapping[str, Any]
    verdict: PreflightVerdict
    blockers: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != "m5-preflight/v1":
            raise ValueError("unsupported preflight schema")
        if not isinstance(self.verdict, PreflightVerdict):
            raise ValueError("verdict must be PreflightVerdict")
        if self.expires_at_ns < self.generated_at_ns:
            raise ValueError("preflight expiry precedes generation")
        canonical = _canonical(dict(self.sections))
        object.__setattr__(self, "_sections_canonical", canonical)
        object.__setattr__(self, "sections", _freeze(json.loads(canonical)))
        object.__setattr__(self, "blockers", tuple(str(b) for b in self.blockers))

    def _sections_dict(self) -> dict:
        return json.loads(self._sections_canonical)

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "generated_at_ns": self.generated_at_ns,
            "expires_at_ns": self.expires_at_ns,
            "station_id": self.station_id,
            "config_hash": self.config_hash,
            "sections": self._sections_dict(),
            "verdict": self.verdict.value,
            "blockers": list(self.blockers),
        }

    def report_hash(self) -> str:
        import hashlib

        return hashlib.sha256(self._canonical_payload().encode("utf-8")).hexdigest()

    def _canonical_payload(self) -> str:
        return _canonical(self.as_dict())

    def validate_for_enable(
        self,
        *,
        expected_station_id: Optional[str] = None,
        expected_config_hash: Optional[str] = None,
        now_ns: Optional[int] = None,
    ) -> None:
        if expected_station_id is not None and self.station_id != expected_station_id:
            raise ValueError("preflight station_id does not match expected")
        if expected_config_hash is not None and self.config_hash != expected_config_hash:
            raise ValueError("preflight config_hash does not match expected")
        if now_ns is not None and not (self.generated_at_ns <= int(now_ns) <= self.expires_at_ns):
            raise ValueError("preflight report is stale or not yet valid")
        if self.blockers:
            raise ValueError("preflight report has blockers")
        sections = self._sections_dict()
        for name in REQUIRED_SECTIONS:
            item = sections.get(name)
            if not isinstance(item, Mapping) or item.get("status") != ProbeStatus.PASS.value:
                raise ValueError(f"readonly section {name!r} is not PASS")
        for name in REQUIRED_ENABLE_PROBES:
            item = sections.get(f"enable_{name}")
            if not isinstance(item, Mapping) or item.get("status") != ProbeStatus.PASS.value:
                raise ValueError(f"enable section {name!r} is not PASS")
        if self.verdict is not PreflightVerdict.PASS_ENABLE_ALLOWED:
            raise ValueError("preflight verdict does not allow Enable")
        # Recompute the expected verdict from canonical sections.
        any_fail = any(
            not isinstance(sections.get(f"enable_{n}"), Mapping)
            or sections[f"enable_{n}"].get("status") != ProbeStatus.PASS.value
            for n in REQUIRED_ENABLE_PROBES
        )
        if any_fail:
            raise ValueError("recomputed preflight verdict is FAIL")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "M5PreflightReport":
        if not isinstance(value, Mapping):
            raise ValueError("preflight report must be a mapping")
        sections = value.get("sections")
        if not isinstance(sections, Mapping):
            raise ValueError("preflight sections must be a mapping")
        missing = [name for name in REQUIRED_SECTIONS if name not in sections]
        if missing:
            raise ValueError(f"preflight report missing sections: {missing}")
        try:
            verdict = PreflightVerdict(str(value["verdict"]))
        except Exception as exc:
            raise ValueError("invalid preflight verdict") from exc
        generated = int(value["generated_at_ns"])
        expires = int(value.get("expires_at_ns", generated))
        if expires < generated:
            raise ValueError("preflight expiry precedes generation")
        return cls(
            schema_version=str(value.get("schema_version", "m5-preflight/v1")),
            generated_at_ns=generated,
            expires_at_ns=expires,
            station_id=str(value["station_id"]),
            config_hash=str(value["config_hash"]),
            sections=dict(sections),
            verdict=verdict,
            blockers=tuple(str(v) for v in value.get("blockers", ())),
        )

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


class M5PreflightRunner:
    """Collects injected probes; default maximum verdict is PASS_READONLY."""

    def __init__(
        self,
        *,
        station_id: str,
        config_hash: str,
        readonly_probes: Mapping[str, Callable[[], Any]],
        enable_probes: Mapping[str, Callable[[], Any]] | None = None,
        clock_ns: Callable[[], int] | None = None,
        enable_ttl_ns: int = 60_000_000_000,
    ) -> None:
        self.station_id = station_id
        self.config_hash = config_hash
        self.readonly_probes = dict(readonly_probes)
        self.enable_probes = dict(enable_probes or {})
        self._clock_ns = clock_ns or (lambda: time.monotonic_ns())
        if isinstance(enable_ttl_ns, bool) or not isinstance(enable_ttl_ns, int) or enable_ttl_ns <= 0:
            raise ValueError("enable_ttl_ns must be a positive integer")
        self.enable_ttl_ns = enable_ttl_ns

    def _run_probes(
        self,
        names: Tuple[str, ...],
        probes: Mapping[str, Callable[[], Any]],
        sections: dict,
        blockers: list,
        prefix: str = "",
    ) -> None:
        for name in names:
            key = f"{prefix}{name}"
            probe = probes.get(name)
            if probe is None:
                sections[key] = {"status": "MISSING", "details": {}}
                blockers.append(f"missing_probe:{key}")
                continue
            try:
                sections[key] = _normalise_probe_result(probe())
            except Exception as exc:  # noqa: BLE001
                sections[key] = {"status": "PROBE_ERROR", "details": {"reason": type(exc).__name__}}
                blockers.append(f"probe_error:{key}")
                continue
            if sections[key]["status"] != ProbeStatus.PASS.value:
                blockers.append(f"probe_failed:{key}")

    def run(self, *, allow_enable: bool = False, operator_ack: bool = False) -> M5PreflightReport:
        sections: dict[str, Any] = {}
        blockers: list[str] = []
        self._run_probes(REQUIRED_SECTIONS, self.readonly_probes, sections, blockers)
        generated = int(self._clock_ns())
        verdict = PreflightVerdict.FAIL if blockers else PreflightVerdict.PASS_READONLY
        if allow_enable:
            self._run_probes(REQUIRED_ENABLE_PROBES, self.enable_probes, sections, blockers, "enable_")
            if operator_ack is not True:
                blockers.append("operator_ack_missing")
            if blockers:
                verdict = PreflightVerdict.FAIL
            else:
                verdict = PreflightVerdict.PASS_ENABLE_ALLOWED
        return M5PreflightReport(
            schema_version="m5-preflight/v1",
            generated_at_ns=generated,
            expires_at_ns=generated + self.enable_ttl_ns,
            station_id=self.station_id,
            config_hash=self.config_hash,
            sections=sections,
            verdict=verdict,
            blockers=tuple(blockers),
        )


__all__ = [
    "M5PreflightReport",
    "M5PreflightRunner",
    "PreflightVerdict",
    "ProbeResult",
    "ProbeStatus",
    "REQUIRED_ENABLE_PROBES",
    "REQUIRED_SECTIONS",
]
