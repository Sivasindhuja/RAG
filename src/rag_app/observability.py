from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List


@dataclass
class RequestTrace:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timings_ms: Dict[str, float] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str, **meta: Any) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        except Exception as exc:  # noqa: BLE001
            self.errors.append(f"{name}: {type(exc).__name__}: {exc}")
            self.events.append({"stage": name, "status": "error", "error": str(exc), **meta})
            raise
        finally:
            elapsed = (time.perf_counter() - start) * 1000.0
            self.timings_ms[name] = round(elapsed, 3)
            if not any(e.get("stage") == name and e.get("status") == "error" for e in self.events):
                self.events.append({"stage": name, "status": "ok", "ms": round(elapsed, 3), **meta})

    def add_detail(self, key: str, value: Any) -> None:
        self.details[key] = value

    def total_ms(self) -> float:
        return round(sum(self.timings_ms.values()), 3) if self.timings_ms else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "timings_ms": dict(self.timings_ms),
            "events": list(self.events),
            "errors": list(self.errors),
            "details": dict(self.details),
            "total_ms": self.total_ms(),
        }
