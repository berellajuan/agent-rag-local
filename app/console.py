from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StepTracer:
    """Small educational console tracer.

    This is intentionally simple: the goal is not production logging, the goal is
    to teach what the app is doing at each stage when run from PowerShell.
    """

    enabled: bool = True

    def title(self, text: str) -> None:
        if self.enabled:
            print(f"\n=== {text} ===")

    def step(self, text: str) -> None:
        if self.enabled:
            print(f"[paso] {text}")

    def detail(self, key: str, value: Any) -> None:
        if self.enabled:
            print(f"       {key}: {value}")

    def concept(self, text: str) -> None:
        if self.enabled:
            print(f"[concepto] {text}")

    def result(self, text: str) -> None:
        if self.enabled:
            print(f"[resultado] {text}")
