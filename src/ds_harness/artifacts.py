"""Atomic checkpoints, artifact integrity, and a local append-only event log."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ds_harness.config import canonical_json
from ds_harness.errors import PolicyError


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunStore:
    def __init__(self, path: Path):
        self.path = path.resolve()
        self.state = {}

    def location(self, name: str) -> Path:
        target = (self.path / name).resolve()
        if not target.is_relative_to(self.path):
            raise PolicyError("Artifact path escapes the run directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    @contextmanager
    def lock(self):
        self.path.mkdir(parents=True, exist_ok=True)
        lock = self.path / ".run.lock"
        try:
            handle = lock.open("x", encoding="utf-8")
        except FileExistsError:
            raise PolicyError("Run is locked; see docs/operations.md for crash recovery") from None
        try:
            with handle:
                handle.write(str(os.getpid()))
            yield
        finally:
            lock.unlink(missing_ok=True)

    def _remember(self, name: str):
        self.state.setdefault("artifacts", {})[name] = sha256(self.location(name))

    def text(self, name: str, value: str):
        path = self.location(name)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
        self._remember(name)

    def json(self, name: str, value):
        self.text(name, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    def read_json(self, name: str):
        return json.loads(self.location(name).read_text(encoding="utf-8"))

    def frame(self, name: str, frame: pd.DataFrame):
        path = self.location(name)
        temporary = path.with_suffix(".tmp.parquet")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        self._remember(name)

    def csv(self, name: str, frame: pd.DataFrame):
        # Prevent spreadsheet formula execution when a categorical value is opened in Excel.
        safe = frame.copy()
        for column in safe.select_dtypes(include=["object", "string"]).columns:
            safe[column] = safe[column].map(
                lambda value: "'" + value if isinstance(value, str)
                and value.lstrip().startswith(("=", "+", "-", "@")) else value
            )
        self.text(name, safe.to_csv(index=False))

    def event(self, event: dict):
        entry = {"time": now(), **event}
        with self.location("events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(entry) + "\n")

    def save(self):
        self.state["updated_at"] = now()
        target = self.location("state.json")
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(target)

    def verify(self):
        for name, digest in self.state.get("artifacts", {}).items():
            path = self.location(name)
            if not path.is_file() or sha256(path) != digest:
                raise PolicyError(f"Run artifact is missing or changed: {name}")

    def done(self, phase: str):
        self.state.setdefault("phases", {})[phase] = "completed"
        self.event({"event": "phase_completed", "phase": phase})
        self.save()
