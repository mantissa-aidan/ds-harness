"""Resumable CRISP-DM orchestration with fail-closed quality gates."""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from pathlib import Path

import pandas as pd

from ds_harness import __version__
from ds_harness.agent import interpret, make_plan
from ds_harness.analysis import execute, validate_plan
from ds_harness.artifacts import RunStore, now
from ds_harness.config import HarnessConfig, Plan, Task, canonical_json, load_yaml
from ds_harness.connectors import Connector
from ds_harness.errors import HarnessError, LLMError, PolicyError
from ds_harness.llm import ChatClient
from ds_harness.quality import assess, fingerprint, integrity, prepare, profile
from ds_harness.report import render_report

PHASES = ["business_understanding", "data_understanding", "data_preparation",
          "modeling", "evaluation", "deployment"]
TERMINAL = {"completed", "needs_review", "blocked"}


def required_datasets(task: Task, config: HarnessConfig) -> list[str]:
    selected = set(task.datasets)
    if not selected <= config.datasets.keys():
        raise PolicyError("Task references an unknown dataset")
    while True:
        previous = selected.copy()
        for rel in config.relationships:
            pair = {rel.child, rel.parent}
            if pair & selected:
                selected |= pair
        for rec in config.reconciliations:
            pair = {rec.left, rec.right}
            if pair & selected:
                selected |= pair
        if selected == previous:
            return sorted(selected)


def environment() -> dict:
    dependencies = {}
    for package in ["pandas", "numpy", "pydantic", "httpx", "scikit-learn", "pyarrow", "PyYAML"]:
        dependencies[package] = importlib.metadata.version(package)
    implementation = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        implementation.update(path.name.encode())
        implementation.update(path.read_bytes())
    return {"python": platform.python_version(), "platform": platform.system(),
            "implementation_sha256": implementation.hexdigest(),
            "harness": __version__, "packages": dependencies}


class Runner:
    def __init__(self, config_path: Path, task_path: Path, output: Path, *,
                 offline: bool = False, resume: bool = False, transport=None):
        self.config_path = config_path.resolve()
        self.config: HarnessConfig = load_yaml(self.config_path, HarnessConfig)
        self.task: Task = load_yaml(task_path, Task)
        # Validate the date now, before any connection or inference is attempted.
        pd.Timestamp(self.task.as_of, tz="UTC")
        if offline:
            self.config.llm.mode = "offline"
        self.store = RunStore(output)
        self.resume = resume
        self.connector = Connector(self.config, self.config_path.parent)
        self.datasets = required_datasets(self.task, self.config)
        self.client = None if self.config.llm.mode == "offline" else ChatClient(
            self.config.llm, self._record_llm, transport=transport
        )
        self.signature = hashlib.sha256(canonical_json({
            "config": self.config.model_dump(mode="json"),
            "task": self.task.model_dump(mode="json"), "harness": __version__,
        }).encode()).hexdigest()

    def _record_llm(self, entry: dict):
        self.store.event(entry)
        if entry["event"] == "llm_attempt":
            self.store.state["llm_calls"] = entry["call"]
            self.store.save()

    def _initialize(self):
        store = self.store
        state_path = store.path / "state.json"
        if self.resume:
            if not state_path.exists():
                raise PolicyError("Cannot resume: state.json does not exist")
            store.state = store.read_json("state.json")
            if store.state.get("signature") != self.signature:
                raise PolicyError("Configuration/task/version changed; start a new run directory")
            store.verify()
            if store.read_json("environment.json") != environment():
                raise PolicyError("Runtime dependencies changed; start a new run directory")
            if self.client:
                self.client.calls = store.state.get("llm_calls", 0)
        else:
            if state_path.exists() or any(p.name != ".run.lock" for p in store.path.iterdir()):
                raise PolicyError("Output directory must be empty; use --resume for an existing run")
            store.state = {"version": 1, "signature": self.signature, "created_at": now(),
                           "status": "running", "llm_calls": 0, "artifacts": {},
                           "phases": {phase: "pending" for phase in PHASES}}
            store.json("environment.json", environment())
            store.json("config.json", self.config.model_dump(mode="json"))
            store.json("task.json", self.task.model_dump(mode="json"))
            store.save()

    def _selected_data(self) -> dict:
        store = self.store
        if store.state["phases"]["data_understanding"] == "completed":
            return store.read_json("data_understanding.json")
        decisions = {}
        for name in self.datasets:
            spec = self.config.datasets[name]
            record = {"selected": None, "attempts": []}
            for index, candidate in enumerate(spec.candidates):
                attempt = {"candidate": index, "source": candidate.source}
                try:
                    raw, provenance = self.connector.read(candidate)
                    attempt["profile"] = profile(raw, spec.contract)
                    attempt["raw_quality"] = assess(raw, spec.contract, self.task.as_of)
                    prepared, changes = prepare(raw, spec.contract, spec.preparation)
                    attempt["quality"] = assess(prepared, spec.contract, self.task.as_of)
                    attempt["preparation_preview"] = changes
                    attempt["provenance"] = provenance
                    if attempt["quality"]["passed"]:
                        store.frame(f"snapshots/{name}.parquet", raw)
                        attempt["snapshot_fingerprint"] = fingerprint(raw)
                        record["selected"] = index
                except HarnessError as exc:
                    attempt["error"] = str(exc)
                record["attempts"].append(attempt)
                store.event({"event": "candidate_evaluated", "dataset": name,
                             "candidate": index, "accepted": record["selected"] is not None})
                if record["selected"] is not None:
                    break
            decisions[name] = record
        store.json("data_understanding.json", decisions)
        if all(record["selected"] is not None for record in decisions.values()):
            store.done("data_understanding")
        else:
            store.state["phases"]["data_understanding"] = "blocked"
            store.save()
        return decisions

    def _prepare_data(self) -> tuple[dict, dict[str, pd.DataFrame]]:
        store = self.store
        if store.state["phases"]["data_preparation"] == "completed":
            return store.read_json("data_preparation.json"), {
                name: pd.read_parquet(store.location(f"prepared/{name}.parquet"))
                for name in self.datasets
            }
        frames, evidence = {}, {}
        for name in self.datasets:
            spec = self.config.datasets[name]
            raw = pd.read_parquet(store.location(f"snapshots/{name}.parquet"))
            frame, changes = prepare(raw, spec.contract, spec.preparation)
            changes["quality"] = assess(frame, spec.contract, self.task.as_of)
            changes["fingerprint"] = fingerprint(frame)
            frames[name], evidence[name] = frame, changes
            store.frame(f"prepared/{name}.parquet", frame)
        checks = integrity(frames, self.config)
        preparation = {"datasets": evidence, "integrity": checks}
        store.json("data_preparation.json", preparation)
        passed = checks["passed"] and all(r["quality"]["passed"] for r in evidence.values())
        if passed:
            store.done("data_preparation")
        else:
            store.state["phases"]["data_preparation"] = "blocked"
            store.save()
        return preparation, frames

    def _model(self, frames: dict[str, pd.DataFrame]) -> dict:
        store = self.store
        if "plan.json" in store.state["artifacts"]:
            plan = Plan.model_validate(store.read_json("plan.json"))
            validate_plan(plan, frames, self.config, self.task)
        else:
            plan = make_plan(self.config, self.task, frames, self.client)
            store.json("plan.json", plan.model_dump(mode="json"))
            store.save()
        results = store.read_json("results.json") if "results.json" in store.state["artifacts"] else {}
        for spec in plan.analyses:
            if spec.id in results:
                continue
            try:
                result, table = execute(frames[spec.dataset], spec, self.config, self.task)
                store.csv(f"analyses/{spec.id}.csv", table)
                result["dataset"] = spec.dataset
            except HarnessError as exc:
                result = {"kind": spec.kind, "dataset": spec.dataset, "passed": False,
                          "error": str(exc), "limitations": ["Analysis did not pass its input checks."]}
            except (ValueError, TypeError) as exc:
                result = {"kind": spec.kind, "dataset": spec.dataset, "passed": False,
                          "error": f"Analysis could not be computed ({type(exc).__name__})",
                          "limitations": ["Inspect data definitions before attempting a new run."]}
            results[spec.id] = result
            store.json("results.json", results)
            store.event({"event": "analysis_completed", "analysis": spec.id, "passed": result["passed"]})
            store.save()
        store.done("modeling")
        return results

    def _finish(self, selection: dict, preparation: dict, results: dict, status: str, issues: list[str]):
        store = self.store
        evaluation = {"passed": status == "completed", "issues": issues,
                      "business_impact": "not established", "external_actions": "none",
                      "analysis_acceptance": {k: v.get("passed", False) for k, v in results.items()}}
        store.json("evaluation.json", evaluation)
        interpretation = None
        if results and self.client:
            if "interpretation.json" in store.state["artifacts"]:
                interpretation = store.read_json("interpretation.json")
            else:
                try:
                    interpretation = interpret(self.task, results, self.client).model_dump(mode="json")
                    store.json("interpretation.json", interpretation)
                except LLMError as exc:
                    store.event({"event": "interpretation_unavailable", "reason": str(exc)})
                    evaluation["issues"].append("LLM interpretation unavailable; computed evidence is intact.")
                    store.json("evaluation.json", evaluation)
        store.done("evaluation")
        handoff = {
            "status": status, "purpose": "local decision-support package", "external_writes": False,
            "next_actions": issues or [
                "Have a business owner validate the metric definitions and operational usefulness.",
                "Measure proposed interventions with a prospective controlled experiment.",
            ],
            "monitoring": [
                "Re-run freshness, schema, key, missingness and reconciliation contracts each run.",
                "Compare population and feature distributions before using findings on new data.",
                "Re-evaluate predictive performance on newly matured labels; do not reuse the holdout.",
            ],
            "reproduce": "Use the same task/config/runtime; resume reuses the hashed local snapshots.",
            "business_criteria": self.task.success_criteria,
            "owners": "Assign business, data and operational owners before acting on findings.",
        }
        store.json("handoff.json", handoff)
        report_state = {**store.state, "status": status}
        store.text("report.md", render_report(self.task, report_state, selection, preparation,
                                               results, evaluation, interpretation))
        store.state["status"] = status
        store.done("deployment")
        return store.state

    def run(self) -> dict:
        store = self.store
        with store.lock():
            self._initialize()
            if store.state["status"] in TERMINAL:
                return store.state
            store.state["status"] = "running"
            store.state.pop("failure", None)
            store.save()
            try:
                if store.state["phases"]["business_understanding"] != "completed":
                    store.json("business_understanding.json", {
                        "objective": self.task.objective, "decision": self.task.decision,
                        "success_criteria": self.task.success_criteria,
                        "hypotheses": self.task.hypotheses, "constraints": self.task.constraints,
                        "as_of": self.task.as_of, "datasets": self.datasets,
                        "autonomy": "read_only", "deployment": "local report and handoff only",
                    })
                    store.done("business_understanding")
                selection = self._selected_data()
                missing = [name for name, record in selection.items() if record["selected"] is None]
                if missing:
                    return self._finish(selection, {}, {}, "blocked", [
                        "No contract-valid source for: " + ", ".join(missing),
                        "Repair source access or data contracts, then start a new run.",
                    ])
                preparation, frames = self._prepare_data()
                if store.state["phases"]["data_preparation"] == "blocked":
                    return self._finish(selection, preparation, {}, "blocked", [
                        "Cross-dataset integrity or prepared-data contracts failed.",
                        "Resolve orphan keys, reconciliation differences or grain mismatches before analysis.",
                    ])
                results = self._model(frames)
                failures = [key for key, result in results.items() if not result.get("passed")]
                status = "needs_review" if failures else "completed"
                issues = ["Analysis acceptance not established: " + ", ".join(failures)] if failures else []
                return self._finish(selection, preparation, results, status, issues)
            except Exception as exc:
                store.state["status"] = "failed"
                message = str(exc) if isinstance(exc, HarnessError) else type(exc).__name__
                store.state["failure"] = message
                store.event({"event": "run_failed", "reason": message})
                store.save()
                if isinstance(exc, HarnessError):
                    raise
                raise HarnessError(f"Run failed ({type(exc).__name__}); see state.json") from None
