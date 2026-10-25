"""Command-line interface; identical commands work in PowerShell and POSIX shells."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from ds_harness import __version__
from ds_harness.config import HarnessConfig, SelectQuery, Task, load_yaml
from ds_harness.connectors import Connector
from ds_harness.demo import create_demo
from ds_harness.errors import HarnessError
from ds_harness.runner import Runner, required_datasets
from ds_harness.sql import compile_select

EXIT_CODES = {"completed": 0, "failed": 2, "blocked": 3, "needs_review": 4}


def parser():
    root = argparse.ArgumentParser(description="Autonomous read-only CRISP-DM analysis harness")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run or resume a CRISP-DM task")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--task", type=Path, required=True)
    run.add_argument("--output", type=Path)
    run.add_argument("--offline", action="store_true", help="Use explicit analyses without LLM calls")
    run.add_argument("--resume", action="store_true", help="Reuse verified checkpoints in --output")
    validate = commands.add_parser("validate", help="Validate configuration without connecting")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--task", type=Path, required=True)
    catalog = commands.add_parser("catalog", help="Inspect approved SQL tables/views (metadata only)")
    catalog.add_argument("--config", type=Path, required=True)
    demo = commands.add_parser("demo", help="Generate synthetic data and run a complete offline example")
    demo.add_argument("--output", type=Path, default=Path("demo-workspace"))
    init = commands.add_parser("init-demo", help="Generate an editable synthetic example without running")
    init.add_argument("--output", type=Path, default=Path("demo-workspace"))
    status = commands.add_parser("status", help="Inspect a run checkpoint")
    status.add_argument("run_directory", type=Path)
    schema = commands.add_parser("schema", help="Print a JSON schema for editors and integrations")
    schema.add_argument("kind", choices=["config", "task"])
    return root


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    try:
        if arguments.command == "schema":
            model = HarnessConfig if arguments.kind == "config" else Task
            print(json.dumps(model.model_json_schema(), indent=2))
            return 0
        if arguments.command == "status":
            state = json.loads((arguments.run_directory / "state.json").read_text(encoding="utf-8"))
            print(json.dumps({k: state[k] for k in ["status", "phases", "llm_calls", "updated_at"]}, indent=2))
            return EXIT_CODES.get(state["status"], 0)
        if arguments.command == "catalog":
            config = load_yaml(arguments.config, HarnessConfig)
            catalog = Connector(config, arguments.config.resolve().parent).catalog()
            print(json.dumps(catalog, indent=2))
            return 2 if any("error_type" in record for record in catalog.values()) else 0
        if arguments.command == "validate":
            config, task = load_yaml(arguments.config, HarnessConfig), load_yaml(arguments.task, Task)
            required_datasets(task, config)
            for dataset in config.datasets.values():
                for candidate in dataset.candidates:
                    source = config.sources[candidate.source]
                    if source.kind != "files":
                        compile_select(candidate.query or SelectQuery(table=candidate.table),
                                       source.allowed_tables, config.limits.max_rows + 1,
                                       "tsql" if source.kind == "sqlserver" else "sqlite")
            print("Configuration and task are valid. No connections were opened.")
            return 0
        if arguments.command in {"demo", "init-demo"}:
            config_path, task_path = create_demo(arguments.output)
            if arguments.command == "init-demo":
                print(f"Created {config_path} and {task_path}")
                return 0
            output = arguments.output / "runs" / "demo"
            runner = Runner(config_path, task_path, output, offline=True)
        else:
            if arguments.resume and arguments.output is None:
                raise HarnessError("--resume requires --output pointing to an existing run")
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            output = arguments.output or Path("runs") / timestamp
            runner = Runner(arguments.config, arguments.task, output,
                            offline=arguments.offline, resume=arguments.resume)
        print(f"Running CRISP-DM task; artifacts: {runner.store.path}", flush=True)
        state = runner.run()
        print(f"Status: {state['status']}\nReport: {runner.store.path / 'report.md'}")
        return EXIT_CODES[state["status"]]
    except ValidationError as exc:
        errors = [f"{'.'.join(map(str, e['loc']))}: {e['type']}" for e in exc.errors()]
        print("Invalid configuration:\n" + "\n".join(errors), file=sys.stderr)
        return 2
    except HarnessError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"Could not read configuration or artifacts ({type(exc).__name__})", file=sys.stderr)
        return 2
