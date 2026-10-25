# DS Harness

A runnable Python repository for autonomous, **read-only business and customer
intelligence**, organized around all six CRISP-DM phases. SQL Server is the primary
production connector. LLM calls use a configurable OpenAI-compatible inference endpoint.

Give it a business brief, approved data sources and data contracts. It profiles data,
tries equivalent fallback sources when quality fails, prepares local snapshots,
executes validated analyses, evaluates the evidence and writes a local decision-support
package. It does not modify databases or activate campaigns. Runs never pause for
human approval; failed gates produce a remediation report or a resumable failure.

## Quick start

Requires Python **3.11–3.13**; Python **3.12** is recommended and was used for local
validation. After cloning:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[sqlserver]"
ds-harness demo --output demo-workspace
```

On macOS/Linux, activate with `source .venv/bin/activate`. If PowerShell blocks script
activation, use `.venv\Scripts\python.exe -m pip ...` and
`.venv\Scripts\python.exe -m ds_harness ...` instead; no policy change is needed.
For the local demo, `python -m pip install -e .` is sufficient. SQL Server additionally
requires Microsoft's ODBC Driver 18, installed separately on the laptop.

Open `demo-workspace/runs/demo/report.md`. The demo runs without an API key or SQL Server.
It includes a broken warehouse customer extract, operational fallback, duplicate order
repair, cross-source revenue reconciliation, channel KPIs, RFM segments and held-out
churn evaluation. All demo records are synthetic.

## Bring it to the work laptop

The repository can be hosted on your organization's Git server, or transferred as the
provided `ds-harness.bundle` file. A bundle contains the commit and source files, not
local data, credentials or virtual environments. On the work laptop:

```powershell
git clone C:\path\to\ds-harness.bundle ds-harness
cd ds-harness
```

Then follow Quick start. To create a fresh bundle after your own commits:

```powershell
git bundle create ds-harness.bundle --all
```

No remote Git host is required for the bundle workflow. Install dependencies from
your approved package index or internal mirror. `requirements-tested.txt` records
the exact runtime dependencies used for validation on Python 3.12; it is optional:

```powershell
python -m pip install -r requirements-tested.txt
python -m pip install -e ".[sqlserver]"
```

## Connect your environment

1. Copy [examples/sqlserver.yaml](examples/sqlserver.yaml) to `config.local.yaml`.
2. Map its placeholder tables/columns to approved SQL Server reporting data. Define
   grain, time windows, keys, null/range/freshness rules and equivalent fallback sources.
3. Set the ODBC connection string and inference variables in your shell or secret manager.
4. Choose an explicit task, or let the LLM plan analyses from the scoped business brief.

```powershell
$env:DS_SQLSERVER_ODBC = 'Driver={ODBC Driver 18 for SQL Server};Server=tcp:your-server,1433;Database=your-database;Trusted_Connection=yes;Encrypt=yes;TrustServerCertificate=no;ApplicationIntent=ReadOnly;'
$env:DS_LLM_BASE_URL = 'https://your-approved-inference-host/v1'
$env:DS_LLM_MODEL = 'your-model-or-deployment-name'
# Populate DS_LLM_API_KEY from your secret manager if the endpoint requires a key.

ds-harness catalog --config config.local.yaml
ds-harness validate --config config.local.yaml --task examples/tasks/autonomous.yaml
ds-harness run --config config.local.yaml --task examples/tasks/autonomous.yaml --output runs/customer-discovery
```

Use a SQL principal with **SELECT-only permissions**. `ApplicationIntent=ReadOnly`
supports replica routing; it is not a substitute for database permissions.
Remove the sample fallback source if it is not available, or set its `DS_SOURCE_ODBC`
environment variable. Update the example task date and source query windows together.

The inference client uses `POST {base_url}/chat/completions`, optional JSON mode,
bounded retries, local schema validation and a persisted call budget. Model names are
not hard-coded. See [endpoint configuration](docs/operations.md#inference-endpoint).

## What is implemented

| Area | Included |
| --- | --- |
| CRISP-DM | Business brief, understanding, preparation, modeling, evaluation, local deployment/handoff |
| Data access | SQL Server via ODBC, SQLite demo, CSV and Parquet; bounded reads |
| SQL | Compiled parameterized SELECTs, explicit table allowlists, filters, joins, aggregates, query timeouts |
| Variable data quality | Source fallback, schema/type/null/range checks, key uniqueness, freshness, foreign keys, reconciliation |
| Preparation | Declared type conversion, string trimming, exact deduplication, explicit null-row removal, row-loss budgets |
| Analytics | Grouped KPIs, RFM segmentation, binary classification, regression |
| Evaluation | Train-only preprocessing, random/group/time splits, simple baselines, bootstrap uncertainty, acceptance thresholds |
| Autonomy | LLM tool selection and JSON repair; deterministic execution; no arbitrary generated code |
| Reproducibility | Local snapshots, fingerprints, atomic checkpoints, artifact verification, runtime capture, resume |
| Outputs | Markdown report, CSV results, JSON evidence, event log and monitoring/remediation handoff |

Raw records and category labels stay local. The LLM receives the business brief,
structural profiles and numeric aggregate evidence. Local snapshots can contain
confidential data; keep run directories in an approved location. Sensitivity flags
must be configured for your schema; name heuristics alone are not a PII inventory.

This is a working first version with maintained analytics tools. Complex enterprise
metrics belong in reviewed views or configured structured queries. It does not include
unrestricted SQL/Python agents, general forecasting, causal inference, trained-model
serving, a scheduler or a web UI. Business success requires validation beyond technical
model metrics. Detailed boundaries are in [the architecture guide](docs/architecture.md).

## Run artifacts

```text
runs/<run>/
  state.json                  # phase status, call budget and artifact hashes
  config.json, task.json       # configuration snapshots; no resolved environment secrets
  environment.json            # runtime versions and implementation fingerprint
  business_understanding.json
  data_understanding.json     # candidate decisions, profiles, quality evidence and provenance
  data_preparation.json       # transformations and cross-dataset integrity checks
  snapshots/, prepared/       # local data; excluded from Git
  plan.json, results.json
  analyses/*.csv
  evaluation.json
  interpretation.json         # optional, labeled LLM commentary
  events.jsonl
  handoff.json, report.md
```

```powershell
ds-harness status runs/customer-discovery
ds-harness run --config config.local.yaml --task examples/tasks/autonomous.yaml --output runs/customer-discovery --resume
python -m unittest discover -s tests -v
```

Resume uses the same configuration, task, implementation and dependency versions. New
data definitions or a new extraction require a new run directory. Exit codes distinguish
success (`0`), execution failure (`2`), blocked data quality (`3`), and inconclusive
analysis (`4`). See [operations and recovery](docs/operations.md).

## Documentation

- [Architecture, CRISP-DM phases, trust and evaluation](docs/architecture.md)
- [SQL Server setup, permissions and query examples](docs/sql-server.md)
- [Inference endpoints, task configuration, resume and operations](docs/operations.md)
- [Explicit customer intelligence task](examples/tasks/customer_intelligence.yaml)
- [Autonomous analysis brief](examples/tasks/autonomous.yaml)

The test suite covers SQLite and local file execution, SQL Server adapter behavior with
a mock driver, compatible inference requests, data quality, analytical evaluation and
failure recovery. Live SQL Server and endpoint integration must be checked in your
environment. CI is configured for Windows/Linux and Python 3.11–3.13.
