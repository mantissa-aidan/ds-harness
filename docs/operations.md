# Running and operating tasks

## Inference endpoint

The client uses the OpenAI-style **Chat Completions** request shape: `model`, `messages`,
and a configurable output-token limit, with text at `choices[0].message.content`.
The configured base URL must include the provider's API prefix (commonly `/v1`), but
not `/chat/completions`; the client appends that path. See the
[official Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

```powershell
$env:DS_LLM_BASE_URL = 'https://approved-inference.example/v1'
$env:DS_LLM_MODEL = 'your-model-or-deployment-name'
$env:DS_LLM_API_KEY = 'populate-from-your-secret-manager'
```

For a local OpenAI-compatible server, use an address such as
`http://127.0.0.1:8000/v1`; a key can be omitted if that server does not require one.
Remote endpoints require HTTPS. TLS verification is enabled. Redirects are rejected.
Environment-based enterprise proxies and certificate configuration use httpx defaults.
The `.env.example` file is documentation; `.env` files are not automatically loaded.

Compatibility switches in `llm`:

| Setting | Default | Use |
| --- | --- | --- |
| `json_mode` | `true` | Set `false` if the server rejects `response_format`; JSON is still validated locally |
| `token_parameter` | `max_tokens` | Set `max_completion_tokens` for endpoints requiring that name |
| `auth_header` | `Authorization` | Set `api-key` for a gateway using that header |
| `timeout_seconds` | `60` | Timeout per HTTP operation |
| `retries` | `2` | Retry transient network errors, HTTP 429 and 5xx |
| `max_calls` | `10` | Global run budget, including failed/retried/repair requests; persists across resume |
| `max_prompt_chars` | `60000` | Reject oversized contexts before sending |
| `max_output_tokens` | `4096` | Output budget per request |

No model name is hard-coded. Providers must implement this request/response contract;
"OpenAI-compatible" alone does not guarantee all parameters work. Legacy Azure routes
requiring query-string API versions are not implemented; use a compatible `/v1` endpoint
or your enterprise gateway. Streaming, tool-calling wire formats and the Responses API
are not required. There is no dollar-cost estimate because provider prices differ.

## Explicit and autonomous plans

```powershell
# Execute the reviewed analyses; the LLM can interpret computed evidence.
ds-harness run --config config.local.yaml --task examples/tasks/customer_intelligence.yaml --output runs/customer-review

# Let the LLM choose tools and parameters for the scoped business brief.
ds-harness run --config config.local.yaml --task examples/tasks/autonomous.yaml --output runs/customer-discovery

# Run an explicit plan with no inference calls.
ds-harness run --config config.local.yaml --task examples/tasks/customer_intelligence.yaml --output runs/offline --offline
```

Offline mode requires explicit `analyses`. No synthetic LLM responses are substituted
for a missing endpoint. An unavailable planning endpoint fails the run and can be
resumed. An unavailable narrative endpoint leaves computed results intact and records
a warning. Neither condition silently changes the requested analysis.

For a new task, supply `name`, `objective`, `decision`, `success_criteria`, `datasets`
and `as_of`. Set `feature_allowlist` only for predictive inputs verified to precede
the outcome. For repeated entities use `split: group` and a customer/entity split
column; for temporal evaluation use `split: time` and the observation-date split
column. Classification targets must encode positive outcomes as `1`, negatives as `0`.

Generate schemas for editors or integration:

```powershell
ds-harness schema config
ds-harness schema task
```

## Checkpoints and status

```powershell
ds-harness status runs/customer-review
ds-harness run --config config.local.yaml --task examples/tasks/customer_intelligence.yaml --output runs/customer-review --resume
```

Use the same `--offline` flag on resume if it was used initially. Resume verifies
config/task signatures, implementation fingerprint, package versions and artifact
hashes, then reuses local snapshots and completed analyses. It never queries sources
again after the data-understanding checkpoint is complete. Environment variables
can change to repair connectivity/authentication, but changing the YAML or code
requires a new output directory. Finished runs are idempotent on resume.

If a run fails before its extraction checkpoint, its reads may repeat. SQL result sets
are not assumed to be immutable until snapshotted. If a source changes afterward,
start a new run to analyze the new state. A blocked or inconclusive run remains final;
correct the data definitions or source data and run again in a new directory.

| Exit | State | Meaning |
| --- | --- | --- |
| 0 | `completed` | All technical analysis criteria passed; business impact remains unproven |
| 2 | `failed` / validation error | Configuration, inference or execution failed; inspect `state.json` |
| 3 | `blocked` | Data contracts or cross-dataset integrity failed; remediation report exists |
| 4 | `needs_review` | An analysis failed its requirements or acceptance criteria; report exists |

Each directory has a `.run.lock` containing the process ID. Concurrent writers are
rejected. After an abrupt process kill, confirm that process is no longer running
before manually removing only that run's `.run.lock`, then resume. The harness does
not guess whether another process has finished.

Artifacts use atomic file replacement and SHA-256 tracking. Hashes detect accidental
changes; they are not tamper-proof signatures or an access-control system. Event logs
record decisions and request usage, not raw HTTP payloads or secrets. Failed plans
are described by validation errors rather than saved as raw model output.

## Testing and rollout

```powershell
python -m unittest discover -s tests -v
ds-harness demo --output demo-workspace
```

The demo deliberately has conflicting warehouse customer keys and exact duplicated
orders. It falls back to a clean operational customer extract, repairs permitted exact
duplicates, reconciles purchase totals, creates descriptive results and evaluates a
synthetic churn signal. It does not require SQL Server, an ODBC driver or an API key.
Its predictive performance says nothing about your actual business data.

The suite exercises real SQLite reads, local file sources, quality/preparation gates,
split behavior, baseline evaluation, mocked compatible inference and crash recovery.
The CI workflow runs Windows and Linux across Python 3.11–3.13. A live SQL Server and
your organization's inference service still need an integration smoke test using
`catalog` and a small, read-only task. This first version has no scheduler or web UI;
invoke the CLI from your existing job scheduler and preserve its exit code.
