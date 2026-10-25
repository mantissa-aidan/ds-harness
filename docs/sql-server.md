# SQL Server onboarding

Install Microsoft **ODBC Driver 18 for SQL Server** on the work laptop, then install
the optional Python adapter:

```powershell
python -m pip install -e ".[sqlserver]"
```

The ODBC driver is an operating-system dependency; installing `pyodbc` does not install
it. On Linux, use your organization's approved driver installation process. See
[Microsoft's ODBC driver downloads](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server).

## Account and connection

Use a principal with **SELECT-only access on the approved reporting tables/views**.
It must not have write, DDL, stored-procedure execution, database-owner or server-admin
privileges. Database permissions enforce the access boundary even if application code
changes. Have the DBA provision the account; this repository contains no provisioning
commands and never changes grants.

For Windows integrated authentication:

```powershell
$env:DS_SQLSERVER_ODBC = 'Driver={ODBC Driver 18 for SQL Server};Server=tcp:server.example,1433;Database=CustomerAnalytics;Trusted_Connection=yes;Encrypt=yes;TrustServerCertificate=no;ApplicationIntent=ReadOnly;'
```

For SQL authentication, have a secret manager populate the same environment variable
with the approved ODBC string, including `UID` and `PWD`. Do not put credentials in YAML
or commit them. Standard ODBC authentication options remain available; the harness
passes the string to `pyodbc` without printing it.

`ApplicationIntent=ReadOnly` enables routing to readable replicas when configured; it
does **not** replace SELECT-only account permissions. Microsoft documents this as
[read-only routing](https://learn.microsoft.com/en-us/sql/connect/odbc/odbc-driver-support-for-high-availability-disaster-recovery).
The connector opens transactions without autocommit, fetches bounded results, rolls
back and closes connections. Every compiled application query is a SELECT.

If the preferred warehouse has an equivalent operational reporting source, configure
its own environment variable (`DS_SOURCE_ODBC` in the example). Otherwise remove the
operational source and fallback candidate. Fallback is not a requirement.

## Map your schema

Copy `examples/sqlserver.yaml` to `config.local.yaml`. Replace placeholder relation
names and logical columns with your actual reporting schema. Sources list exact
`allowed_tables`; there is no unrestricted discovery of the server. Prefer approved
views with understandable column names, one documented grain, stable windows and
appropriate indexes.

```powershell
ds-harness catalog --config config.local.yaml
ds-harness validate --config config.local.yaml --task examples/tasks/customer_intelligence.yaml
```

`catalog` runs metadata-only SELECTs against approved relations. `validate` performs
no connections. A valid YAML file does not imply the database objects exist or that
the account can access them.

For each dataset specify:

1. Business meaning, population, time window, grain, units and currency.
2. Ordered source candidates with matching definitions.
3. Required columns, expected types, null tolerances, keys, valid ranges and freshness.
4. Only local preparation you intend to allow and its maximum row-loss fraction.
5. Foreign-key relationships and any independent warehouse/source reconciliation.

Sample dates are fixed to make the demo reproducible. Update both the task `as_of`
and query date filters for your run. `as_of` is a UTC calendar date; freshness checks
are relative to its end. It is not automatically injected into queries. A freshness
check on ingestion time does not prove a historical feature/label dataset is correct.

## Structured query examples

`candidate.table` emits a bounded SELECT of the approved relation. For filters,
joins or server-side aggregates, use `candidate.query` instead:

```yaml
candidates:
  - source: warehouse
    query:
      table: analytics.orders
      columns: [order_id, customer_id, amount, order_date, channel]
      filters:
        - {column: order_date, op: gte, value: '2026-01-01'}
        - {column: order_date, op: lt, value: '2026-10-02'}
        - {column: status, op: eq, value: completed}
      order_by: [order_id]
```

For a large table, define a separate logical aggregate dataset and a matching contract:

```yaml
query:
  table: analytics.orders
  group_by: [channel]
  measures:
    - {column: amount, aggregate: sum, alias: revenue}
    - {column: '*', aggregate: count, alias: order_count}
  order_by: [channel]
```

Counts in subsequent local analysis refer to **aggregate rows**, not original orders.
Use `order_count` explicitly when the business metric needs the underlying count. The
small-group threshold also applies to extract rows; pre-aggregated data requires its
own upstream privacy policy, not an assumption that the local threshold protects it.

A join uses base alias `t` and explicit right-hand aliases:

```yaml
query:
  table: analytics.orders
  joins:
    - table: analytics.customers
      alias: c
      left_column: t.customer_id
      right_column: customer_id
      kind: inner
  columns: [t.order_id, t.amount, c.region]
```

Every joined table must be allowlisted. Declare the resulting grain and test key
uniqueness to detect join fan-out. Output column names must be unique. For complex
projections, custom expressions or renamed columns, use a reviewed reporting view.

Supported filters: `eq`, `ne`, `lt`, `lte`, `gt`, `gte`, `in` (1–100 values), `is_null`,
`not_null`. Filters are ANDed. Comparisons use bound parameters. Supported aggregate
functions: `count`, `sum`, `avg`, `min`, `max`. Supported joins: `inner`, `left` (up to four).
Ordering is ascending. There are no raw SQL fragments, stored procedures, cross-database
names, temp tables, dynamic functions, hints or arbitrary expressions.

The compiler supports simple ASCII SQL identifiers (letters, digits, underscore and
interior spaces), using bracket quoting. Use reporting views for unusual names. Only
`table` or `schema.table` relation names are accepted. Different databases use separate
configured connections and local reconciliation, rather than cross-database SQL.
