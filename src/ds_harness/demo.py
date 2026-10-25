"""A reproducible synthetic warehouse/source-system fixture with deliberate quality faults."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ds_harness.errors import PolicyError


def create_demo(root: Path) -> tuple[Path, Path]:
    root = root.resolve()
    if root.exists() and any(root.iterdir()):
        raise PolicyError("Demo directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    count = 400
    inactivity = rng.integers(0, 180, count)
    tickets = rng.poisson(2, count)
    probability = 1 / (1 + np.exp(-((inactivity - 90) / 16 + (tickets - 2) * 0.8)))
    customers = pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(count)],
        "region": rng.choice(["north", "south", "east", "west"], count),
        "tenure_months": rng.integers(1, 72, count),
        "monthly_value": rng.uniform(20, 250, count).round(2),
        "days_since_login": inactivity,
        "support_tickets": tickets,
        "churned": rng.binomial(1, probability),
        "email": [f"synthetic{i}@example.test" for i in range(count)],
        "updated_at": "2026-10-01",
    })
    # Conflicting rows at the same key cannot be fixed by exact deduplication.
    corrupted = pd.concat([customers, customers.iloc[[0]].assign(monthly_value=999)], ignore_index=True)
    n_orders = 2400
    orders = pd.DataFrame({
        "order_id": [f"O{i:06d}" for i in range(n_orders)],
        "customer_id": rng.choice(customers["customer_id"], n_orders),
        "order_date": (pd.Timestamp("2026-10-01") - pd.to_timedelta(
            rng.integers(0, 365, n_orders), unit="D")).strftime("%Y-%m-%d"),
        "amount": rng.gamma(2, 60, n_orders).round(2),
        "channel": rng.choice(["web", "store", "partner"], n_orders),
    })
    warehouse_orders = pd.concat([orders, orders.iloc[:4]], ignore_index=True)
    # Fixture creation is the only code path that creates a database, in a new local directory.
    with closing(sqlite3.connect(root / "demo.sqlite")) as conn, conn:
        corrupted.to_sql("warehouse_customers", conn, index=False)
        customers.to_sql("source_customers", conn, index=False)
        warehouse_orders.to_sql("warehouse_orders", conn, index=False)
        orders.to_sql("source_orders", conn, index=False)
    customer_contract = {
        "primary_key": ["customer_id"], "min_rows": 100,
        "freshness_column": "updated_at", "max_age_days": 7,
        "columns": {
            "customer_id": {"dtype": "string", "max_null_fraction": 0},
            "region": {"dtype": "string", "max_null_fraction": 0},
            "tenure_months": {"dtype": "integer", "minimum": 0, "max_null_fraction": 0},
            "monthly_value": {"dtype": "number", "minimum": 0, "max_null_fraction": 0},
            "days_since_login": {"dtype": "integer", "minimum": 0, "max_null_fraction": 0},
            "support_tickets": {"dtype": "integer", "minimum": 0, "max_null_fraction": 0},
            "churned": {"dtype": "integer", "allowed_values": [0, 1], "max_null_fraction": 0},
            "email": {"dtype": "string", "sensitive": True},
            "updated_at": {"dtype": "datetime", "max_null_fraction": 0},
        },
    }
    order_contract = {
        "primary_key": ["order_id"], "min_rows": 100,
        "freshness_column": "order_date", "max_age_days": 7,
        "columns": {
            "order_id": {"dtype": "string", "max_null_fraction": 0},
            "customer_id": {"dtype": "string", "max_null_fraction": 0},
            "order_date": {"dtype": "datetime", "max_null_fraction": 0},
            "amount": {"dtype": "number", "minimum": 0, "max_null_fraction": 0},
            "channel": {"dtype": "string", "max_null_fraction": 0},
        },
    }
    config = {
        "version": 1, "llm": {"mode": "offline"},
        "sources": {"warehouse": {"kind": "sqlite", "root": "demo.sqlite",
                                   "allowed_tables": ["warehouse_customers", "warehouse_orders"]},
                    "operational": {"kind": "sqlite", "root": "demo.sqlite",
                                     "allowed_tables": ["source_customers", "source_orders"]}},
        "datasets": {
            "customers": {"description": "One row per customer; synthetic pre-outcome features.",
                          "candidates": [{"source": "warehouse", "table": "warehouse_customers"},
                                         {"source": "operational", "table": "source_customers"}],
                          "contract": customer_contract},
            "orders": {"description": "One row per purchase; positive amount, one currency.",
                       "candidates": [{"source": "warehouse", "table": "warehouse_orders"}],
                       "contract": order_contract},
            "source_orders": {"description": "Independent operational reconciliation extract.",
                              "candidates": [{"source": "operational", "table": "source_orders"}],
                              "contract": order_contract},
        },
        "relationships": [{"child": "orders", "child_columns": ["customer_id"],
                           "parent": "customers", "parent_columns": ["customer_id"]}],
        "reconciliations": [{"left": "orders", "right": "source_orders", "metric": "sum",
                             "left_column": "amount", "right_column": "amount",
                             "relative_tolerance": 0.001}],
    }
    features = ["tenure_months", "monthly_value", "days_since_login", "support_tickets", "region"]
    task = {
        "name": "customer_intelligence",
        "objective": "Understand customer value and purchase behavior, and evaluate a churn baseline.",
        "decision": "Prioritize a retention experiment and investigate source data quality.",
        "success_criteria": ["Reconcile purchase revenue between warehouse and operational systems.",
                             "Produce actionable descriptive segments and evaluate churn discrimination."],
        "datasets": ["customers", "orders"], "as_of": "2026-10-01",
        "hypotheses": ["Inactivity and support interactions may help predict subsequent churn."],
        "constraints": ["Read-only analysis; no customer outreach or deployment."],
        "feature_allowlist": {"customers": features},
        "analyses": [
            {"id": "revenue_by_channel", "kind": "kpi", "dataset": "orders",
             "group_by": ["channel"], "measures": ["amount"]},
            {"id": "customer_segments", "kind": "rfm", "dataset": "orders",
             "customer_id": "customer_id", "date_column": "order_date",
             "amount_column": "amount", "transaction_id": "order_id"},
            {"id": "churn_baseline", "kind": "classification", "dataset": "customers",
             "target": "churned", "features": features, "min_improvement": 0.05},
        ],
    }
    config_path, task_path = root / "config.yaml", root / "task.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    task_path.write_text(yaml.safe_dump(task, sort_keys=False), encoding="utf-8")
    return config_path, task_path
