"""Evidence-based contracts, integrity checks and conservative local preparation."""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

from ds_harness.config import Contract, HarnessConfig, Preparation
from ds_harness.errors import PolicyError

SENSITIVE_NAME = re.compile(
    r"(^|_)(email|phone|mobile|address|name|ssn|dob|password|token|secret)($|_)", re.I
)


def sensitive_columns(frame: pd.DataFrame, contract: Contract) -> set[str]:
    return {
        c for c in frame.columns
        if SENSITIVE_NAME.search(c) or (c in contract.columns and contract.columns[c].sensitive)
    }


def fingerprint(frame: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(frame, index=True).values.tobytes()
    schema = "|".join(f"{c}:{t}" for c, t in frame.dtypes.items()).encode()
    return hashlib.sha256(schema + hashed).hexdigest()


def profile(frame: pd.DataFrame, contract: Contract) -> dict:
    """Only structural metadata; no samples, category values, extrema or identifiers."""
    sensitive = sensitive_columns(frame, contract)
    return {
        "rows": len(frame),
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "columns": {
            str(c): {
                "dtype": str(frame[c].dtype),
                "null_fraction": float(frame[c].isna().mean()) if len(frame) else 0.0,
                "distinct_count": int(frame[c].nunique()),
                "sensitive": c in sensitive,
            }
            for c in frame.columns
        },
    }


def coerce(series: pd.Series, dtype: str) -> pd.Series:
    if dtype == "string":
        return series.astype("string")
    if dtype == "datetime":
        return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
    if dtype in {"number", "integer"}:
        result = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        if dtype == "integer":
            result = result.where(result.isna() | (result % 1 == 0))
        return result
    # Never turn a non-empty string such as 'false' into True via astype(bool).
    mapping = {"true": True, "false": False, "1": True, "0": False}
    return series.astype("string").str.lower().map(mapping).astype("boolean")


def prepare(frame: pd.DataFrame, contract: Contract, policy: Preparation) -> tuple[pd.DataFrame, dict]:
    result = frame.copy()
    events = []
    if policy.trim_strings:
        for column in result.select_dtypes(include=["object", "string"]).columns:
            if result[column].dropna().map(lambda value: isinstance(value, str)).all():
                before = result[column].copy()
                result[column] = result[column].astype("string").str.strip().replace("", pd.NA)
                changed = (before.fillna("") != result[column].fillna(""))
                changes = int((changed | (before.isna() != result[column].isna())).sum())
                if changes:
                    events.append({"operation": "trim_strings", "column": column, "cells": changes})
    for column, rule in contract.columns.items():
        if column in result and rule.dtype:
            before_null = result[column].isna()
            result[column] = coerce(result[column], rule.dtype)
            invalid = int((result[column].isna() & ~before_null).sum())
            events.append({"operation": "coerce", "column": column, "invalid_cells": invalid})
            # Invalid parsing is evidence of corruption, not authorized missingness.
            if invalid:
                raise PolicyError(f"Type conversion failed for {column}: {invalid} non-null cells")
    if policy.drop_exact_duplicates:
        before = len(result)
        result = result.drop_duplicates()
        events.append({"operation": "drop_exact_duplicates", "rows": before - len(result)})
    if policy.drop_null_rows:
        if not set(policy.drop_null_rows) <= set(result.columns):
            raise PolicyError("Null-row removal references missing columns")
        before = len(result)
        result = result.dropna(subset=policy.drop_null_rows)
        events.append({"operation": "drop_null_rows", "rows": before - len(result)})
    dropped = len(frame) - len(result)
    fraction = dropped / max(1, len(frame))
    if fraction > policy.max_drop_fraction:
        raise PolicyError("Preparation exceeded max_drop_fraction; candidate rejected")
    if contract.primary_key and set(contract.primary_key) <= set(result):
        result = result.sort_values(contract.primary_key, kind="stable")
        events.append({"operation": "sort_primary_key", "columns": contract.primary_key})
    return result.reset_index(drop=True), {
        "input_rows": len(frame), "output_rows": len(result), "dropped_fraction": fraction,
        "events": events,
    }


def assess(frame: pd.DataFrame, contract: Contract, as_of: str) -> dict:
    checks = []

    def check(code: str, passed: bool, **evidence):
        checks.append({"check": code, "passed": bool(passed), **evidence})

    check("minimum_rows", len(frame) >= contract.min_rows, rows=len(frame), minimum=contract.min_rows)
    for column, rule in contract.columns.items():
        if column not in frame:
            check("column_present", not rule.required, column=column)
            continue
        series = frame[column]
        null_fraction = float(series.isna().mean()) if len(frame) else 1.0
        check("null_fraction", null_fraction <= rule.max_null_fraction,
              column=column, observed=null_fraction, maximum=rule.max_null_fraction)
        if rule.dtype:
            converted = coerce(series, rule.dtype)
            invalid = int((series.notna() & converted.isna()).sum())
            check("type_valid", invalid == 0, column=column, invalid_cells=invalid)
        else:
            converted = series
        for bound_name, bound in (("minimum", rule.minimum), ("maximum", rule.maximum)):
            if bound is not None:
                numeric = pd.to_numeric(converted, errors="coerce")
                invalid = (series.notna() & numeric.isna())
                bad = numeric < bound if bound_name == "minimum" else numeric > bound
                count = int((invalid | bad).sum())
                check(bound_name, count == 0, column=column, violations=count)
        if rule.allowed_values is not None:
            invalid = int((series.notna() & ~converted.isin(rule.allowed_values)).sum())
            check("allowed_values", invalid == 0, column=column, violations=invalid)
    if contract.primary_key:
        present = set(contract.primary_key) <= set(frame.columns)
        check("key_columns_present", present)
        if present:
            nulls = int(frame[contract.primary_key].isna().any(axis=1).sum())
            duplicates = int(frame.duplicated(subset=contract.primary_key).sum())
            check("key_non_null", nulls == 0, rows=nulls)
            check("key_unique", duplicates == 0, duplicates=duplicates)
    if contract.freshness_column:
        column = contract.freshness_column
        if column not in frame:
            check("freshness", False, reason="missing freshness column")
        else:
            times = pd.to_datetime(frame[column], errors="coerce", utc=True, format="mixed")
            reference = pd.Timestamp(as_of, tz="UTC") + pd.Timedelta(days=1)
            latest = times.max()
            age = (reference - latest).total_seconds() / 86400 if pd.notna(latest) else None
            future = int((times >= reference).sum())
            invalid = int((frame[column].notna() & times.isna()).sum())
            check("freshness", age is not None and age <= contract.max_age_days + 1
                  and future == 0 and invalid == 0,
                  age_days=age, future_rows=future, maximum_days=contract.max_age_days)
    return {"passed": all(c["passed"] for c in checks), "checks": checks}


def integrity(frames: dict[str, pd.DataFrame], config: HarnessConfig) -> dict:
    checks = []
    for rel in config.relationships:
        if not ({rel.child, rel.parent} & frames.keys()):
            continue
        present = {rel.child, rel.parent} <= frames.keys()
        if not present:
            checks.append({"check": "relationship", "passed": False,
                           "child": rel.child, "parent": rel.parent, "reason": "dataset unavailable"})
            continue
        child, parent = frames[rel.child], frames[rel.parent]
        if not (set(rel.child_columns) <= set(child) and set(rel.parent_columns) <= set(parent)):
            checks.append({"check": "relationship", "passed": False,
                           "child": rel.child, "parent": rel.parent, "reason": "missing keys"})
            continue
        parent_keys = parent[rel.parent_columns]
        unique = not parent_keys.duplicated().any() and not parent_keys.isna().any(axis=None)
        child_keys = child[rel.child_columns]
        # Null foreign keys are orphans; no null == null join behavior.
        known = set(parent_keys.dropna().itertuples(index=False, name=None))
        orphan = sum(any(pd.isna(v) for v in key) or key not in known
                     for key in child_keys.itertuples(index=False, name=None))
        fraction = orphan / max(1, len(child))
        checks.append({"check": "relationship", "child": rel.child, "parent": rel.parent,
                       "passed": unique and fraction <= rel.max_orphan_fraction,
                       "parent_key_unique": unique, "orphan_rows": orphan,
                       "orphan_fraction": fraction})
    for rec in config.reconciliations:
        if not ({rec.left, rec.right} & frames.keys()):
            continue
        entry = {"check": "reconciliation", "left": rec.left, "right": rec.right,
                 "metric": rec.metric, "passed": False}
        if {rec.left, rec.right} <= frames.keys():
            left, right = frames[rec.left], frames[rec.right]
            if rec.metric == "row_count":
                a, b = len(left), len(right)
            elif rec.left_column in left and rec.right_column in right:
                a_values = pd.to_numeric(left[rec.left_column], errors="coerce")
                b_values = pd.to_numeric(right[rec.right_column], errors="coerce")
                if not np.isfinite(a_values.to_numpy(dtype=float)).all() or not np.isfinite(
                    b_values.to_numpy(dtype=float)
                ).all():
                    entry["reason"] = "sum contains null or invalid values"
                    checks.append(entry)
                    continue
                a, b = float(a_values.sum()), float(b_values.sum())
            else:
                entry["reason"] = "missing reconciliation columns"
                checks.append(entry)
                continue
            delta = abs(a - b) / max(abs(a), abs(b), 1e-12)
            entry.update({"left_value": a, "right_value": b, "relative_difference": delta,
                          "passed": delta <= rec.relative_tolerance})
        else:
            entry["reason"] = "dataset unavailable"
        checks.append(entry)
    return {"passed": all(c["passed"] for c in checks), "checks": checks}
