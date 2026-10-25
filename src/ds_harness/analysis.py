"""Deterministic analytics. Model code is maintained here, never executed from LLM output."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ds_harness.config import AnalysisSpec, HarnessConfig, Plan, Task
from ds_harness.errors import PolicyError
from ds_harness.quality import sensitive_columns


def referenced_columns(spec: AnalysisSpec) -> set[str]:
    return set(spec.group_by + spec.measures + spec.features + [
        c for c in [spec.customer_id, spec.date_column, spec.amount_column,
                    spec.transaction_id, spec.target, spec.split_column] if c
    ])


def validate_plan(plan: Plan, frames: dict[str, pd.DataFrame], config: HarnessConfig, task: Task):
    if len(plan.analyses) > config.limits.max_analyses:
        raise ValueError("Plan exceeds max_analyses")
    if len({a.id for a in plan.analyses}) != len(plan.analyses):
        raise ValueError("Analysis IDs must be unique")
    for spec in plan.analyses:
        if spec.dataset not in task.datasets or spec.dataset not in frames:
            raise ValueError("Analysis dataset is outside the task or unavailable")
        frame = frames[spec.dataset]
        if not referenced_columns(spec) <= set(frame):
            raise ValueError(f"Analysis {spec.id} references unknown columns")
        contract = config.datasets[spec.dataset].contract
        restricted = sensitive_columns(frame, contract) | set(contract.primary_key)
        if set(spec.group_by + spec.measures + spec.features) & restricted:
            raise ValueError("Sensitive columns and primary keys cannot be features or KPI dimensions")
        if spec.target in sensitive_columns(frame, contract):
            raise ValueError("Sensitive targets are not supported")
        if spec.kind in {"classification", "regression"}:
            if not set(spec.features) <= set(task.feature_allowlist.get(spec.dataset, [])):
                raise ValueError("Predictive features must be explicitly approved in feature_allowlist")
        if len(spec.group_by) > 3:
            raise ValueError("Use at most three KPI grouping columns")


def _numeric(frame: pd.DataFrame, columns: list[str], allow_missing: bool = False):
    for column in columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise PolicyError(f"Numeric column required: {column}")
        nonnull = frame[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(nonnull).all():
            raise PolicyError(f"Non-finite values in numeric column: {column}")
        if not allow_missing and frame[column].isna().any():
            raise PolicyError(f"Missing values in required numeric column: {column}")


def kpi(frame: pd.DataFrame, spec: AnalysisSpec, config: HarnessConfig) -> tuple[dict, pd.DataFrame]:
    _numeric(frame, spec.measures)
    group_size = config.limits.min_group_size
    if spec.group_by:
        for column in spec.group_by:
            if frame[column].nunique() > config.limits.max_category_levels:
                raise PolicyError("KPI dimension exceeds max_category_levels")
        grouped = frame.groupby(spec.group_by, dropna=False, observed=True)
        rows = grouped.size().rename("rows").to_frame()
        for measure in spec.measures:
            rows[f"{measure}_sum"] = grouped[measure].sum(min_count=1)
            rows[f"{measure}_mean"] = grouped[measure].mean()
        small = rows["rows"] < group_size
        suppressed = int(rows.loc[small, "rows"].sum())
        rows = rows.loc[~small].reset_index()
    else:
        suppressed = len(frame) if len(frame) < group_size else 0
        rows = pd.DataFrame([] if suppressed else [{
            "rows": len(frame),
            **{f"{m}_sum": float(frame[m].sum()) for m in spec.measures},
            **{f"{m}_mean": float(frame[m].mean()) for m in spec.measures},
        }])
    metrics = {"input_rows": len(frame), "reported_groups": len(rows), "suppressed_rows": suppressed}
    if not rows.empty:
        for measure in spec.measures:
            metrics[f"reported_{measure}_sum"] = float(rows[f"{measure}_sum"].sum())
    return {
        "kind": "kpi", "passed": not rows.empty,
        "metrics": metrics,
        "limitations": ["Descriptive associations do not establish causality.",
                        "Counts are rows at the configured extract grain, not unique customers."],
    }, rows


def rfm(frame: pd.DataFrame, spec: AnalysisSpec, config: HarnessConfig, task: Task):
    _numeric(frame, [spec.amount_column])
    if frame[[spec.customer_id, spec.transaction_id]].isna().any(axis=None):
        raise PolicyError("RFM identifiers must be complete")
    dates = pd.to_datetime(frame[spec.date_column], errors="coerce", utc=True, format="mixed")
    as_of = pd.Timestamp(task.as_of, tz="UTC") + pd.Timedelta(days=1)
    if dates.isna().any() or (dates >= as_of).any():
        raise PolicyError("RFM dates must be valid and no later than as_of")
    if (frame[spec.amount_column] < 0).any():
        raise PolicyError("RFM requires non-negative purchase amounts; define treatment of returns")
    owners = frame.groupby(spec.transaction_id)[spec.customer_id].nunique()
    if (owners > 1).any():
        raise PolicyError("One transaction is assigned to multiple customers")
    work = frame.assign(_purchase_date=dates)
    groups = work.groupby(spec.customer_id)
    customers = pd.DataFrame({
        "recency_days": (as_of - groups["_purchase_date"].max()).dt.days,
        "frequency": groups[spec.transaction_id].nunique(),
        "monetary": groups[spec.amount_column].sum(),
    })
    if len(customers) < max(20, config.limits.min_group_size):
        raise PolicyError("RFM requires at least 20 customers")
    for name, ascending in [("recency_days", False), ("frequency", True), ("monetary", True)]:
        # Ties get the same score; no arbitrary row-order tie breaking.
        customers[f"{name}_score"] = np.ceil(customers[name].rank(pct=True, ascending=ascending) * 5)
    recent = customers["recency_days_score"] >= 4
    valuable = (customers["frequency_score"] >= 4) & (customers["monetary_score"] >= 4)
    customers["segment"] = np.select(
        [recent & valuable, ~recent & valuable, recent],
        ["champions", "valuable_lapsed", "recent"], default="other",
    )
    segments = customers.groupby("segment").agg(
        customers=("frequency", "size"), mean_recency_days=("recency_days", "mean"),
        mean_frequency=("frequency", "mean"), revenue=("monetary", "sum"),
    ).reset_index()
    small = segments["customers"] < config.limits.min_group_size
    suppressed = int(segments.loc[small, "customers"].sum())
    segments = segments.loc[~small]
    return {
        "kind": "rfm", "passed": not segments.empty,
        "metrics": {"customers": len(customers), "reported_segments": len(segments),
                    "suppressed_customers": suppressed},
        "limitations": ["RFM segments are descriptive heuristics, not churn predictions.",
                        "Frequency counts distinct transactions; monetary value sums extract rows.",
                        "Use an experiment to measure campaign impact."],
    }, segments


def _split(frame: pd.DataFrame, spec: AnalysisSpec, target: pd.Series, seed: int):
    indices = np.arange(len(frame))
    if spec.split == "time":
        times = pd.to_datetime(frame[spec.split_column], errors="coerce", utc=True, format="mixed")
        if times.isna().any():
            raise PolicyError("Time split column contains missing or invalid timestamps")
        ordered = times.sort_values()
        cutoff = ordered.iloc[int(len(frame) * (1 - spec.test_fraction))]
        train, test = indices[times < cutoff], indices[times >= cutoff]
    elif spec.split == "group":
        groups = frame[spec.split_column]
        if groups.isna().any() or groups.nunique() < 4:
            raise PolicyError("Group split requires at least four non-null groups")
        splitter = GroupShuffleSplit(n_splits=1, test_size=spec.test_fraction, random_state=seed)
        train, test = next(splitter.split(frame, target, groups))
    else:
        stratify = target if spec.kind == "classification" else None
        train, test = train_test_split(indices, test_size=spec.test_fraction,
                                       random_state=seed, stratify=stratify)
    if len(train) < 30 or len(test) < 10:
        raise PolicyError("Need at least 30 training and 10 test rows")
    return train, test


def predictive(frame: pd.DataFrame, spec: AnalysisSpec, config: HarnessConfig, task: Task):
    if len(frame) < 50 or frame[spec.target].isna().any():
        raise PolicyError("Predictive modeling needs at least 50 rows and complete labels")
    target = frame[spec.target]
    if spec.kind == "classification":
        if not set(target.unique()) <= {0, 1, False, True} or target.nunique() != 2:
            raise PolicyError("Classification requires an explicit binary 0/1 target; 1 is positive")
        target = target.astype(int)
        if target.value_counts().min() < 10:
            raise PolicyError("Classification requires at least ten observations per class")
    else:
        _numeric(frame, [spec.target])
    train, test = _split(frame, spec, target, task.seed)
    x = frame[spec.features].copy()
    numeric = list(x.select_dtypes(include="number").columns)
    categorical = [c for c in x if c not in numeric]
    _numeric(x, numeric, allow_missing=True)
    for c in categorical:
        if pd.api.types.is_datetime64_any_dtype(x[c]):
            raise PolicyError("Derive time features explicitly in the configured extract")
        if x.iloc[train][c].nunique() > config.limits.max_category_levels:
            raise PolicyError("Categorical feature exceeds max_category_levels")
        x[c] = x[c].astype(object).where(x[c].notna(), np.nan)
        x[c] = x[c].map(lambda v: str(v) if pd.notna(v) else np.nan)
    if x.iloc[train].isna().all().any():
        raise PolicyError("A feature is entirely missing in the training data")
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("category", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]), categorical))
    classification = spec.kind == "classification"
    estimator = LogisticRegression(max_iter=1000, random_state=task.seed) if classification else Ridge(
        alpha=1.0, solver="lsqr"
    )
    model = Pipeline([("prepare", ColumnTransformer(transformers)), ("model", estimator)])
    y_train, y_test = target.iloc[train], target.iloc[test]
    if classification and (y_train.nunique() != 2 or y_test.nunique() != 2):
        raise PolicyError("Both train and test partitions must contain both classes")
    model.fit(x.iloc[train], y_train)
    if classification:
        baseline = DummyClassifier(strategy="prior").fit(x.iloc[train], y_train)
        scores = model.predict_proba(x.iloc[test])[:, 1]
        base_scores = baseline.predict_proba(x.iloc[test])[:, 1]
        metric = float(roc_auc_score(y_test, scores))
        base_metric = float(roc_auc_score(y_test, base_scores))
        improvement = metric - base_metric
        metrics = {
            "roc_auc": metric, "baseline_roc_auc": base_metric,
            "average_precision": float(average_precision_score(y_test, scores)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, scores >= 0.5)),
        }
    else:
        baseline = DummyRegressor(strategy="mean").fit(x.iloc[train], y_train)
        scores = model.predict(x.iloc[test])
        base_scores = baseline.predict(x.iloc[test])
        metric = math.sqrt(mean_squared_error(y_test, scores))
        base_metric = math.sqrt(mean_squared_error(y_test, base_scores))
        improvement = (base_metric - metric) / max(base_metric, 1e-12)
        metrics = {"rmse": metric, "baseline_rmse": base_metric,
                   "mae": float(mean_absolute_error(y_test, scores)),
                   "r2": float(r2_score(y_test, scores))}
    # An uncertainty estimate, never used to tune or choose a model on the holdout.
    rng = np.random.default_rng(task.seed)
    estimates = []
    observed = y_test.to_numpy()
    for _ in range(200):
        sample = rng.integers(0, len(test), size=len(test))
        if classification:
            if len(np.unique(observed[sample])) < 2:
                continue
            score = roc_auc_score(observed[sample], scores[sample]) - 0.5
        else:
            model_error = math.sqrt(mean_squared_error(observed[sample], scores[sample]))
            baseline_error = math.sqrt(mean_squared_error(observed[sample], base_scores[sample]))
            score = (baseline_error - model_error) / max(baseline_error, 1e-12)
        estimates.append(score)
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    metrics.update({"train_rows": len(train), "test_rows": len(test), "improvement": improvement,
                    "improvement_ci95_low": float(lower), "improvement_ci95_high": float(upper)})
    passed = improvement > spec.min_improvement and lower > 0
    limitations = [
        "Feature availability before the outcome must be verified by the data owner.",
        "Holdout performance is not proof of business impact; use a prospective experiment.",
        "Bootstrap intervals assume independent observations; time/group dependence can widen them.",
        "No model is tuned or retried based on this holdout.",
    ]
    if spec.split == "random":
        limitations.append("Random splitting assumes independent rows; use group/time for repeated entities.")
    return {
        "kind": spec.kind, "passed": bool(passed), "metrics": metrics,
        "method": "logistic_regression" if classification else "ridge_regression",
        "split": spec.split, "features": spec.features,
        "acceptance": {"min_improvement": spec.min_improvement, "ci_lower_must_exceed": 0},
        "limitations": limitations,
    }, pd.DataFrame([{"metric": key, "value": value} for key, value in metrics.items()])


def execute(frame: pd.DataFrame, spec: AnalysisSpec, config: HarnessConfig, task: Task):
    if spec.kind == "kpi":
        result, table = kpi(frame, spec, config)
    elif spec.kind == "rfm":
        result, table = rfm(frame, spec, config, task)
    else:
        result, table = predictive(frame, spec, config, task)
    _numeric(table, list(table.select_dtypes(include="number").columns))
    if any(isinstance(v, (int, float)) and not math.isfinite(v)
           for v in result.get("metrics", {}).values()):
        raise PolicyError("Analysis produced non-finite metrics")
    return result, table
