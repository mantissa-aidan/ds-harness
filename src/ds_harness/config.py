"""Strict, portable configuration. Credentials are referenced by environment variable only."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Name = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class LLMConfig(StrictModel):
    mode: Literal["endpoint", "offline"] = "endpoint"
    base_url_env: str = "DS_LLM_BASE_URL"
    model_env: str = "DS_LLM_MODEL"
    api_key_env: str = "DS_LLM_API_KEY"
    auth_header: Literal["Authorization", "api-key"] = "Authorization"
    json_mode: bool = True
    token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    max_output_tokens: int = Field(default=4096, ge=256, le=16384)
    max_calls: int = Field(default=10, ge=1, le=50)
    max_prompt_chars: int = Field(default=60000, ge=1000, le=500000)
    timeout_seconds: int = Field(default=60, ge=1, le=300)
    retries: int = Field(default=2, ge=0, le=5)


class Limits(StrictModel):
    max_rows: int = Field(default=100000, ge=10, le=5000000)
    query_timeout_seconds: int = Field(default=60, ge=1, le=3600)
    max_analyses: int = Field(default=8, ge=1, le=30)
    min_group_size: int = Field(default=5, ge=2)
    max_category_levels: int = Field(default=100, ge=2, le=1000)


class Source(StrictModel):
    kind: Literal["sqlserver", "sqlite", "files"]
    connection_env: str | None = None
    root: str | None = None
    allowed_tables: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_location(self):
        if self.kind == "sqlserver" and not self.connection_env:
            raise ValueError("SQL Server requires connection_env (an ODBC connection string)")
        if self.kind in {"files", "sqlite"} and not self.root:
            raise ValueError("Local sources require root")
        if self.kind != "files" and not self.allowed_tables:
            raise ValueError("SQL sources require an explicit allowed_tables list")
        return self


class ColumnContract(StrictModel):
    dtype: Literal["string", "number", "integer", "datetime", "boolean"] | None = None
    required: bool = True
    max_null_fraction: float = Field(default=1.0, ge=0, le=1)
    minimum: float | None = None
    maximum: float | None = None
    allowed_values: list[str | int | float | bool] | None = None
    sensitive: bool = False
    description: str = ""

    @model_validator(mode="after")
    def range_order(self):
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum:
                raise ValueError("minimum must not exceed maximum")
        return self


class Contract(StrictModel):
    columns: dict[str, ColumnContract] = Field(default_factory=dict)
    primary_key: list[str] = Field(default_factory=list)
    min_rows: int = Field(default=1, ge=1)
    freshness_column: str | None = None
    max_age_days: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def freshness_pair(self):
        if (self.freshness_column is None) != (self.max_age_days is None):
            raise ValueError("freshness_column and max_age_days must be supplied together")
        return self


class QueryFilter(StrictModel):
    column: str
    op: Literal["eq", "ne", "lt", "lte", "gt", "gte", "in", "is_null", "not_null"]
    value: str | int | float | bool | list[str | int | float | bool] | None = None


class QueryMeasure(StrictModel):
    column: str = "*"
    aggregate: Literal["count", "sum", "avg", "min", "max"]
    alias: Name


class QueryJoin(StrictModel):
    table: str
    alias: Name
    left_column: str
    right_column: str
    kind: Literal["inner", "left"] = "inner"


class SelectQuery(StrictModel):
    table: str
    columns: list[str] = Field(default_factory=list)
    filters: list[QueryFilter] = Field(default_factory=list)
    joins: list[QueryJoin] = Field(default_factory=list, max_length=4)
    group_by: list[str] = Field(default_factory=list)
    measures: list[QueryMeasure] = Field(default_factory=list)
    order_by: list[str] = Field(default_factory=list)


class Candidate(StrictModel):
    source: Name
    table: str | None = None
    query: SelectQuery | None = None
    path: str | None = None
    note: str = ""

    @model_validator(mode="after")
    def one_locator(self):
        if sum(x is not None for x in (self.table, self.query, self.path)) != 1:
            raise ValueError("Specify exactly one of table, query or path")
        return self


class Preparation(StrictModel):
    drop_exact_duplicates: bool = True
    trim_strings: bool = True
    drop_null_rows: list[str] = Field(default_factory=list)
    # Learned imputation belongs inside the model's training pipeline.
    max_drop_fraction: float = Field(default=0.05, ge=0, le=1)


class Dataset(StrictModel):
    description: str = ""
    candidates: list[Candidate] = Field(min_length=1)
    contract: Contract = Field(default_factory=Contract)
    preparation: Preparation = Field(default_factory=Preparation)


class Relationship(StrictModel):
    child: Name
    child_columns: list[str] = Field(min_length=1)
    parent: Name
    parent_columns: list[str] = Field(min_length=1)
    max_orphan_fraction: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode="after")
    def equal_keys(self):
        if len(self.child_columns) != len(self.parent_columns):
            raise ValueError("Relationship key lengths must match")
        return self


class Reconciliation(StrictModel):
    left: Name
    right: Name
    metric: Literal["row_count", "sum"] = "row_count"
    left_column: str | None = None
    right_column: str | None = None
    relative_tolerance: float = Field(default=0.01, ge=0)

    @model_validator(mode="after")
    def sum_columns(self):
        if self.metric == "sum" and not (self.left_column and self.right_column):
            raise ValueError("Sum reconciliation needs both column names")
        return self


class HarnessConfig(StrictModel):
    version: Literal[1] = 1
    llm: LLMConfig = Field(default_factory=LLMConfig)
    limits: Limits = Field(default_factory=Limits)
    sources: dict[Name, Source]
    datasets: dict[Name, Dataset]
    relationships: list[Relationship] = Field(default_factory=list)
    reconciliations: list[Reconciliation] = Field(default_factory=list)

    @model_validator(mode="after")
    def references(self):
        if not self.datasets:
            raise ValueError("At least one dataset is required")
        for name, dataset in self.datasets.items():
            for candidate in dataset.candidates:
                if candidate.source not in self.sources:
                    raise ValueError(f"Unknown source for dataset {name}")
                source = self.sources[candidate.source]
                if (source.kind == "files") != (candidate.path is not None):
                    raise ValueError(f"Candidate locator does not match source type for {name}")
        for rel in self.relationships:
            if rel.child not in self.datasets or rel.parent not in self.datasets:
                raise ValueError("Unknown dataset in relationship")
        for rec in self.reconciliations:
            if rec.left not in self.datasets or rec.right not in self.datasets:
                raise ValueError("Unknown dataset in reconciliation")
        return self


class AnalysisSpec(StrictModel):
    id: Name
    kind: Literal["kpi", "rfm", "classification", "regression"]
    dataset: Name
    rationale: str = ""
    group_by: list[str] = Field(default_factory=list)
    measures: list[str] = Field(default_factory=list)
    customer_id: str | None = None
    date_column: str | None = None
    amount_column: str | None = None
    transaction_id: str | None = None
    target: str | None = None
    features: list[str] = Field(default_factory=list)
    split: Literal["random", "time", "group"] = "random"
    split_column: str | None = None
    test_fraction: float = Field(default=0.25, ge=0.1, le=0.4)
    min_improvement: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def required_fields(self):
        if self.kind == "kpi" and not self.measures:
            raise ValueError("KPI requires at least one numeric measure")
        if self.kind == "rfm" and not all(
            [self.customer_id, self.date_column, self.amount_column, self.transaction_id]
        ):
            raise ValueError("RFM requires customer, date, amount and transaction ID columns")
        if self.kind in {"classification", "regression"}:
            if not self.target or not self.features:
                raise ValueError("Predictive models require target and explicit feature columns")
            if self.target in self.features:
                raise ValueError("The target cannot be a feature")
            if self.split != "random" and not self.split_column:
                raise ValueError("Time/group split requires split_column")
            if self.split_column in self.features:
                raise ValueError("The split column cannot be a feature")
        if len(set(self.features)) != len(self.features):
            raise ValueError("Feature names must be unique")
        return self


class Task(StrictModel):
    name: Name
    objective: str = Field(min_length=10)
    decision: str = Field(min_length=5)
    success_criteria: list[str] = Field(min_length=1)
    datasets: list[Name] = Field(min_length=1)
    as_of: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    hypotheses: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    analyses: list[AnalysisSpec] = Field(default_factory=list)
    # Only these columns may be used in models (must be known before the outcome).
    feature_allowlist: dict[Name, list[str]] = Field(default_factory=dict)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def valid_task(self):
        date.fromisoformat(self.as_of)
        if len(set(self.datasets)) != len(self.datasets):
            raise ValueError("Task datasets must be unique")
        return self


class Plan(StrictModel):
    hypotheses: list[str]
    assumptions: list[str]
    analyses: list[AnalysisSpec] = Field(min_length=1)


def load_yaml(path: Path, model: type[StrictModel]):
    with path.open(encoding="utf-8") as handle:
        return model.model_validate(yaml.safe_load(handle))


def canonical_json(value) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
