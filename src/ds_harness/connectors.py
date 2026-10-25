"""Bounded, read-only SQL Server, SQLite, CSV and Parquet extracts."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ds_harness.config import Candidate, HarnessConfig, SelectQuery
from ds_harness.errors import PolicyError, SourceError
from ds_harness.sql import compile_select, relation


def contained_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise PolicyError("File path must remain inside the configured source root")
    return path


class Connector:
    def __init__(self, config: HarnessConfig, config_dir: Path):
        self.config = config
        self.config_dir = config_dir.resolve()

    def _connect(self, source_name: str):
        source = self.config.sources[source_name]
        if source.kind == "sqlite":
            path = (self.config_dir / source.root).resolve()
            if not path.is_file():
                raise SourceError("SQLite source does not exist")
            conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
            conn.execute("PRAGMA query_only = ON")
            deadline = time.monotonic() + self.config.limits.query_timeout_seconds
            conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
            return conn
        connection = os.environ.get(source.connection_env or "")
        if not connection:
            raise SourceError(f"Set the connection environment variable: {source.connection_env}")
        try:
            import pyodbc
        except ImportError:
            raise SourceError("Install ds-harness[sqlserver] and Microsoft ODBC Driver 18") from None
        conn = pyodbc.connect(connection, timeout=15, autocommit=False)
        conn.timeout = self.config.limits.query_timeout_seconds
        return conn

    def read(self, candidate: Candidate) -> tuple[pd.DataFrame, dict]:
        source = self.config.sources[candidate.source]
        limit = self.config.limits.max_rows
        provenance = {"source": candidate.source, "kind": source.kind, "row_limit": limit}
        try:
            if source.kind == "files":
                root = (self.config_dir / source.root).resolve()
                path = contained_path(root, candidate.path)
                if path.suffix.lower() == ".csv":
                    # Preserve identifiers such as 00042; contracts perform explicit type conversion.
                    frame = pd.read_csv(path, nrows=limit + 1, dtype="string")
                elif path.suffix.lower() == ".parquet":
                    parquet = pq.ParquetFile(path)
                    batches, count = [], 0
                    for batch in parquet.iter_batches(batch_size=min(limit + 1, 65536)):
                        batches.append(batch.to_pandas())
                        count += len(batches[-1])
                        if count > limit:
                            break
                    frame = pd.concat(batches, ignore_index=True) if batches else pd.DataFrame(
                        columns=parquet.schema_arrow.names
                    )
                    frame = frame.head(limit + 1)
                else:
                    raise PolicyError("Only CSV and Parquet files are supported")
                provenance["locator"] = candidate.path
            else:
                dialect = "tsql" if source.kind == "sqlserver" else "sqlite"
                query = candidate.query or SelectQuery(table=candidate.table)
                sql, parameters = compile_select(query, source.allowed_tables, limit + 1, dialect)
                provenance["query_sha256"] = hashlib.sha256(sql.encode()).hexdigest()
                provenance["query"] = sql
                conn = self._connect(candidate.source)
                try:
                    cursor = conn.cursor()
                    try:
                        cursor.execute(sql, parameters)
                        columns = [c[0] for c in cursor.description]
                        frame = pd.DataFrame.from_records(cursor.fetchmany(limit + 1), columns=columns)
                    finally:
                        cursor.close()
                        conn.rollback()
                finally:
                    conn.close()
            if frame.columns.duplicated().any():
                raise PolicyError("Extract has duplicate column names; use a view with unique names")
            provenance.update({"rows": len(frame), "truncated": len(frame) > limit})
            if len(frame) > limit:
                raise SourceError("Extract exceeded max_rows; narrow or aggregate the source query")
            return frame, provenance
        except (PolicyError, SourceError):
            raise
        except Exception as exc:
            raise SourceError(
                f"Source {candidate.source} could not be read ({type(exc).__name__}); "
                "check connectivity, driver, permissions and configured columns"
            ) from None

    def catalog(self) -> dict:
        """Read metadata only for explicitly approved relations."""
        catalog = {}
        for name, source in self.config.sources.items():
            if source.kind == "files":
                catalog[name] = {"kind": "files", "datasets": [
                    d for d, spec in self.config.datasets.items()
                    if any(c.source == name for c in spec.candidates)
                ]}
                continue
            conn = None
            try:
                conn = self._connect(name)
                tables = {}
                for table in source.allowed_tables:
                    cursor = conn.cursor()
                    try:
                        cursor.execute(f"SELECT * FROM {relation(table)} WHERE 1 = 0")
                        tables[table] = [{"name": c[0], "type": str(c[1])}
                                         for c in cursor.description]
                    finally:
                        cursor.close()
                catalog[name] = {"kind": source.kind, "tables": tables}
            except Exception as exc:
                catalog[name] = {"kind": source.kind, "error_type": type(exc).__name__}
            finally:
                if conn is not None:
                    conn.rollback()
                    conn.close()
        return catalog
