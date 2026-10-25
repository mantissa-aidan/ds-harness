import sqlite3
import os
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ds_harness.config import Candidate, HarnessConfig, SelectQuery
from ds_harness.connectors import Connector, contained_path
from ds_harness.errors import PolicyError, SourceError
from ds_harness.sql import compile_select


class SQLTests(unittest.TestCase):
    def test_filter_values_are_parameters(self):
        payload = "x'; DROP TABLE customers; --"
        query = SelectQuery.model_validate({"table": "dbo.customers", "columns": ["id"],
                                          "filters": [{"column": "name", "op": "eq", "value": payload}]})
        sql, params = compile_select(query, ["dbo.customers"], 101)
        self.assertNotIn(payload, sql)
        self.assertEqual(params, [payload])
        self.assertIn("TOP (101)", sql)
        self.assertIn("[t].[name] = ?", sql)

    def test_identifiers_cannot_inject(self):
        for column in ["id; DELETE FROM x", "id] FROM x --", "dbo.fn()", "@@version", "a.b.c"]:
            with self.subTest(column=column), self.assertRaises(PolicyError):
                compile_select(SelectQuery(table="dbo.customers", columns=[column]),
                               ["dbo.customers"], 10)

    def test_tables_must_be_allowlisted(self):
        with self.assertRaises(PolicyError):
            compile_select(SelectQuery(table="dbo.secret"), ["dbo.customers"], 10)

    def test_cross_database_disallowed_even_if_listed(self):
        with self.assertRaises(PolicyError):
            compile_select(SelectQuery(table="db.dbo.customers"), ["db.dbo.customers"], 10)

    def test_raw_sql_is_not_a_query(self):
        with self.assertRaises(ValueError):
            Candidate(source="warehouse", query="DELETE FROM customers")

    def test_join_and_aggregate_execute(self):
        query = SelectQuery.model_validate({
            "table": "orders", "joins": [{"table": "customers", "alias": "c",
                                           "left_column": "customer_id", "right_column": "id"}],
            "group_by": ["c.region"], "measures": [{"column": "amount", "aggregate": "sum", "alias": "revenue"}],
            "order_by": ["revenue"],
        })
        sql, params = compile_select(query, ["orders", "customers"], 11, "sqlite")
        with closing(sqlite3.connect(":memory:")) as conn, conn:
            conn.executescript("CREATE TABLE customers(id INTEGER, region TEXT);"
                               "CREATE TABLE orders(customer_id INTEGER, amount REAL);"
                               "INSERT INTO customers VALUES(1, 'west');"
                               "INSERT INTO orders VALUES(1, 10), (1, 20);")
            self.assertEqual(conn.execute(sql, params).fetchall(), [("west", 30.0)])

    def test_join_target_must_be_allowed(self):
        query = SelectQuery.model_validate({"table": "orders", "columns": ["amount"],
                                           "joins": [{"table": "secret", "alias": "s",
                                                      "left_column": "id", "right_column": "id"}]})
        with self.assertRaises(PolicyError):
            compile_select(query, ["orders"], 10)

    def test_file_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(PolicyError):
                contained_path(Path(folder), "../secret.csv")


class ConnectorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        with closing(sqlite3.connect(self.root / "test.sqlite")) as conn, conn:
            conn.execute("CREATE TABLE customers (id INTEGER, value REAL)")
            conn.executemany("INSERT INTO customers VALUES (?, ?)", [(i, i * 2) for i in range(20)])
        self.config = HarnessConfig.model_validate({
            "sources": {"db": {"kind": "sqlite", "root": "test.sqlite", "allowed_tables": ["customers"]}},
            "datasets": {"customers": {"candidates": [{"source": "db", "table": "customers"}]}},
        })
        self.connector = Connector(self.config, self.root)

    def tearDown(self):
        self.directory.cleanup()

    def test_real_read_and_catalog(self):
        frame, metadata = self.connector.read(Candidate(source="db", table="customers"))
        self.assertEqual(len(frame), 20)
        self.assertFalse(metadata["truncated"])
        self.assertIn("customers", self.connector.catalog()["db"]["tables"])

    def test_sqlite_connection_rejects_writes(self):
        conn = self.connector._connect("db")
        try:
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("DELETE FROM customers")
        finally:
            conn.close()

    def test_row_limit_does_not_silently_truncate(self):
        self.config.limits.max_rows = 10
        with self.assertRaisesRegex(SourceError, "exceeded max_rows"):
            self.connector.read(Candidate(source="db", table="customers"))

    def test_missing_database_not_created(self):
        self.config.sources["db"].root = "missing.sqlite"
        with self.assertRaises(SourceError):
            self.connector.read(Candidate(source="db", table="customers"))
        self.assertFalse((self.root / "missing.sqlite").exists())

    def test_driver_errors_do_not_disclose_secrets(self):
        with patch.object(self.connector, "_connect", side_effect=RuntimeError("password=SECRET")):
            with self.assertRaises(SourceError) as context:
                self.connector.read(Candidate(source="db", table="customers"))
        self.assertNotIn("SECRET", str(context.exception))

    def test_csv_preserves_leading_zero_ids(self):
        (self.root / "customers.csv").write_text("id,value\n00001,2\n00002,3\n", encoding="utf-8")
        config = HarnessConfig.model_validate({
            "sources": {"files": {"kind": "files", "root": "."}},
            "datasets": {"customers": {"candidates": [{"source": "files", "path": "customers.csv"}]}},
        })
        frame, _ = Connector(config, self.root).read(Candidate(source="files", path="customers.csv"))
        self.assertEqual(frame["id"].tolist(), ["00001", "00002"])

    def test_sqlserver_driver_timeout_bounded_select_and_cleanup(self):
        config = HarnessConfig.model_validate({
            "sources": {"sql": {"kind": "sqlserver", "connection_env": "TEST_ODBC",
                                "allowed_tables": ["dbo.customers"]}},
            "datasets": {"customers": {"candidates": [{"source": "sql", "table": "dbo.customers"}]}},
        })
        connection = MagicMock()
        cursor = connection.cursor.return_value
        cursor.description = [("id", int), ("value", float)]
        cursor.fetchmany.return_value = [(1, 2.0)]
        driver = SimpleNamespace(connect=MagicMock(return_value=connection))
        with patch.dict(sys.modules, {"pyodbc": driver}), patch.dict(os.environ, {"TEST_ODBC": "SECRET"}):
            frame, provenance = Connector(config, self.root).read(Candidate(source="sql", table="dbo.customers"))
        driver.connect.assert_called_once_with("SECRET", timeout=15, autocommit=False)
        self.assertEqual(connection.timeout, 60)
        self.assertIn("TOP (100001)", cursor.execute.call_args.args[0])
        self.assertEqual(len(frame), 1)
        connection.rollback.assert_called_once()
        connection.close.assert_called_once()
        cursor.close.assert_called_once()
        self.assertNotIn("SECRET", str(provenance))


if __name__ == "__main__":
    unittest.main()
