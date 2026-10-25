import unittest

import numpy as np
import pandas as pd

from ds_harness.analysis import _split, execute, validate_plan
from ds_harness.config import AnalysisSpec, HarnessConfig, Plan, Task
from ds_harness.errors import PolicyError


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.config = HarnessConfig.model_validate({
            "sources": {"files": {"kind": "files", "root": "."}},
            "datasets": {"data": {"candidates": [{"source": "files", "path": "data.csv"}],
                                  "contract": {"primary_key": ["id"]}}},
        })
        self.task = Task(name="analysis_test", objective="Test measurable analytical outputs",
                         decision="Evaluate results", success_criteria=["Known relationship recovered"],
                         datasets=["data"], as_of="2026-10-01", feature_allowlist={"data": ["x"]})

    def test_small_kpi_groups_suppressed(self):
        frame = pd.DataFrame({"group": ["a"] * 8 + ["b"] * 2, "amount": [10] * 10})
        spec = AnalysisSpec(id="kpi", kind="kpi", dataset="data", group_by=["group"], measures=["amount"])
        result, table = execute(frame, spec, self.config, self.task)
        self.assertEqual(table["group"].tolist(), ["a"])
        self.assertEqual(result["metrics"]["suppressed_rows"], 2)
        self.assertEqual(table["amount_sum"].iloc[0], 80)

    def test_null_measure_not_silently_summed(self):
        spec = AnalysisSpec(id="kpi", kind="kpi", dataset="data", measures=["amount"])
        with self.assertRaises(PolicyError):
            execute(pd.DataFrame({"amount": [1.0, None]}), spec, self.config, self.task)

    def test_target_cannot_be_feature(self):
        with self.assertRaises(ValueError):
            AnalysisSpec(id="model", kind="regression", dataset="data", target="y", features=["y"])

    def test_sensitive_and_unapproved_features_rejected(self):
        frame = pd.DataFrame({"x": [1, 2], "email": ["a", "b"], "y": [0, 1]})
        for feature in ["email", "y"]:
            with self.subTest(feature=feature), self.assertRaises(ValueError):
                spec = AnalysisSpec(id="model", kind="regression", dataset="data", target="x", features=[feature])
                validate_plan(Plan(hypotheses=[], assumptions=[], analyses=[spec]), {"data": frame},
                              self.config, self.task)

    def test_time_split_keeps_tied_dates_together(self):
        frame = pd.DataFrame({"date": ["2025-01-01"] * 60 + ["2025-02-01"] * 40})
        spec = AnalysisSpec(id="model", kind="regression", dataset="data", target="y", features=["x"],
                            split="time", split_column="date")
        train, test = _split(frame, spec, pd.Series(range(100)), 42)
        self.assertEqual(len(train), 60)
        self.assertEqual(len(test), 40)
        self.assertLess(frame.iloc[train]["date"].max(), frame.iloc[test]["date"].min())

    def test_group_split_has_no_entity_overlap(self):
        frame = pd.DataFrame({"customer": np.repeat(np.arange(20), 5)})
        spec = AnalysisSpec(id="model", kind="regression", dataset="data", target="y", features=["x"],
                            split="group", split_column="customer")
        train, test = _split(frame, spec, pd.Series(range(100)), 42)
        self.assertFalse(set(frame.iloc[train]["customer"]) & set(frame.iloc[test]["customer"]))

    def test_regression_beats_mean_baseline(self):
        rng = np.random.default_rng(42)
        x = rng.normal(size=200)
        frame = pd.DataFrame({"x": x, "y": x * 5 + rng.normal(scale=0.1, size=200)})
        spec = AnalysisSpec(id="regression", kind="regression", dataset="data", target="y", features=["x"])
        result, _ = execute(frame, spec, self.config, self.task)
        self.assertTrue(result["passed"])
        self.assertGreater(result["metrics"]["improvement"], 0.9)

    def test_rfm_counts_distinct_orders(self):
        frame = pd.DataFrame({"customer": np.repeat(np.arange(30), 2),
                              "order": np.repeat(np.arange(30), 2),
                              "date": ["2026-09-01"] * 60, "amount": [5.0] * 60})
        spec = AnalysisSpec(id="rfm", kind="rfm", dataset="data", customer_id="customer",
                            transaction_id="order", date_column="date", amount_column="amount")
        result, table = execute(frame, spec, self.config, self.task)
        self.assertEqual(result["metrics"]["customers"], 30)
        self.assertTrue((table["mean_frequency"] == 1).all())
        self.assertEqual(table["revenue"].sum(), 300)


if __name__ == "__main__":
    unittest.main()
