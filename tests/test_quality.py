import unittest

import pandas as pd

from ds_harness.config import Contract, HarnessConfig, Preparation
from ds_harness.errors import PolicyError
from ds_harness.quality import assess, integrity, prepare, profile


class QualityTests(unittest.TestCase):
    def test_conflicting_keys_fail(self):
        frame = pd.DataFrame({"id": [1, 1], "value": [2, 3]})
        contract = Contract(primary_key=["id"])
        prepared, _ = prepare(frame, contract, Preparation())
        self.assertFalse(assess(prepared, contract, "2026-10-01")["passed"])

    def test_exact_duplicates_can_be_repaired_within_budget(self):
        frame = pd.DataFrame({"id": [1, 2, 2]})
        prepared, changes = prepare(frame, Contract(), Preparation(max_drop_fraction=0.4))
        self.assertEqual(len(prepared), 2)
        self.assertAlmostEqual(changes["dropped_fraction"], 1 / 3)

    def test_drop_budget_is_enforced(self):
        with self.assertRaises(PolicyError):
            prepare(pd.DataFrame({"id": [1, 1]}), Contract(), Preparation())

    def test_invalid_parsing_is_not_silently_imputed(self):
        contract = Contract.model_validate({"columns": {"amount": {"dtype": "number"}}})
        with self.assertRaises(PolicyError):
            prepare(pd.DataFrame({"amount": ["bad"]}), contract, Preparation())

    def test_boolean_false_is_false(self):
        contract = Contract.model_validate({"columns": {"flag": {"dtype": "boolean"}}})
        frame, _ = prepare(pd.DataFrame({"flag": ["false", "true"]}), contract, Preparation())
        self.assertEqual(frame["flag"].tolist(), [False, True])

    def test_integer_fraction_is_invalid(self):
        contract = Contract.model_validate({"columns": {"count": {"dtype": "integer"}}})
        self.assertFalse(assess(pd.DataFrame({"count": [1.5]}), contract, "2026-10-01")["passed"])

    def test_freshness_and_future_dates_fail(self):
        contract = Contract(freshness_column="time", max_age_days=7)
        for date in ["2020-01-01", "2026-10-03", "invalid"]:
            with self.subTest(date=date):
                self.assertFalse(assess(pd.DataFrame({"time": [date]}), contract, "2026-10-01")["passed"])

    def test_profile_excludes_raw_values(self):
        result = profile(pd.DataFrame({"email": ["private@example.test"], "amount": [76543]}), Contract())
        self.assertNotIn("private@example.test", str(result))
        self.assertNotIn("76543", str(result))
        self.assertTrue(result["columns"]["email"]["sensitive"])

    def test_nulls_range_and_schema_failure(self):
        contract = Contract.model_validate({"columns": {"missing": {},
                         "amount": {"dtype": "number", "minimum": 0, "max_null_fraction": 0}}})
        result = assess(pd.DataFrame({"amount": [None, -1]}), contract, "2026-10-01")
        self.assertFalse(result["passed"])
        self.assertEqual(sum(not c["passed"] for c in result["checks"]), 3)

    def test_orphans_and_reconciliation_fail(self):
        config = HarnessConfig.model_validate({
            "sources": {"files": {"kind": "files", "root": "."}},
            "datasets": {n: {"candidates": [{"source": "files", "path": f"{n}.csv"}]}
                         for n in ["parent", "child"]},
            "relationships": [{"parent": "parent", "parent_columns": ["id"],
                               "child": "child", "child_columns": ["id"]}],
            "reconciliations": [{"left": "parent", "right": "child"}],
        })
        result = integrity({"parent": pd.DataFrame({"id": [1]}),
                            "child": pd.DataFrame({"id": [1, 2]})}, config)
        self.assertFalse(result["passed"])
        self.assertEqual(result["checks"][0]["orphan_rows"], 1)
        self.assertFalse(result["checks"][1]["passed"])


if __name__ == "__main__":
    unittest.main()
