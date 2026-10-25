import json
import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import httpx
import yaml

from ds_harness.connectors import Connector
from ds_harness.demo import create_demo
from ds_harness.errors import HarnessError, LLMError, PolicyError
from ds_harness.runner import PHASES, Runner


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.config, self.task = create_demo(self.root / "fixture")
        self.output = self.root / "run"

    def tearDown(self):
        self.directory.cleanup()

    def read(self, name):
        return json.loads((self.output / name).read_text(encoding="utf-8"))

    def test_end_to_end_fallback_repair_and_evaluation(self):
        state = Runner(self.config, self.task, self.output).run()
        self.assertEqual(state["status"], "completed")
        self.assertEqual(set(state["phases"]), set(PHASES))
        self.assertTrue(all(v == "completed" for v in state["phases"].values()))
        self.assertEqual(self.read("data_understanding.json")["customers"]["selected"], 1)
        preparation = self.read("data_preparation.json")
        self.assertEqual(preparation["datasets"]["orders"]["input_rows"], 2404)
        self.assertEqual(preparation["datasets"]["orders"]["output_rows"], 2400)
        self.assertTrue(preparation["integrity"]["passed"])
        self.assertGreater(self.read("results.json")["churn_baseline"]["metrics"]["roc_auc"], 0.8)
        self.assertTrue((self.output / "report.md").exists())

    def test_resume_reuses_snapshots_without_source_reads(self):
        Runner(self.config, self.task, self.output).run()
        with patch.object(Connector, "read", side_effect=AssertionError("should not query")):
            state = Runner(self.config, self.task, self.output, resume=True).run()
        self.assertEqual(state["status"], "completed")

    def test_artifact_tampering_rejected(self):
        Runner(self.config, self.task, self.output).run()
        (self.output / "prepared" / "orders.parquet").write_bytes(b"changed")
        with self.assertRaisesRegex(PolicyError, "changed"):
            Runner(self.config, self.task, self.output, resume=True).run()

    def test_changed_task_cannot_reuse_old_run(self):
        Runner(self.config, self.task, self.output).run()
        task = yaml.safe_load(self.task.read_text())
        task["objective"] = "A different business objective requires a different run"
        self.task.write_text(yaml.safe_dump(task), encoding="utf-8")
        with self.assertRaisesRegex(PolicyError, "changed"):
            Runner(self.config, self.task, self.output, resume=True).run()

    def test_orphan_keys_block_analysis_and_emit_handoff(self):
        with closing(sqlite3.connect(self.config.parent / "demo.sqlite")) as conn, conn:
            conn.execute("UPDATE warehouse_orders SET customer_id = 'unknown' WHERE order_id = 'O000010'")
        state = Runner(self.config, self.task, self.output).run()
        self.assertEqual(state["status"], "blocked")
        self.assertFalse((self.output / "results.json").exists())
        self.assertTrue((self.output / "report.md").exists())
        self.assertFalse(self.read("handoff.json")["external_writes"])

    def test_failed_inference_can_resume_without_rereading_sources(self):
        with patch("ds_harness.runner.make_plan", side_effect=LLMError("Temporary endpoint error")):
            with self.assertRaises(LLMError):
                Runner(self.config, self.task, self.output).run()
        self.assertEqual(self.read("state.json")["status"], "failed")
        with patch.object(Connector, "read", side_effect=AssertionError("should not query")):
            state = Runner(self.config, self.task, self.output, resume=True).run()
        self.assertEqual(state["status"], "completed")

    def test_partial_handoff_is_resumable_not_prematurely_complete(self):
        runner = Runner(self.config, self.task, self.output)
        original = runner.store.text

        def fail_report(name, content):
            if name == "report.md":
                raise OSError("Simulated crash")
            return original(name, content)

        with patch.object(runner.store, "text", side_effect=fail_report), self.assertRaises(HarnessError):
            runner.run()
        self.assertEqual(self.read("state.json")["status"], "failed")
        state = Runner(self.config, self.task, self.output, resume=True).run()
        self.assertEqual(state["status"], "completed")
        self.assertTrue((self.output / "report.md").exists())

    def test_endpoint_plans_and_interprets_without_raw_records(self):
        config = yaml.safe_load(self.config.read_text())
        config["llm"]["mode"] = "endpoint"
        self.config.write_text(yaml.safe_dump(config), encoding="utf-8")
        task = yaml.safe_load(self.task.read_text())
        task["analyses"] = []
        self.task.write_text(yaml.safe_dump(task), encoding="utf-8")
        captured = []
        replies = iter([
            {"hypotheses": [], "assumptions": ["Revenue is comparable across channels"], "analyses": [
                {"id": "revenue", "kind": "kpi", "dataset": "orders",
                 "group_by": ["channel"], "measures": ["amount"]}]},
            {"observations": [{"evidence_ids": ["revenue.input_rows"],
                               "interpretation": "The configured purchase population was analyzed.",
                               "suggested_action": "Review channel definitions."}], "caveats": []},
        ])

        def endpoint(request):
            captured.append(request.content.decode())
            return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(next(replies))}}]})

        with patch.dict(os.environ, {"DS_LLM_BASE_URL": "http://localhost:8080/v1", "DS_LLM_MODEL": "test"}):
            state = Runner(self.config, self.task, self.output, transport=httpx.MockTransport(endpoint)).run()
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state["llm_calls"], 2)
        self.assertNotIn("synthetic0@example.test", str(captured))
        self.assertNotIn("C00001", str(captured))
        self.assertTrue((self.output / "interpretation.json").exists())


if __name__ == "__main__":
    unittest.main()
