from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from openttd_le.research.validity import ValidityConfig, load_validity_suite, run_validity_pack


class ValidityPackTests(unittest.TestCase):
    def test_default_validity_suite_loads_known_tasks(self) -> None:
        suite = load_validity_suite()
        self.assertIn("lab_raw_to_processor", suite.tasks)
        self.assertGreaterEqual(len(suite.tasks), 10)
        self.assertTrue(suite.benchmark_file.exists())
        self.assertIn("first_valid", suite.agents)

    def test_validity_pack_writes_manifest_when_sections_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = run_validity_pack(
                ValidityConfig(
                    workbook=Path("scenario.xlsx"),
                    tasks=("lab_raw_to_processor",),
                    agents=("first_valid",),
                    seeds=(1,),
                    output_root=Path(tmp),
                    skip_determinism=True,
                    skip_baselines=True,
                    skip_throughput=True,
                    skip_route_builder=True,
                )
            )

            self.assertTrue(payload["ok"])
            self.assertTrue((Path(tmp) / "suite_manifest.json").exists())
            self.assertTrue((Path(tmp) / "benchmark_report.md").exists())
            report = json.loads((Path(tmp) / "validity_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["tasks"], ["lab_raw_to_processor"])
            self.assertTrue(report["sections"]["determinism"]["skipped"])
            self.assertIn("benchmark_report", report["artifacts"])


if __name__ == "__main__":
    unittest.main()
