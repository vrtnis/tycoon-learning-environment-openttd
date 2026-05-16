from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from openttd_le.cli import main


class CliTests(unittest.TestCase):
    def test_eval_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "eval",
                        "--agent",
                        "greedy",
                        "--scenario",
                        "coal_easy_001",
                        "--out",
                        tmp,
                    ]
                )
            self.assertEqual(code, 0)
            summary = json.loads(stdout.getvalue().strip().splitlines()[0])
            run_dir = Path(summary["run_dir"])
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "actions.jsonl").exists())
            self.assertTrue((run_dir / "metrics.csv").exists())
            self.assertTrue((run_dir / "final_state.json").exists())
            self.assertTrue((run_dir / "screenshots" / "final_map.svg").exists())


if __name__ == "__main__":
    unittest.main()
