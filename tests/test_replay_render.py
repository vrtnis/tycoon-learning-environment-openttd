from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from openttd_le.cli import main
from openttd_le.replay_render import render_core_replay


class ReplayRenderTests(unittest.TestCase):
    def test_render_core_replay_writes_svg_frames_from_episode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(
                    main(["eval", "--agent", "candidate_rank", "--scenario", "coal_easy_001", "--out", tmp, "--max-steps", "2"]),
                    0,
                )
            summary = json.loads(stdout.getvalue().splitlines()[0])
            run_dir = Path(summary["run_dir"])
            out_dir = Path(tmp) / "frames"

            payload = render_core_replay(episode=run_dir / "episode.jsonl", out=out_dir)

            self.assertEqual(payload["frames"], 2)
            self.assertTrue((out_dir / "frame_0001.svg").exists())
            self.assertTrue((out_dir / "index.html").exists())

    def test_render_core_replay_cli_resolves_episode_from_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(
                    main(["eval", "--agent", "candidate_rank", "--scenario", "coal_easy_001", "--out", tmp, "--max-steps", "1"]),
                    0,
                )
            summary = json.loads(stdout.getvalue().splitlines()[0])
            run_dir = Path(summary["run_dir"])
            out_dir = Path(tmp) / "frames_cli"

            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(["render-core-replay", "--replay", str(run_dir / "replay.json"), "--out", str(out_dir)])

            self.assertEqual(code, 0)
            self.assertTrue((out_dir / "frame_0001.svg").exists())


if __name__ == "__main__":
    unittest.main()
