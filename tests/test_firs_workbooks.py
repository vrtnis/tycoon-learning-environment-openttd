from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from openttd_le.backends.firs import (
    FIRSInstall,
    FIRSRunConfig,
    cfg_entry_for_newgrf,
    render_firs_live_config,
    render_newgrf_section,
)
from openttd_le.backends.live import (
    _candidate_firs_pairs_from_io,
    _choose_firs_action,
    _firs_reward_snapshot,
    _firs_step_reward,
    _parse_json,
    _route_already_registered,
)
from openttd_le.replay import export_replay, replay_actions
from openttd_le.workbooks.export import export_run_to_xlsx
from openttd_le.workbooks.template import create_firs_ops_workbook, read_firs_ops_workbook


class FIRSWorkbookTests(unittest.TestCase):
    def test_template_round_trips_scenario_and_objectives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scenario.xlsx"
            create_firs_ops_workbook(path)

            config, meta = read_firs_ops_workbook(path)

            self.assertEqual(config.economy, "basic_temperate")
            self.assertEqual(config.economy_parameter, 0)
            self.assertEqual(meta["objectives"][0]["source_type"], "Coal Mine")
            self.assertEqual(meta["objectives"][1]["cargo"], "STEL")

    def test_export_run_to_xlsx_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workbook = root / "scenario.xlsx"
            run_dir = root / "run"
            run_dir.mkdir()
            create_firs_ops_workbook(workbook)
            (run_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "completed": True,
                        "model": "gpt-5.5",
                        "seed": 1,
                        "economy": "basic_arctic",
                        "run_dir": str(run_dir),
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "firs_trace.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"event": "action", "step": 1, "data": {"type": "build_cargo_route"}}),
                        json.dumps(
                            {
                                "event": "observation",
                                "step": 1,
                                "data": {
                                    "company_finances": {"bank_balance": 12345},
                                    "routes": [
                                        {
                                            "route_id": "route_001",
                                            "cargo_label": "WOOD",
                                            "source_name": "Forest",
                                            "destination_name": "Sawmill",
                                            "vehicles": 5,
                                            "delivered": 10,
                                            "profit": 100,
                                            "source_waiting": 0,
                                        }
                                    ],
                                },
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            report = export_run_to_xlsx(run_dir, root / "report.xlsx", source_workbook=workbook)

            self.assertTrue(report.exists())
            _, meta = read_firs_ops_workbook(workbook)
            self.assertEqual(meta["objectives"][0]["cargo"], "COAL")

    def test_export_replay_writes_programs_actions_and_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()
            (run_dir / "summary.json").write_text(
                json.dumps({"objective": "firs_ops_chain", "model": "gpt-5.5", "seed": 1, "economy": "basic_temperate"}),
                encoding="utf-8",
            )
            (run_dir / "launch.json").write_text(
                json.dumps({"recording": str(run_dir / "gameplay.mp4"), "report": str(run_dir / "report.xlsx")}),
                encoding="utf-8",
            )
            (run_dir / "firs_trace.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"event": "initial_observation", "data": {"tick": 0}}),
                        json.dumps({"event": "repl_program", "step": 1, "data": {"code": "print('build')"}}),
                        json.dumps({"event": "repl_feedback", "step": 1, "data": {"stdout": "build\n", "stderr": "", "actions": 1}}),
                        json.dumps({"event": "action", "step": 1, "data": {"type": "build_cargo_route"}}),
                        json.dumps({"event": "result", "step": 1, "data": {"type": "result", "route_id": "route_1"}}),
                        json.dumps({"event": "observation", "step": 1, "data": {"routes": [{"route_id": "route_1"}]}}),
                    ]
                ),
                encoding="utf-8",
            )

            path = export_replay(run_dir)
            replay = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(replay["schema"], "openttd-le-replay-v1")
            self.assertEqual(replay["scenario"]["model"], "gpt-5.5")
            self.assertEqual(replay["steps"][0]["program"], "print('build')")
            self.assertEqual(replay["steps"][0]["actions"][0]["type"], "build_cargo_route")
            self.assertEqual(replay["steps"][0]["results"][0]["route_id"], "route_1")
            self.assertEqual(replay_actions(replay)[0]["type"], "build_cargo_route")


class FIRSConfigTests(unittest.TestCase):
    def test_newgrf_section_uses_economy_parameter(self) -> None:
        install = FIRSInstall(
            user_dir=Path("OpenTTD"),
            newgrf_path=Path("OpenTTD/content_download/newgrf/firs.tar"),
            cfg_entry="content_download/newgrf/firs.tar",
        )
        section = render_newgrf_section(install, FIRSRunConfig(economy="steeltown"))
        self.assertIn("[newgrf]", section)
        self.assertIn("content_download/newgrf/firs.tar = 3", section)

    def test_live_config_includes_firs_newgrf_and_script(self) -> None:
        install = FIRSInstall(Path("OpenTTD"), Path("OpenTTD/newgrf/firs.tar"), "newgrf/firs.tar")
        config = render_firs_live_config(
            run_config=FIRSRunConfig(economy="basic_arctic"),
            install=install,
            game_port=3979,
            admin_port=3977,
            admin_password="pw",
        )
        self.assertIn("newgrf/firs.tar = 1", config)
        self.assertIn("OpenTTDLEGameScript =", config)

    def test_tar_config_entry_uses_inner_grf_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "newgrf" / "firs.tar"
            archive_path.parent.mkdir()
            import tarfile

            payload = root / "firs.grf"
            payload.write_bytes(b"placeholder")
            with tarfile.open(archive_path, "w") as archive:
                archive.add(payload, arcname="FIRS_Industries_5-5.2.0/firs.grf")

            self.assertEqual(cfg_entry_for_newgrf(root, archive_path), "FIRS_Industries_5-5.2.0/firs.grf")


class FIRSActionTests(unittest.TestCase):
    def test_parse_json_accepts_firs_actions(self) -> None:
        action = _parse_json('{"type":"build_cargo_route","source_id":1,"destination_id":2,"cargo_id":3,"vehicles":5}')
        self.assertEqual(action["type"], "build_cargo_route")

    def test_heuristic_uses_workbook_objective(self) -> None:
        action = _choose_firs_action(
            {
                "routes": [],
                "industry_graph": [
                    {
                        "source_id": 11,
                        "source_type": "Forest",
                        "destination_id": 12,
                        "destination_type": "Sawmill",
                        "cargo_id": 7,
                        "cargo": "WOOD",
                    }
                ],
            },
            workbook_meta={
                "objectives": [
                    {
                        "source_type": "Forest",
                        "destination_type": "Sawmill",
                        "cargo": "WOOD",
                    }
                ]
            },
            model="gpt-5.5",
            allow_heuristic=True,
            vehicles_per_route=5,
        )
        self.assertEqual(action["type"], "build_cargo_route")
        self.assertEqual(action["source_id"], 11)
        self.assertEqual(action["vehicles"], 5)
        self.assertIs(action["physical"], True)
        self.assertIs(action["allow_virtual"], False)
        self.assertGreaterEqual(action["max_path_tiles"], 256)

    def test_io_candidate_fallback_matches_objective_chain(self) -> None:
        candidates = _candidate_firs_pairs_from_io(
            {
                "industry_outputs": [
                    {
                        "industry_id": 29,
                        "industry_name": "Fort Drardingworth Coal Mine",
                        "cargo_id": 2,
                        "cargo_label": "COAL",
                        "cargo_name": "Coal",
                        "production": 120,
                    }
                ],
                "industry_inputs": [
                    {
                        "industry_id": 12,
                        "industry_name": "Little Grintburg Steel Mill",
                        "cargo_id": 2,
                        "cargo_label": "COAL",
                        "cargo_name": "Coal",
                    }
                ],
            },
            [{"source_type": "Coal Mine", "destination_type": "Steel Mill", "cargo": "COAL"}],
            limit=5,
        )
        self.assertEqual(candidates[0]["source_id"], 29)
        self.assertEqual(candidates[0]["destination_id"], 12)

    def test_duplicate_route_detection(self) -> None:
        self.assertTrue(
            _route_already_registered(
                {"source_id": 29, "destination_id": 12, "cargo_id": 2},
                [{"source_id": 29, "destination_id": 12, "cargo_id": 2}],
            )
        )

    def test_reward_snapshot_tracks_fle_style_milestones(self) -> None:
        workbook_meta = {"objectives": [{"cargo": "COAL"}, {"cargo": "STEL"}]}
        before = _firs_reward_snapshot({"routes": []}, workbook_meta)
        after = _firs_reward_snapshot(
            {
                "routes": [
                    {"cargo_label": "COAL", "delivered": 20, "profit": 1500},
                    {"cargo_label": "STEL", "delivered": 1, "profit": 100},
                ]
            },
            workbook_meta,
        )
        reward = _firs_step_reward(before, after, [])

        self.assertTrue(after["milestones"]["first_route"])
        self.assertTrue(after["milestones"]["first_delivery"])
        self.assertTrue(after["milestones"]["first_processed_delivery"])
        self.assertIn("first_processed_delivery", reward["new_milestones"])
        self.assertGreater(reward["reward"], 0)


if __name__ == "__main__":
    unittest.main()
