import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tetris_ai.afterstates import apply_actions, enumerate_placements, reward_from_metrics
from tetris_ai.engine import ACTIONS, COLS, ROWS, Piece, clone_game, create_game, get_state, hard_drop, soft_drop, step_game
from tetris_ai.features import FEATURE_SIZE, board_metrics, feature_vector
from tetris_ai.model import best_device, make_value_net, require_torch
from tetris_ai.train import evaluate, performance_metrics, resolved_eval_workers

ROOT = Path(__file__).resolve().parents[1]


def js_state(seed, actions=()):
    result = subprocess.run(
        ["node", str(ROOT / "scripts" / "js-state.mjs"), str(seed), ",".join(actions)],
        check=True,
        text=True,
        capture_output=True,
        cwd=ROOT,
    )
    return json.loads(result.stdout)


def py_state(seed, actions=()):
    game = create_game(seed)
    for action in actions:
        resolved_action = ACTIONS[action] if action in ACTIONS else action
        step_game(game, resolved_action)
    return get_state(game)


def js_scenario(name):
    result = subprocess.run(
        ["node", str(ROOT / "scripts" / "js-scenario.mjs"), name],
        check=True,
        text=True,
        capture_output=True,
        cwd=ROOT,
    )
    return json.loads(result.stdout)


class PythonEngineParityTest(unittest.TestCase):
    def test_seeded_initial_state_matches_js(self):
        self.assertEqual(py_state("agent-parity"), js_state("agent-parity"))

    def test_representative_action_trace_matches_js(self):
        actions = ("left", "rotate", "right", "down", "hardDrop", "rotate", "hardDrop")
        self.assertEqual(py_state(42, actions), js_state(42, actions))

    def test_line_clear_scenario_matches_js(self):
        game = create_game(1)
        game.board[ROWS - 2] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        game.board[ROWS - 1] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
        game.active_piece = Piece("O", [[1, 0], [2, 0], [1, 1], [2, 1]], 7, ROWS - 2)
        step_game(game, ACTIONS["hardDrop"])

        self.assertEqual(get_state(game), js_scenario("line-clear"))

    def test_top_out_scenario_matches_js(self):
        game = create_game(1)
        game.board[0][4] = 1
        game.next_piece = Piece("O", [[1, 0], [2, 0], [1, 1], [2, 1]], 3, 0)
        step_game(game, ACTIONS["hardDrop"])

        self.assertEqual(get_state(game), js_scenario("top-out"))

    def test_hard_drop_matches_repeated_soft_drops(self):
        prefixes = [
            (),
            (ACTIONS["left"],),
            (ACTIONS["rotate"], ACTIONS["left"]),
            (ACTIONS["right"], ACTIONS["right"], ACTIONS["rotate"]),
        ]

        for seed in (7, "hard-drop-a", "hard-drop-b"):
            for prefix in prefixes:
                with self.subTest(seed=seed, prefix=prefix):
                    game = create_game(seed)
                    for action in prefix:
                        step_game(game, action)

                    hard_game = clone_game(game)
                    soft_game = clone_game(game)

                    hard_drop(hard_game)
                    while not soft_game.game_over:
                        active_piece = soft_game.active_piece
                        soft_drop(soft_game)
                        if soft_game.active_piece is not active_piece:
                            break

                    self.assertEqual(get_state(hard_game), get_state(soft_game))


class AfterstateTest(unittest.TestCase):
    def test_enumerates_legal_placements_with_replayable_actions(self):
        game = create_game("placements")
        placements = enumerate_placements(game)

        self.assertGreater(len(placements), 0)
        for placement in placements:
            replayed = apply_actions(game, placement.actions)
            self.assertEqual(tuple(tuple(row) for row in replayed.board), placement.board)
            self.assertEqual(replayed.last_cleared, placement.cleared)

    def test_feature_vector_size_and_metrics(self):
        game = create_game("features")
        placement = enumerate_placements(game)[0]
        vector = placement.vector
        metrics = board_metrics([list(row) for row in placement.board])

        self.assertEqual(len(vector), FEATURE_SIZE)
        self.assertEqual(len(feature_vector([list(row) for row in placement.board], placement.next_piece)), FEATURE_SIZE)
        self.assertIn("holes", metrics)
        self.assertIn("bumpiness", metrics)

    def test_phase2_reward_increases_line_clear_incentive(self):
        metrics = {
            "holes": 0,
            "maxHeight": 4,
            "aggregateHeight": 12,
            "bumpiness": 2,
            "completeLines": 0,
            "wells": 0,
            "filledCells": 20,
        }

        survival_reward = reward_from_metrics(metrics, 4, False, "survival")
        phase2_reward = reward_from_metrics(metrics, 4, False, "phase2-score")

        self.assertGreater(phase2_reward, survival_reward)


class TrainingSmokeTest(unittest.TestCase):
    def test_eval_worker_resolution(self):
        self.assertEqual(resolved_eval_workers(1, 200), 1)
        self.assertEqual(resolved_eval_workers(8, 2), 2)
        self.assertGreaterEqual(resolved_eval_workers(0, 200), 1)

    def test_performance_metrics_use_completed_steps(self):
        metrics = performance_metrics(100, 5, 3600.0)

        self.assertEqual(metrics["stepsPerHour"], 100.0)
        self.assertEqual(metrics["episodesPerHour"], 5.0)

    def test_evaluation_reports_score_stats(self):
        class DummyTorch:
            class no_grad:
                def __enter__(self):
                    return None

                def __exit__(self, *args):
                    return None

            @staticmethod
            def tensor(values, **_kwargs):
                return values

            @staticmethod
            def argmax(_values):
                class Index:
                    @staticmethod
                    def item():
                        return 0

                return Index()

            float32 = "float32"

        class DummyModel:
            def __call__(self, batch):
                return [0 for _ in batch]

        result = evaluate(DummyModel(), DummyTorch, "cpu", ["score-stats"], 1.4)

        self.assertIn("meanScore", result)
        self.assertIn("medianScore", result)
        self.assertIn("maxScore", result)
        self.assertIn("meanLinesCleared", result)
        self.assertIn("maxLinesCleared", result)

    def test_parallel_evaluation_runs_when_torch_is_available(self):
        try:
            torch, _ = require_torch()
        except SystemExit:
            self.skipTest("PyTorch is not installed")

        device = best_device(torch)
        model = make_value_net().to(device)
        model.eval()

        result = evaluate(
            model,
            torch,
            device,
            ["parallel-eval-0", "parallel-eval-1", "parallel-eval-2", "parallel-eval-3"],
            0.2,
            eval_workers=2,
        )

        self.assertIn("episodes", result)
        self.assertEqual(len(result["episodes"]), 4)

    def test_tiny_training_run_writes_artifacts_when_torch_is_available(self):
        if shutil.which("python3") is None:
            self.skipTest("python3 is not available")

        probe = subprocess.run(
            [sys.executable, "-c", "import torch"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if probe.returncode != 0:
            self.skipTest("PyTorch is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tetris_ai.train",
                    "--episodes",
                    "2",
                    "--max-pieces",
                    "8",
                    "--batch-size",
                    "2",
                    "--eval-interval",
                    "1",
                    "--eval-seeds",
                    "2",
                    "--output-dir",
                    tmp,
                ],
                check=True,
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            metrics_lines = [
                json.loads(line)
                for line in (Path(tmp) / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            train_metrics = next(metric for metric in metrics_lines if metric["type"] == "trainEpisode")
            self.assertTrue((Path(tmp) / "metrics.jsonl").exists())
            self.assertTrue((Path(tmp) / "latest-model.json").exists())
            self.assertTrue((Path(tmp) / "best-replay.json").exists())
            self.assertTrue((Path(tmp) / "checkpoint.pt").exists())
            self.assertIn("stepsPerSecond", train_metrics)
            self.assertIn("stepsPerHour", train_metrics)
            self.assertIn("episodesPerHour", train_metrics)

    def test_training_resume_continues_from_checkpoint_when_torch_is_available(self):
        probe = subprocess.run(
            [sys.executable, "-c", "import torch"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if probe.returncode != 0:
            self.skipTest("PyTorch is not installed")

        base_command = [
            sys.executable,
            "-m",
            "tetris_ai.train",
            "--episodes",
            "1",
            "--max-pieces",
            "4",
            "--batch-size",
            "2",
            "--eval-interval",
            "1",
            "--eval-seeds",
            "2",
        ]
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                [*base_command, "--output-dir", tmp],
                check=True,
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            resumed = subprocess.run(
                [*base_command, "--output-dir", tmp, "--resume"],
                check=True,
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

            self.assertIn("Resumed checkpoint from episode 1", resumed.stdout)
            metrics = (Path(tmp) / "metrics.jsonl").read_text(encoding="utf-8")
            self.assertIn('"episode": 0', metrics)
            self.assertIn('"episode": 1', metrics)


if __name__ == "__main__":
    unittest.main()
