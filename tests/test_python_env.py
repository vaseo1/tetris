import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tetris_ai.afterstates import apply_actions, enumerate_placements
from tetris_ai.engine import ACTIONS, COLS, ROWS, Piece, create_game, get_state, step_game
from tetris_ai.features import FEATURE_SIZE, board_metrics, feature_vector
from tetris_ai.train import evaluate, resolved_eval_workers

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
        step_game(game, ACTIONS.get(action, action))
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


class TrainingSmokeTest(unittest.TestCase):
    def test_eval_worker_resolution(self):
        self.assertEqual(resolved_eval_workers(1, 200), 1)
        self.assertEqual(resolved_eval_workers(8, 2), 2)
        self.assertGreaterEqual(resolved_eval_workers(0, 200), 1)

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
            self.assertTrue((Path(tmp) / "metrics.jsonl").exists())
            self.assertTrue((Path(tmp) / "latest-model.json").exists())
            self.assertTrue((Path(tmp) / "best-replay.json").exists())
            self.assertTrue((Path(tmp) / "checkpoint.pt").exists())

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
