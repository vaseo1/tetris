import json
import random
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tetris_ai.afterstates import apply_actions, enumerate_placements, reward_from_metrics
from tetris_ai.engine import ACTIONS, COLS, ROWS, Game, Piece, clone_game, create_game, get_state, hard_drop, soft_drop, step_game
from tetris_ai.features import FEATURE_SIZE, board_metrics, feature_vector
from tetris_ai.model import best_device, make_value_net, require_torch
from tetris_ai.recovery import create_recovery_game, make_recovery_board, recovery_summary
from tetris_ai.train import (
    best_tracking_values,
    choose_placement,
    create_start_game,
    default_init_model_step,
    evaluate,
    export_model,
    load_exported_model,
    model_export_metadata,
    performance_metrics,
    resolved_eval_workers,
)

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

    def test_restart_uses_fresh_seed(self):
        game = create_game(7)
        start_seed = game.seed
        original_randrange = random.randrange
        random.randrange = lambda _: 123456789
        try:
            step_game(game, ACTIONS["restart"])
        finally:
            random.randrange = original_randrange

        state = get_state(game)
        self.assertEqual(state["status"], "READY")
        self.assertNotEqual(state["seed"], start_seed)

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
            "coveredHoles": 0,
            "maxHeight": 4,
            "aggregateHeight": 12,
            "bumpiness": 2,
            "completeLines": 0,
            "wells": 0,
            "filledCells": 20,
            "topZoneCells": 0,
            "dangerZoneCells": 0,
        }

        survival_reward = reward_from_metrics(metrics, 4, False, "survival")
        phase2_reward = reward_from_metrics(metrics, 4, False, "phase2-score")

        self.assertGreater(phase2_reward, survival_reward)

    def test_survival_v2_penalizes_top_danger_and_terminal_states(self):
        stable_metrics = {
            "holes": 2,
            "coveredHoles": 4,
            "maxHeight": 8,
            "aggregateHeight": 40,
            "bumpiness": 4,
            "completeLines": 0,
            "wells": 0,
            "filledCells": 40,
            "topZoneCells": 0,
            "dangerZoneCells": 0,
        }
        dangerous_metrics = {
            **stable_metrics,
            "holes": 8,
            "coveredHoles": 60,
            "maxHeight": 17,
            "wells": 8,
            "topZoneCells": 3,
            "dangerZoneCells": 8,
        }

        stable_reward = reward_from_metrics(stable_metrics, 0, False, "survival-v2")
        dangerous_reward = reward_from_metrics(dangerous_metrics, 0, False, "survival-v2")
        terminal_reward = reward_from_metrics(dangerous_metrics, 0, True, "survival-v2")

        self.assertLess(dangerous_reward, stable_reward)
        self.assertLess(terminal_reward, dangerous_reward - 10.0)

    def test_choose_placement_filters_terminal_moves_when_safe_move_exists(self):
        try:
            torch, _ = require_torch()
        except SystemExit:
            self.skipTest("PyTorch is not installed")

        board = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
            [0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        ]
        game = Game(
            COLS,
            ROWS,
            board,
            Piece("Z", [[0, 0], [1, 0], [1, 1], [2, 1]], 3, 0),
            Piece("J", [[0, 0], [0, 1], [1, 1], [2, 1]], 3, 0),
            110,
            110,
            False,
            False,
            "PLAY",
            83,
            0,
            0,
        )
        placements = enumerate_placements(game, "survival-v2")
        terminal_placement = next(placement for placement in placements if placement.done)
        self.assertTrue(any(not placement.done for placement in placements))

        class TerminalPreferenceModel:
            def __init__(self, vector):
                self.vector = torch.tensor(vector, dtype=torch.float32)

            def __call__(self, batch):
                matches = torch.isclose(batch, self.vector.to(batch.device)).all(dim=1)
                return torch.where(
                    matches,
                    torch.full_like(matches, 100.0, dtype=torch.float32),
                    torch.zeros_like(matches, dtype=torch.float32),
                )

        placement = choose_placement(
            TerminalPreferenceModel(terminal_placement.vector),
            torch,
            torch.device("cpu"),
            game,
            epsilon=0.0,
            reward_profile="survival-v2",
        )

        self.assertIsNotNone(placement)
        self.assertFalse(placement.done)


class RecoveryStartTest(unittest.TestCase):
    def test_recovery_board_is_deterministic_and_damaged(self):
        first = make_recovery_board("recovery-seed", "medium")
        second = make_recovery_board("recovery-seed", "medium")
        metrics = board_metrics(first)

        self.assertEqual(first, second)
        self.assertGreater(metrics["maxHeight"], 0)
        self.assertGreater(metrics["holes"], 0)
        self.assertFalse(any(all(row) for row in first))

    def test_recovery_game_is_playable(self):
        game = create_recovery_game("playable-recovery", "hard")
        placements = enumerate_placements(game)

        self.assertFalse(game.game_over)
        self.assertGreater(len(placements), 0)
        self.assertEqual(game.status, "PLAY")

    def test_mixed_recovery_severity_is_reproducible(self):
        first = recovery_summary("mixed-seed", "mixed")
        second = recovery_summary("mixed-seed", "mixed")

        self.assertEqual(first, second)
        self.assertIn(first["severity"], ("easy", "medium", "hard"))

    def test_start_game_can_create_recovery_mode(self):
        clean = create_start_game("start-mode", "clean")
        recovery = create_start_game("start-mode", "recovery", "medium")

        self.assertEqual(board_metrics(clean.board)["filledCells"], 0)
        self.assertGreater(board_metrics(recovery.board)["filledCells"], 0)


class TrainingSmokeTest(unittest.TestCase):
    def test_eval_worker_resolution(self):
        self.assertEqual(resolved_eval_workers(1, 200), 1)
        self.assertEqual(resolved_eval_workers(8, 2), 2)
        self.assertGreaterEqual(resolved_eval_workers(0, 200), 1)

    def test_performance_metrics_use_completed_steps(self):
        metrics = performance_metrics(100, 5, 3600.0, total_episodes=10)

        self.assertEqual(metrics["stepsPerHour"], 100.0)
        self.assertEqual(metrics["stepsPerEpisode"], 20.0)
        self.assertEqual(metrics["episodesPerHour"], 5.0)
        self.assertEqual(metrics["estimatedHoursLeft"], 1.0)

    def test_model_export_metadata_uses_completed_episode_count(self):
        metadata = model_export_metadata(episode_index=4, exported_at="2026-05-13T17:59:00Z")

        self.assertEqual(metadata["episodes"], 5)
        self.assertEqual(metadata["exportedAt"], "2026-05-13T17:59:00Z")

    def test_init_model_uses_mature_exploration_step_by_default(self):
        class Args:
            eps_start = 1.0
            eps_end = 0.05
            eps_decay = 12000

        self.assertEqual(default_init_model_step(Args), 120000)

    def test_best_tracking_values_use_recovery_objective_when_requested(self):
        class Args:
            best_model_objective = "recovery"

        clean_eval = {
            "medianSurvivalSeconds": 14400.0,
            "topOutRate": 0.4,
            "meanScore": 6000.0,
        }
        recovery_eval = {
            "medianSurvivalSeconds": 300.0,
            "topOutRate": 0.2,
            "meanScore": 120.0,
        }

        self.assertEqual(best_tracking_values(Args, clean_eval, recovery_eval), (300.0, 0.2, 120.0))

    def test_exported_json_can_initialize_model_when_torch_is_available(self):
        try:
            torch, _ = require_torch()
        except SystemExit:
            self.skipTest("PyTorch is not installed")

        device = best_device(torch)
        source = make_value_net().to(device)
        target = make_value_net().to(device)

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "best-model.json"
            export_model(source, torch, model_path, model_export_metadata(episode_index=2))
            metadata = load_exported_model(target, torch, model_path)

        self.assertEqual(metadata["episodes"], 3)
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value.cpu(), target.state_dict()[key].cpu()))

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

    def test_recovery_evaluation_reports_start_mode(self):
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

        result = evaluate(
            DummyModel(),
            DummyTorch,
            "cpu",
            ["recovery-score-stats"],
            1.4,
            start_mode="recovery",
            recovery_severity="medium",
        )

        self.assertEqual(result["startMode"], "recovery")
        self.assertIn("topOutRate", result)

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
                    "--checkpoint-dir",
                    str(Path(tmp) / "checkpoints"),
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
            self.assertTrue((Path(tmp) / "checkpoints" / "checkpoint.pt.gz").exists())
            self.assertTrue((Path(tmp) / "checkpoints" / "checkpoint-best.pt.gz").exists())
            self.assertIn("stepsPerSecond", train_metrics)
            self.assertIn("stepsPerHour", train_metrics)
            self.assertIn("stepsPerEpisode", train_metrics)
            self.assertIn("episodesPerHour", train_metrics)
            self.assertIn("estimatedHoursLeft", train_metrics)
            latest_model = json.loads((Path(tmp) / "latest-model.json").read_text(encoding="utf-8"))
            self.assertEqual(latest_model["metadata"]["episodes"], 2)
            self.assertRegex(latest_model["metadata"]["exportedAt"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

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
            checkpoint_dir = Path(tmp) / "checkpoints"
            subprocess.run(
                [*base_command, "--output-dir", tmp, "--checkpoint-dir", str(checkpoint_dir)],
                check=True,
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            resumed = subprocess.run(
                [*base_command, "--output-dir", tmp, "--checkpoint-dir", str(checkpoint_dir), "--resume"],
                check=True,
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

            self.assertIn("Resumed checkpoint from episode 1", resumed.stdout)
            metrics = (Path(tmp) / "metrics.jsonl").read_text(encoding="utf-8")
            self.assertIn('"episode": 0', metrics)
            self.assertIn('"episode": 1', metrics)

    def test_training_resume_best_continues_from_best_checkpoint_when_torch_is_available(self):
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
            checkpoint_dir = Path(tmp) / "checkpoints"
            subprocess.run(
                [*base_command, "--output-dir", tmp, "--checkpoint-dir", str(checkpoint_dir)],
                check=True,
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            resumed = subprocess.run(
                [*base_command, "--output-dir", tmp, "--checkpoint-dir", str(checkpoint_dir), "--resume-best"],
                check=True,
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

            self.assertIn("Resumed best checkpoint from episode 1", resumed.stdout)

    def test_training_init_model_starts_fresh_phase_when_torch_is_available(self):
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
            first_output = Path(tmp) / "first"
            second_output = Path(tmp) / "second"
            subprocess.run(
                [*base_command, "--output-dir", str(first_output), "--checkpoint-dir", str(first_output / "checkpoints")],
                check=True,
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            initialized = subprocess.run(
                [
                    *base_command,
                    "--output-dir",
                    str(second_output),
                    "--checkpoint-dir",
                    str(second_output / "checkpoints"),
                    "--init-model",
                    str(first_output / "best-model.json"),
                ],
                check=True,
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

            self.assertIn(f"Initialized model from {first_output / 'best-model.json'}", initialized.stdout)
            self.assertIn("Initialized exploration schedule at step 120000", initialized.stdout)
            self.assertIn("Initialized best tracking from source model", initialized.stdout)
            metrics = [
                json.loads(line)
                for line in (second_output / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            train_metric = next(metric for metric in metrics if metric["type"] == "trainEpisode")
            self.assertEqual(train_metric["episode"], 0)
            self.assertLess(train_metric["epsilon"], 0.051)
            self.assertFalse((second_output / "checkpoints" / "checkpoint.pt.gz").exists())
            self.assertTrue((second_output / "checkpoints" / "checkpoint-init-model.pt.gz").exists())

    def test_evaluate_cli_accepts_eval_workers_when_torch_is_available(self):
        probe = subprocess.run(
            [sys.executable, "-c", "import torch"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if probe.returncode != 0:
            self.skipTest("PyTorch is not installed")

        try:
            torch, _ = require_torch()
        except SystemExit:
            self.skipTest("PyTorch is not installed")

        device = best_device(torch)
        model = make_value_net().to(device)
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.json"
            export_model(model, torch, model_path, model_export_metadata(episode_index=0))
            failures_path = Path(tmp) / "failures.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tetris_ai.evaluate",
                    str(model_path),
                    "--seeds",
                    "2",
                    "--seconds",
                    "0.2",
                    "--eval-workers",
                    "1",
                    "--failures-output",
                    str(failures_path),
                ],
                check=True,
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
            failures = json.loads(failures_path.read_text(encoding="utf-8"))

        self.assertIn('"successRate"', result.stdout)
        self.assertIsInstance(failures, list)

    def test_training_rejects_ambiguous_resume_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tetris_ai.train",
                    "--episodes",
                    "0",
                    "--output-dir",
                    tmp,
                    "--resume",
                    "--init-model",
                    str(Path(tmp) / "best-model.json"),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not allowed with argument", result.stderr)

    def test_training_resume_best_missing_checkpoint_has_clear_message_when_torch_is_available(self):
        probe = subprocess.run(
            [sys.executable, "-c", "import torch"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if probe.returncode != 0:
            self.skipTest("PyTorch is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "tetris_ai.train",
                    "--episodes",
                    "0",
                    "--output-dir",
                    tmp,
                    "--checkpoint-dir",
                    str(Path(tmp) / "checkpoints"),
                    "--resume-best",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No best checkpoint found", result.stderr)
        self.assertIn("--init-model runs/tetris-agent/best-model.json", result.stderr)


if __name__ == "__main__":
    unittest.main()
