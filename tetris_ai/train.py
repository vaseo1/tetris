from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from .afterstates import REWARD_PROFILES, Placement, apply_actions, enumerate_placements
from .engine import Game, create_game, get_state
from .features import FEATURE_SIZE, board_metrics
from .model import best_device, make_value_net, require_torch
from .recovery import RECOVERY_SEVERITIES, create_recovery_game

GRAVITY_SECONDS = 0.7
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/tetris-agent")
CHECKPOINT_FILENAME = "checkpoint.pt.gz"
HELD_OUT_SEEDS = [f"heldout-{index}" for index in range(200)]
_EVAL_MODEL = None
_EVAL_TORCH = None
_EVAL_DEVICE = "cpu"


@dataclass
class Transition:
    state: list[float]
    reward: float
    next_states: list[list[float]]
    done: bool


class PrioritizedReplay:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.items: list[Transition] = []
        self.priorities: list[float] = []
        self.next_index = 0

    def __len__(self) -> int:
        return len(self.items)

    def add(self, transition: Transition, priority: float | None = None) -> None:
        value = priority if priority is not None else (max(self.priorities) if self.priorities else 1.0)
        if len(self.items) < self.capacity:
            self.items.append(transition)
            self.priorities.append(value)
        else:
            self.items[self.next_index] = transition
            self.priorities[self.next_index] = value
            self.next_index = (self.next_index + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4):
        weights = [priority**self.alpha for priority in self.priorities]
        total = sum(weights)
        probabilities = [weight / total for weight in weights]
        indices = random.choices(range(len(self.items)), probabilities, k=batch_size)
        samples = [self.items[index] for index in indices]
        importance = [(len(self.items) * probabilities[index]) ** (-beta) for index in indices]
        max_importance = max(importance)
        importance = [value / max_importance for value in importance]
        return indices, samples, importance

    def update_priorities(self, indices: list[int], priorities) -> None:
        for index, priority in zip(indices, priorities):
            self.priorities[index] = float(abs(priority)) + 1e-5

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "alpha": self.alpha,
            "items": [asdict(item) for item in self.items],
            "priorities": self.priorities,
            "next_index": self.next_index,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.capacity = int(state["capacity"])
        self.alpha = float(state["alpha"])
        self.items = [Transition(**item) for item in state["items"]]
        self.priorities = [float(priority) for priority in state["priorities"]]
        self.next_index = int(state["next_index"])


def choose_placement(model, torch, device, game: Game, epsilon: float, reward_profile: str = "survival") -> Placement | None:
    placements = enumerate_placements(game, reward_profile)
    if not placements:
        return None
    if random.random() < epsilon:
        return random.choice(placements)
    with torch.no_grad():
        batch = torch.tensor([placement.vector for placement in placements], dtype=torch.float32, device=device)
        values = model(batch)
        return placements[int(torch.argmax(values).item())]


def serialize_model_state(model, torch) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "values": value.detach().cpu().tolist(),
            "dtype": str(value.dtype).removeprefix("torch."),
        }
        for key, value in model.state_dict().items()
    }


def deserialize_model_state(torch, model_state: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        key: torch.tensor(value["values"], dtype=getattr(torch, value["dtype"]))
        for key, value in model_state.items()
    }


def next_vectors(game: Game, reward_profile: str = "survival") -> list[list[float]]:
    return [placement.vector for placement in enumerate_placements(game, reward_profile)]


def optimize(model, target_model, optimizer, replay: PrioritizedReplay, torch, device, args) -> float | None:
    if len(replay) < args.batch_size:
        return None

    indices, samples, weights = replay.sample(args.batch_size, args.priority_beta)
    states = torch.tensor([sample.state for sample in samples], dtype=torch.float32, device=device)
    rewards = torch.tensor([sample.reward for sample in samples], dtype=torch.float32, device=device)
    done = torch.tensor([sample.done for sample in samples], dtype=torch.float32, device=device)
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    q_values = model(states)
    targets = []
    with torch.no_grad():
        for sample in samples:
            if sample.done or not sample.next_states:
                targets.append(0.0)
                continue
            next_batch = torch.tensor(sample.next_states, dtype=torch.float32, device=device)
            best_index = int(torch.argmax(model(next_batch)).item())
            targets.append(float(target_model(next_batch)[best_index].item()))

    target_values = rewards + args.gamma * (1.0 - done) * torch.tensor(targets, dtype=torch.float32, device=device)
    td_errors = target_values - q_values
    loss = (torch.nn.functional.smooth_l1_loss(q_values, target_values, reduction="none") * weights_tensor).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    replay.update_priorities(indices, td_errors.detach().abs().cpu().tolist())
    return float(loss.item())


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def model_export_metadata(episode_index: int, exported_at: str | None = None) -> dict[str, Any]:
    return {
        "episodes": episode_index + 1,
        "exportedAt": exported_at or iso_utc_now(),
    }


def export_model(model, torch, path: Path, metadata: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu")
    state = model_cpu.state_dict()
    layers = []
    for prefix in ("layers.0", "layers.2", "layers.4"):
        layers.append(
            {
                "weight": state[f"{prefix}.weight"].tolist(),
                "bias": state[f"{prefix}.bias"].tolist(),
            }
        )
    payload = {
        "type": "afterstate-value-mlp",
        "version": 1,
        "inputSize": FEATURE_SIZE,
        "layers": layers,
        "activation": "relu",
    }
    if metadata is not None:
        payload["metadata"] = metadata
    path.write_text(json.dumps(payload), encoding="utf-8")
    model.to(best_device(torch))


def checkpoint_payload(
    model,
    target_model,
    optimizer,
    replay: PrioritizedReplay,
    episode_index: int,
    global_step: int,
    best_median: float,
    best_top_out: float,
    best_score: float,
    recent_losses: deque,
    args,
) -> dict[str, Any]:
    return {
        "version": 1,
        "episodeIndex": episode_index,
        "globalStep": global_step,
        "model": model.state_dict(),
        "targetModel": target_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "replay": replay.state_dict(),
        "bestMedian": best_median,
        "bestTopOut": best_top_out,
        "bestScore": best_score,
        "recentLosses": list(recent_losses),
        "randomState": random.getstate(),
        "args": vars(args),
    }


def save_checkpoint(torch, checkpoint_path: Path, payload: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")
    if checkpoint_path.suffix == ".gz":
        with gzip.open(temporary_path, "wb", compresslevel=6) as handle:
            torch.save(payload, handle)
    else:
        torch.save(payload, temporary_path)
    os.replace(temporary_path, checkpoint_path)


def load_checkpoint(torch, checkpoint_path: Path) -> dict[str, Any] | None:
    if not checkpoint_path.exists():
        return None
    if checkpoint_path.suffix == ".gz":
        with gzip.open(checkpoint_path, "rb") as handle:
            return torch.load(handle, map_location="cpu", weights_only=False)
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def resolve_checkpoint_path(args) -> Path:
    return Path(args.checkpoint_dir) / CHECKPOINT_FILENAME


def create_start_game(seed: Any, start_mode: str = "clean", recovery_severity: str = "medium") -> Game:
    if start_mode == "clean":
        return create_game(seed)
    if start_mode == "recovery":
        return create_recovery_game(seed, recovery_severity)
    raise ValueError(f"Unknown start mode: {start_mode}")


def choose_training_start_mode(args) -> str:
    if args.recovery_start_rate <= 0:
        return "clean"
    return "recovery" if random.random() < args.recovery_start_rate else "clean"


def evaluate_seed(
    model,
    torch,
    device,
    seed: Any,
    max_seconds: float,
    capture_replay: bool = False,
    start_mode: str = "clean",
    recovery_severity: str = "medium",
):
    max_pieces = max(1, math.ceil(max_seconds / GRAVITY_SECONDS))
    game = create_start_game(seed, start_mode, recovery_severity)
    frames = []
    pieces = 0
    while not game.game_over and pieces < max_pieces:
        placement = choose_placement(model, torch, device, game, epsilon=0.0)
        if placement is None:
            break
        if capture_replay:
            frames.append({"state": get_state(game), "actions": list(placement.actions)})
        game = apply_actions(game, placement.actions)
        pieces += 1

    metrics = board_metrics(game.board)
    result = {
        "seed": str(seed),
        "survivalSeconds": min(max_seconds, pieces * GRAVITY_SECONDS),
        "pieces": pieces,
        "linesCleared": game.lines_cleared,
        "score": game.score,
        "gameOver": game.game_over,
        "maxHeight": metrics["maxHeight"],
        "holes": metrics["holes"],
        "bumpiness": metrics["bumpiness"],
        "startMode": start_mode,
    }
    replay = None
    if capture_replay:
        replay = {
            "seed": str(seed),
            "gravitySeconds": GRAVITY_SECONDS,
            "startMode": start_mode,
            "recoverySeverity": recovery_severity if start_mode == "recovery" else None,
            "frames": frames,
            "finalState": get_state(game),
            "result": result,
        }
    return result, replay


def performance_metrics(completed_steps: int, completed_episodes: int, elapsed_seconds: float) -> dict[str, float]:
    safe_elapsed_seconds = max(elapsed_seconds, 1e-9)
    return {
        "elapsedSeconds": elapsed_seconds,
        "stepsPerSecond": completed_steps / safe_elapsed_seconds,
        "stepsPerHour": (completed_steps * 3600.0) / safe_elapsed_seconds,
        "episodesPerHour": (completed_episodes * 3600.0) / safe_elapsed_seconds,
    }


def init_eval_worker(model_state: dict[str, dict[str, Any]]) -> None:
    global _EVAL_DEVICE, _EVAL_MODEL, _EVAL_TORCH
    _EVAL_TORCH, _ = require_torch()
    _EVAL_TORCH.set_num_threads(1)
    _EVAL_DEVICE = _EVAL_TORCH.device("cpu")
    _EVAL_MODEL = make_value_net().to(_EVAL_DEVICE)
    _EVAL_MODEL.load_state_dict(deserialize_model_state(_EVAL_TORCH, model_state))
    _EVAL_MODEL.eval()


def evaluate_seed_in_worker(task):
    seed, max_seconds, capture_replay, start_mode, recovery_severity = task
    return evaluate_seed(
        _EVAL_MODEL,
        _EVAL_TORCH,
        _EVAL_DEVICE,
        seed,
        max_seconds,
        capture_replay,
        start_mode,
        recovery_severity,
    )


def resolved_eval_workers(eval_workers: int, seed_count: int) -> int:
    if seed_count <= 1:
        return 1
    if eval_workers > 0:
        return max(1, min(eval_workers, seed_count))
    return max(1, min(os.cpu_count() or 1, seed_count))


def summarize_eval_results(results):
    episodes = [result for result, _replay in results]
    best_replay = next((replay for _result, replay in results if replay is not None), None)
    return episodes, best_replay


def evaluate(
    model,
    torch,
    device,
    seeds: list[Any],
    max_seconds: float,
    capture_replay: bool = False,
    eval_workers: int = 1,
    start_mode: str = "clean",
    recovery_severity: str = "medium",
):
    workers = resolved_eval_workers(eval_workers, len(seeds))
    if workers <= 1:
        results = [
            evaluate_seed(
                model,
                torch,
                device,
                seed,
                max_seconds,
                capture_replay and index == 0,
                start_mode,
                recovery_severity,
            )
            for index, seed in enumerate(seeds)
        ]
    else:
        model_state = serialize_model_state(model, torch)
        tasks = [
            (seed, max_seconds, capture_replay and index == 0, start_mode, recovery_severity)
            for index, seed in enumerate(seeds)
        ]
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=init_eval_worker,
            initargs=(model_state,),
        ) as executor:
            results = list(executor.map(evaluate_seed_in_worker, tasks))

    episodes, best_replay = summarize_eval_results(results)
    success_rate = mean(1.0 if episode["survivalSeconds"] >= max_seconds else 0.0 for episode in episodes)
    scores = [episode["score"] for episode in episodes]
    lines = [episode["linesCleared"] for episode in episodes]
    return {
        "startMode": start_mode,
        "successRate": success_rate,
        "meanSurvivalSeconds": mean(episode["survivalSeconds"] for episode in episodes),
        "medianSurvivalSeconds": median(episode["survivalSeconds"] for episode in episodes),
        "topOutRate": mean(1.0 if episode["gameOver"] else 0.0 for episode in episodes),
        "meanScore": mean(scores),
        "medianScore": median(scores),
        "maxScore": max(scores),
        "meanLinesCleared": mean(lines),
        "maxLinesCleared": max(lines),
        "episodes": episodes,
        "replay": best_replay,
    }


def print_training_legend(args) -> None:
    workers = "auto" if args.eval_workers == 0 else str(args.eval_workers)
    print("Training output guide:")
    print(f"  reward_profile = {args.reward_profile};")
    print(f"  recovery_start_rate = {args.recovery_start_rate:.2f};")
    print(f"  eval_success = fraction of held-out seeds that survive {args.milestone_seconds:.0f}s;")
    print("  median = median survival time;")
    print("  mean_score/median_score/max_score = held-out evaluation scores;")
    print(f"  eval_workers = {workers} parallel worker process(es) for held-out evaluation;")
    print("Resume hint:")
    print(f"  uv run npm run train:ai -- --resume --episodes {args.episodes};")
    print("  uv run python -m tetris_ai.train --resume --episodes N;")


def run(args) -> None:
    if not 0.0 <= args.recovery_start_rate <= 1.0:
        raise ValueError("--recovery-start-rate must be between 0.0 and 1.0")
    if args.best_model_objective == "recovery" and args.recovery_eval_seeds <= 0:
        raise ValueError("--best-model-objective recovery requires --recovery-eval-seeds > 0")

    torch, _ = require_torch()
    random.seed(args.seed)
    device = best_device(torch)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print_training_legend(args)

    model = make_value_net().to(device)
    target_model = make_value_net().to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, amsgrad=True)
    replay = PrioritizedReplay(args.replay_size)
    metrics_path = output_dir / "metrics.jsonl"
    best_model_path = output_dir / "best-model.json"
    best_replay_path = output_dir / "best-replay.json"
    checkpoint_path = resolve_checkpoint_path(args)
    best_median = -1.0
    best_top_out = 1.0
    best_score = -1.0
    recent_losses = deque(maxlen=100)
    global_step = 0
    start_episode = 0

    if args.resume:
        checkpoint = load_checkpoint(torch, checkpoint_path)
        if checkpoint is None:
            print(f"No checkpoint found at {checkpoint_path}; starting a fresh run.")
        else:
            model.load_state_dict(checkpoint["model"])
            target_model.load_state_dict(checkpoint["targetModel"])
            if args.reset_optimizer_on_resume:
                print("Reset optimizer state for resumed training.")
            else:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if args.reset_replay_on_resume:
                print("Reset replay buffer for resumed training.")
            else:
                replay.load_state_dict(checkpoint["replay"])
            best_median = float(checkpoint["bestMedian"])
            best_top_out = float(checkpoint["bestTopOut"])
            best_score = float(checkpoint.get("bestScore", -1.0))
            if args.reset_replay_on_resume or args.reset_optimizer_on_resume:
                recent_losses = deque(maxlen=100)
            else:
                recent_losses = deque(checkpoint.get("recentLosses", []), maxlen=100)
            global_step = int(checkpoint["globalStep"])
            start_episode = int(checkpoint["episodeIndex"]) + 1
            random.setstate(checkpoint["randomState"])
            if checkpoint.get("args", {}).get("best_model_objective") != args.best_model_objective:
                best_median = -1.0
                best_top_out = 1.0
                best_score = -1.0
                print(f"Reset best model tracking for {args.best_model_objective} objective.")
            print(
                f"Resumed checkpoint from episode {start_episode} "
                f"at step {global_step}."
            )

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(str(output_dir / "tensorboard"))
    except ModuleNotFoundError:
        writer = None

    start = time.perf_counter()
    start_step = global_step
    final_episode_index = start_episode - 1
    target_episode = start_episode + args.episodes
    for episode_index in range(start_episode, target_episode):
        final_episode_index = episode_index
        start_mode = choose_training_start_mode(args)
        game_seed = f"train-{args.seed}-{episode_index}"
        if start_mode == "recovery":
            game_seed = f"recovery-{game_seed}"
        game = create_start_game(game_seed, start_mode, args.recovery_severity)
        episode_reward = 0.0
        pieces = 0
        epsilon = args.eps_end + (args.eps_start - args.eps_end) * math.exp(-global_step / args.eps_decay)

        while not game.game_over and pieces < args.max_pieces:
            placement = choose_placement(model, torch, device, game, epsilon, args.reward_profile)
            if placement is None:
                break
            next_game = apply_actions(game, placement.actions)
            replay.add(
                Transition(
                    placement.vector,
                    placement.reward,
                    next_vectors(next_game, args.reward_profile),
                    placement.done,
                )
            )
            game = next_game
            episode_reward += placement.reward
            pieces += 1
            global_step += 1

            loss = optimize(model, target_model, optimizer, replay, torch, device, args)
            if loss is not None:
                recent_losses.append(loss)

            if global_step % args.target_update == 0:
                target_model.load_state_dict(model.state_dict())

        elapsed_seconds = time.perf_counter() - start
        throughput = performance_metrics(
            global_step - start_step,
            episode_index - start_episode + 1,
            elapsed_seconds,
        )
        metrics = {
            "type": "trainEpisode",
            "episode": episode_index,
            "step": global_step,
            "epsilon": epsilon,
            "reward": episode_reward,
            "pieces": pieces,
            "survivalSeconds": pieces * GRAVITY_SECONDS,
            "linesCleared": game.lines_cleared,
            "gameOver": game.game_over,
            "loss": mean(recent_losses) if recent_losses else None,
            "device": str(device),
            "rewardProfile": args.reward_profile,
            "startMode": start_mode,
            "recoverySeverity": args.recovery_severity if start_mode == "recovery" else None,
            **throughput,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
        if writer:
            writer.add_scalar("train/reward", episode_reward, episode_index)
            writer.add_scalar("train/pieces", pieces, episode_index)
            writer.add_scalar("train/epsilon", epsilon, episode_index)
            writer.add_scalar("train/stepsPerSecond", throughput["stepsPerSecond"], episode_index)
            writer.add_scalar("train/stepsPerHour", throughput["stepsPerHour"], episode_index)
            writer.add_scalar("train/episodesPerHour", throughput["episodesPerHour"], episode_index)
            if recent_losses:
                writer.add_scalar("train/loss", mean(recent_losses), episode_index)

        if (episode_index + 1) % args.eval_interval == 0:
            eval_result = evaluate(
                model,
                torch,
                device,
                HELD_OUT_SEEDS[: args.eval_seeds],
                args.milestone_seconds,
                capture_replay=True,
                eval_workers=args.eval_workers,
            )
            recovery_eval_result = None
            if args.recovery_eval_seeds > 0:
                recovery_eval_result = evaluate(
                    model,
                    torch,
                    device,
                    [f"recovery-heldout-{index}" for index in range(args.recovery_eval_seeds)],
                    args.recovery_eval_seconds,
                    capture_replay=False,
                    eval_workers=args.eval_workers,
                    start_mode="recovery",
                    recovery_severity=args.recovery_severity,
                )
            eval_metrics = {
                "type": "eval",
                "episode": episode_index,
                "step": global_step,
                **throughput,
                **{key: value for key, value in eval_result.items() if key not in ("episodes", "replay")},
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(eval_metrics) + "\n")
                if recovery_eval_result:
                    recovery_eval_metrics = {
                        "type": "recoveryEval",
                        "episode": episode_index,
                        "step": global_step,
                        **throughput,
                        **{
                            key: value
                            for key, value in recovery_eval_result.items()
                            if key not in ("episodes", "replay")
                        },
                    }
                    handle.write(json.dumps(recovery_eval_metrics) + "\n")
            if writer:
                writer.add_scalar("eval/successRate", eval_result["successRate"], episode_index)
                writer.add_scalar("eval/medianSurvivalSeconds", eval_result["medianSurvivalSeconds"], episode_index)
                writer.add_scalar("eval/topOutRate", eval_result["topOutRate"], episode_index)
                writer.add_scalar("eval/meanScore", eval_result["meanScore"], episode_index)
                writer.add_scalar("eval/medianScore", eval_result["medianScore"], episode_index)
                writer.add_scalar("eval/maxScore", eval_result["maxScore"], episode_index)
                writer.add_scalar("eval/meanLinesCleared", eval_result["meanLinesCleared"], episode_index)
                writer.add_scalar("eval/stepsPerSecond", throughput["stepsPerSecond"], episode_index)
                writer.add_scalar("eval/stepsPerHour", throughput["stepsPerHour"], episode_index)
                if recovery_eval_result:
                    writer.add_scalar("recovery_eval/successRate", recovery_eval_result["successRate"], episode_index)
                    writer.add_scalar(
                        "recovery_eval/medianSurvivalSeconds",
                        recovery_eval_result["medianSurvivalSeconds"],
                        episode_index,
                    )
                    writer.add_scalar("recovery_eval/topOutRate", recovery_eval_result["topOutRate"], episode_index)

            if args.best_model_objective == "score":
                meets_floor = eval_result["successRate"] >= args.score_survival_floor
                improved = meets_floor and eval_result["meanScore"] > best_score
            elif args.best_model_objective == "recovery":
                meets_floor = eval_result["successRate"] >= args.recovery_clean_floor
                improved = bool(
                    recovery_eval_result
                    and meets_floor
                    and recovery_eval_result["topOutRate"] < best_top_out
                )
            else:
                improved = (
                    eval_result["medianSurvivalSeconds"] >= best_median + 10.0
                    or eval_result["topOutRate"] < best_top_out
                )
            if improved:
                best_median = eval_result["medianSurvivalSeconds"]
                best_top_out = eval_result["topOutRate"]
                best_score = eval_result["meanScore"]
                if args.best_model_objective == "recovery" and recovery_eval_result:
                    best_median = recovery_eval_result["medianSurvivalSeconds"]
                    best_top_out = recovery_eval_result["topOutRate"]
                    best_score = recovery_eval_result["meanScore"]
                export_model(model, torch, best_model_path, model_export_metadata(episode_index))
                if eval_result["replay"]:
                    best_replay_path.write_text(json.dumps(eval_result["replay"]), encoding="utf-8")

            save_checkpoint(
                torch,
                checkpoint_path,
                checkpoint_payload(
                    model,
                    target_model,
                    optimizer,
                    replay,
                    episode_index,
                    global_step,
                    best_median,
                    best_top_out,
                    best_score,
                    recent_losses,
                    args,
                ),
            )

            print(
                f"episode={episode_index + 1} step={global_step} "
                f"steps_per_hour={throughput['stepsPerHour']:.0f} "
                f"eval_success={eval_result['successRate']:.3f} "
                f"median={eval_result['medianSurvivalSeconds']:.1f}s "
                f"mean_score={eval_result['meanScore']:.1f} "
                f"median_score={eval_result['medianScore']:.1f} "
                f"max_score={eval_result['maxScore']}"
            )
            if recovery_eval_result:
                print(
                    f"recovery_eval_success={recovery_eval_result['successRate']:.3f} "
                    f"recovery_median={recovery_eval_result['medianSurvivalSeconds']:.1f}s "
                    f"recovery_mean_score={recovery_eval_result['meanScore']:.1f}"
                )

    export_model(model, torch, output_dir / "latest-model.json", model_export_metadata(final_episode_index))
    save_checkpoint(
        torch,
        checkpoint_path,
        checkpoint_payload(
            model,
            target_model,
            optimizer,
            replay,
            final_episode_index,
            global_step,
            best_median,
            best_top_out,
            best_score,
            recent_losses,
            args,
        ),
    )
    if writer:
        writer.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a local afterstate DQN Tetris agent.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--max-pieces", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-size", type=int, default=50000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=float, default=12000)
    parser.add_argument("--priority-beta", type=float, default=0.4)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--target-update", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=25)
    parser.add_argument("--eval-seeds", type=int, default=200)
    parser.add_argument("--eval-workers", type=int, default=0, help="Evaluation worker processes. Use 0 for auto.")
    parser.add_argument("--milestone-seconds", type=float, default=60.0)
    parser.add_argument("--reward-profile", choices=REWARD_PROFILES, default="survival")
    parser.add_argument("--reset-replay-on-resume", action="store_true")
    parser.add_argument("--reset-optimizer-on-resume", action="store_true")
    parser.add_argument("--best-model-objective", choices=("survival", "score", "recovery"), default="survival")
    parser.add_argument("--score-survival-floor", type=float, default=0.70)
    parser.add_argument("--recovery-start-rate", type=float, default=0.0)
    parser.add_argument("--recovery-severity", choices=RECOVERY_SEVERITIES, default="medium")
    parser.add_argument("--recovery-eval-seeds", type=int, default=0)
    parser.add_argument("--recovery-eval-seconds", type=float, default=300.0)
    parser.add_argument("--recovery-clean-floor", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="runs/tetris-agent")
    parser.add_argument(
        "--checkpoint-dir",
        default=str(DEFAULT_CHECKPOINT_DIR),
        help=f"Directory for compressed checkpoint data. Writes {CHECKPOINT_FILENAME} here.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"Continue from checkpoint-dir/{CHECKPOINT_FILENAME} when present.",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
