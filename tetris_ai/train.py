from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from .afterstates import Placement, apply_actions, enumerate_placements
from .engine import Game, create_game, get_state
from .features import FEATURE_SIZE, board_metrics
from .model import best_device, make_value_net, require_torch

GRAVITY_SECONDS = 0.7
HELD_OUT_SEEDS = [f"heldout-{index}" for index in range(200)]


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


def choose_placement(model, torch, device, game: Game, epsilon: float) -> Placement | None:
    placements = enumerate_placements(game)
    if not placements:
        return None
    if random.random() < epsilon:
        return random.choice(placements)
    with torch.no_grad():
        batch = torch.tensor([placement.vector for placement in placements], dtype=torch.float32, device=device)
        values = model(batch)
        return placements[int(torch.argmax(values).item())]


def next_vectors(game: Game) -> list[list[float]]:
    return [placement.vector for placement in enumerate_placements(game)]


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


def export_model(model, torch, path: Path) -> None:
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
        "recentLosses": list(recent_losses),
        "randomState": random.getstate(),
        "args": vars(args),
    }


def save_checkpoint(torch, checkpoint_path: Path, payload: dict[str, Any]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_suffix(".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, checkpoint_path)


def load_checkpoint(torch, checkpoint_path: Path) -> dict[str, Any] | None:
    if not checkpoint_path.exists():
        return None
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def evaluate(model, torch, device, seeds: list[Any], max_seconds: float, capture_replay: bool = False):
    episodes = []
    best_replay = None
    max_pieces = max(1, math.ceil(max_seconds / GRAVITY_SECONDS))

    for seed in seeds:
        game = create_game(seed)
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
        }
        episodes.append(result)
        if capture_replay and best_replay is None:
            best_replay = {
                "seed": str(seed),
                "gravitySeconds": GRAVITY_SECONDS,
                "frames": frames,
                "finalState": get_state(game),
                "result": result,
            }

    success_rate = mean(1.0 if episode["survivalSeconds"] >= max_seconds else 0.0 for episode in episodes)
    scores = [episode["score"] for episode in episodes]
    lines = [episode["linesCleared"] for episode in episodes]
    return {
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
    print("Training output guide:")
    print(f"  eval_success = fraction of held-out seeds that survive {args.milestone_seconds:.0f}s;")
    print("  median = median survival time;")
    print("  mean_score/median_score/max_score = held-out evaluation scores;")
    print("Resume hint:")
    print(f"  uv run npm run train:ai -- --resume --episodes {args.episodes};")
    print("  uv run python -m tetris_ai.train --resume --episodes N;")


def run(args) -> None:
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
    checkpoint_path = output_dir / "checkpoint.pt"
    best_median = -1.0
    best_top_out = 1.0
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
            optimizer.load_state_dict(checkpoint["optimizer"])
            replay.load_state_dict(checkpoint["replay"])
            best_median = float(checkpoint["bestMedian"])
            best_top_out = float(checkpoint["bestTopOut"])
            recent_losses = deque(checkpoint.get("recentLosses", []), maxlen=100)
            global_step = int(checkpoint["globalStep"])
            start_episode = int(checkpoint["episodeIndex"]) + 1
            random.setstate(checkpoint["randomState"])
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

    start = time.time()
    final_episode_index = start_episode - 1
    target_episode = start_episode + args.episodes
    for episode_index in range(start_episode, target_episode):
        final_episode_index = episode_index
        game = create_game(f"train-{args.seed}-{episode_index}")
        episode_reward = 0.0
        pieces = 0
        epsilon = args.eps_end + (args.eps_start - args.eps_end) * math.exp(-global_step / args.eps_decay)

        while not game.game_over and pieces < args.max_pieces:
            placement = choose_placement(model, torch, device, game, epsilon)
            if placement is None:
                break
            next_game = apply_actions(game, placement.actions)
            replay.add(Transition(placement.vector, placement.reward, next_vectors(next_game), placement.done))
            game = next_game
            episode_reward += placement.reward
            pieces += 1
            global_step += 1

            loss = optimize(model, target_model, optimizer, replay, torch, device, args)
            if loss is not None:
                recent_losses.append(loss)

            if global_step % args.target_update == 0:
                target_model.load_state_dict(model.state_dict())

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
            "elapsedSeconds": time.time() - start,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
        if writer:
            writer.add_scalar("train/reward", episode_reward, episode_index)
            writer.add_scalar("train/pieces", pieces, episode_index)
            writer.add_scalar("train/epsilon", epsilon, episode_index)
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
            )
            eval_metrics = {
                "type": "eval",
                "episode": episode_index,
                "step": global_step,
                **{key: value for key, value in eval_result.items() if key not in ("episodes", "replay")},
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(eval_metrics) + "\n")
            if writer:
                writer.add_scalar("eval/successRate", eval_result["successRate"], episode_index)
                writer.add_scalar("eval/medianSurvivalSeconds", eval_result["medianSurvivalSeconds"], episode_index)
                writer.add_scalar("eval/topOutRate", eval_result["topOutRate"], episode_index)
                writer.add_scalar("eval/meanScore", eval_result["meanScore"], episode_index)
                writer.add_scalar("eval/medianScore", eval_result["medianScore"], episode_index)
                writer.add_scalar("eval/maxScore", eval_result["maxScore"], episode_index)
                writer.add_scalar("eval/meanLinesCleared", eval_result["meanLinesCleared"], episode_index)

            improved = (
                eval_result["medianSurvivalSeconds"] >= best_median + 10.0
                or eval_result["topOutRate"] < best_top_out
            )
            if improved:
                best_median = eval_result["medianSurvivalSeconds"]
                best_top_out = eval_result["topOutRate"]
                export_model(model, torch, best_model_path)
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
                    recent_losses,
                    args,
                ),
            )

            print(
                f"episode={episode_index + 1} step={global_step} "
                f"eval_success={eval_result['successRate']:.3f} "
                f"median={eval_result['medianSurvivalSeconds']:.1f}s "
                f"mean_score={eval_result['meanScore']:.1f} "
                f"median_score={eval_result['medianScore']:.1f} "
                f"max_score={eval_result['maxScore']}"
            )

    export_model(model, torch, output_dir / "latest-model.json")
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
    parser.add_argument("--milestone-seconds", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="runs/tetris-agent")
    parser.add_argument("--resume", action="store_true", help="Continue from output-dir/checkpoint.pt when present.")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
