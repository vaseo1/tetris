from __future__ import annotations

import argparse
import json
import sys
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tetris_ai.afterstates import apply_actions, enumerate_placements
from tetris_ai.engine import create_game


def benchmark_enumeration(seed_prefix: str, runs: int) -> dict[str, float | int]:
    started = time.perf_counter()
    placements_total = 0
    counts: list[int] = []
    for index in range(runs):
        placements = enumerate_placements(create_game(f"{seed_prefix}-enum-{index}"))
        count = len(placements)
        placements_total += count
        counts.append(count)
    elapsed = time.perf_counter() - started
    return {
        "runs": runs,
        "seconds": elapsed,
        "iterationsPerSecond": runs / elapsed if elapsed else 0.0,
        "placementsPerSecond": placements_total / elapsed if elapsed else 0.0,
        "meanPlacements": statistics.mean(counts) if counts else 0.0,
        "medianPlacements": statistics.median(counts) if counts else 0.0,
    }


def benchmark_rollout(seed_prefix: str, episodes: int, max_pieces: int) -> dict[str, float | int]:
    started = time.perf_counter()
    pieces_total = 0
    placements_total = 0
    for episode_index in range(episodes):
        game = create_game(f"{seed_prefix}-rollout-{episode_index}")
        episode_pieces = 0
        while not game.game_over and episode_pieces < max_pieces:
            placements = enumerate_placements(game)
            placements_total += len(placements)
            if not placements:
                break
            game = apply_actions(game, placements[0].actions)
            pieces_total += 1
            episode_pieces += 1
    elapsed = time.perf_counter() - started
    return {
        "episodes": episodes,
        "maxPiecesPerEpisode": max_pieces,
        "seconds": elapsed,
        "pieces": pieces_total,
        "piecesPerSecond": pieces_total / elapsed if elapsed else 0.0,
        "placementsPerSecond": placements_total / elapsed if elapsed else 0.0,
    }


def summarize_trials(results: list[dict[str, float | int]]) -> dict[str, object]:
    if not results:
        return {"trials": []}

    first = results[0]
    summary: dict[str, object] = {"trials": results}
    for key, value in first.items():
        if isinstance(value, (int, float)):
            numeric_values = [float(result[key]) for result in results]
            summary[key] = statistics.median(numeric_values)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Python-side Tetris training logic.")
    parser.add_argument("--enum-runs", type=int, default=200)
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--max-pieces", type=int, default=80)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--seed-prefix", default="bench")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = {
        "enumeration": summarize_trials(
            [
                benchmark_enumeration(f"{args.seed_prefix}-trial-{trial_index}", args.enum_runs)
                for trial_index in range(args.repeat)
            ]
        ),
        "rollout": summarize_trials(
            [
                benchmark_rollout(f"{args.seed_prefix}-trial-{trial_index}", args.episodes, args.max_pieces)
                for trial_index in range(args.repeat)
            ]
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
