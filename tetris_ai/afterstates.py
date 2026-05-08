from __future__ import annotations

from dataclasses import dataclass

from .engine import ACTIONS, Game, clone_game, collides, hard_drop, move, rotate, step_game
from .features import board_metrics, feature_vector

REWARD_PROFILES = ("survival", "phase2-score")


@dataclass(frozen=True)
class Placement:
    actions: tuple[str, ...]
    rotation_count: int
    target_x: int
    board: tuple[tuple[int, ...], ...]
    cleared: int
    reward: float
    done: bool
    next_piece: str
    vector: list[float]


def reward_from_metrics(metrics: dict[str, int], cleared: int, done: bool, reward_profile: str = "survival") -> float:
    if reward_profile == "phase2-score":
        line_reward = [0.0, 1.5, 4.0, 7.0, 12.0][min(4, cleared)]
        return (
            0.04
            + line_reward
            - 0.04 * metrics["holes"]
            - 0.02 * metrics["maxHeight"]
            - 0.012 * metrics["bumpiness"]
            - (8.0 if done else 0.0)
        )
    if reward_profile != "survival":
        raise ValueError(f"Unknown reward profile: {reward_profile}")

    line_reward = [0.0, 1.0, 3.0, 5.0, 8.0][min(4, cleared)]
    return (
        0.08
        + line_reward
        - 0.035 * metrics["holes"]
        - 0.018 * metrics["maxHeight"]
        - 0.01 * metrics["bumpiness"]
        - (8.0 if done else 0.0)
    )


def reward_for(board: list[list[int]], cleared: int, done: bool, reward_profile: str = "survival") -> float:
    return reward_from_metrics(board_metrics(board), cleared, done, reward_profile)


def apply_actions(game: Game, actions: tuple[str, ...]) -> Game:
    clone = clone_game(game)
    for action in actions:
        clone.last_cleared = 0
        if action == ACTIONS["left"]:
            move(clone, -1, 0)
        elif action == ACTIONS["right"]:
            move(clone, 1, 0)
        elif action == ACTIONS["rotate"]:
            rotate(clone)
        elif action == ACTIONS["hardDrop"]:
            hard_drop(clone)
        else:
            step_game(clone, action, capture_state=False)
    return clone


def enumerate_placements(game: Game, reward_profile: str = "survival") -> list[Placement]:
    if game.game_over or game.paused:
        return []

    placements: list[Placement] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()

    for rotation_count in range(4):
        rotated_game = clone_game(game)
        rotation_actions: list[str] = []
        for _ in range(rotation_count):
            if rotate(rotated_game):
                rotation_actions.append(ACTIONS["rotate"])

        for target_x in range(-4, game.cols + 4):
            candidate = clone_game(rotated_game)
            horizontal_actions: list[str] = []

            while candidate.active_piece.x > target_x:
                before = candidate.active_piece.x
                move(candidate, -1, 0)
                if candidate.active_piece.x == before:
                    break
                horizontal_actions.append(ACTIONS["left"])

            while candidate.active_piece.x < target_x:
                before = candidate.active_piece.x
                move(candidate, 1, 0)
                if candidate.active_piece.x == before:
                    break
                horizontal_actions.append(ACTIONS["right"])

            if candidate.active_piece.x != target_x or collides(candidate, candidate.active_piece):
                continue

            hard_drop(candidate)
            board_key = tuple(tuple(row) for row in candidate.board)
            if board_key in seen:
                continue
            seen.add(board_key)

            actions = tuple(rotation_actions + horizontal_actions + [ACTIONS["hardDrop"]])
            cleared = candidate.last_cleared
            metrics = board_metrics(candidate.board)
            placements.append(
                Placement(
                    actions=actions,
                    rotation_count=rotation_count,
                    target_x=target_x,
                    board=board_key,
                    cleared=cleared,
                    reward=reward_from_metrics(metrics, cleared, candidate.game_over, reward_profile),
                    done=candidate.game_over,
                    next_piece=candidate.active_piece.name,
                    vector=feature_vector(candidate.board, candidate.active_piece.name, cleared, metrics=metrics),
                )
            )

    placements.sort(key=lambda placement: (placement.rotation_count, placement.target_x, placement.actions))
    return placements
