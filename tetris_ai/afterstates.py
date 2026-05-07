from __future__ import annotations

from dataclasses import dataclass

from .engine import ACTIONS, Game, clone_game, collides, rotate, step_game
from .features import board_metrics, feature_vector


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

    @property
    def vector(self) -> list[float]:
        return feature_vector([list(row) for row in self.board], self.next_piece, self.cleared)


def reward_for(board: list[list[int]], cleared: int, done: bool) -> float:
    metrics = board_metrics(board)
    line_reward = [0.0, 1.0, 3.0, 5.0, 8.0][min(4, cleared)]
    return (
        0.08
        + line_reward
        - 0.035 * metrics["holes"]
        - 0.018 * metrics["maxHeight"]
        - 0.01 * metrics["bumpiness"]
        - (8.0 if done else 0.0)
    )


def apply_actions(game: Game, actions: tuple[str, ...]) -> Game:
    clone = clone_game(game)
    for action in actions:
        step_game(clone, action)
    return clone


def enumerate_placements(game: Game) -> list[Placement]:
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
                step_game(candidate, ACTIONS["left"])
                if candidate.active_piece.x == before:
                    break
                horizontal_actions.append(ACTIONS["left"])

            while candidate.active_piece.x < target_x:
                before = candidate.active_piece.x
                step_game(candidate, ACTIONS["right"])
                if candidate.active_piece.x == before:
                    break
                horizontal_actions.append(ACTIONS["right"])

            if candidate.active_piece.x != target_x or collides(candidate, candidate.active_piece):
                continue

            step_game(candidate, ACTIONS["hardDrop"])
            board_key = tuple(tuple(row) for row in candidate.board)
            if board_key in seen:
                continue
            seen.add(board_key)

            actions = tuple(rotation_actions + horizontal_actions + [ACTIONS["hardDrop"]])
            cleared = candidate.last_cleared
            placements.append(
                Placement(
                    actions=actions,
                    rotation_count=rotation_count,
                    target_x=target_x,
                    board=board_key,
                    cleared=cleared,
                    reward=reward_for(candidate.board, cleared, candidate.game_over),
                    done=candidate.game_over,
                    next_piece=candidate.active_piece.name,
                )
            )

    placements.sort(key=lambda placement: (placement.rotation_count, placement.target_x, placement.actions))
    return placements
