from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .engine import COLS, ROWS, Game, collides, create_game, normalize_seed
from .features import board_metrics

RECOVERY_SEVERITIES = ("easy", "medium", "hard", "mixed")


@dataclass(frozen=True)
class RecoveryConfig:
    min_height: int
    max_height: int
    height_jitter: int
    holes: int
    well_depth: int


_CONFIGS = {
    "easy": RecoveryConfig(3, 8, 2, 4, 3),
    "medium": RecoveryConfig(5, 12, 3, 9, 5),
    "hard": RecoveryConfig(8, 16, 4, 16, 7),
}


def resolve_recovery_severity(seed: Any, severity: str) -> str:
    if severity != "mixed":
        if severity not in _CONFIGS:
            raise ValueError(f"Unknown recovery severity: {severity}")
        return severity

    rng = random.Random(normalize_seed(f"recovery-severity-{seed}"))
    return rng.choices(("easy", "medium", "hard"), weights=(2, 5, 3), k=1)[0]


def make_recovery_board(seed: Any, severity: str = "medium") -> list[list[int]]:
    resolved = resolve_recovery_severity(seed, severity)
    config = _CONFIGS[resolved]
    rng = random.Random(normalize_seed(f"recovery-board-{resolved}-{seed}"))

    base_height = rng.randint(config.min_height, config.max_height)
    heights: list[int] = []
    previous = base_height
    for _ in range(COLS):
        previous += rng.randint(-config.height_jitter, config.height_jitter)
        heights.append(max(config.min_height, min(config.max_height, previous)))

    well_column = rng.randrange(COLS)
    heights[well_column] = max(1, heights[well_column] - rng.randint(2, config.well_depth))
    if rng.random() < 0.65:
        spike_column = rng.randrange(COLS)
        heights[spike_column] = min(config.max_height, heights[spike_column] + rng.randint(2, config.height_jitter + 3))

    board = [[0 for _ in range(COLS)] for _ in range(ROWS)]
    for x, height in enumerate(heights):
        for y in range(ROWS - height, ROWS):
            board[y][x] = 1

    occupied = [
        (x, y)
        for x, height in enumerate(heights)
        for y in range(ROWS - height + 1, ROWS)
        if x != well_column
    ]
    rng.shuffle(occupied)
    for x, y in occupied[: config.holes]:
        board[y][x] = 0

    for y, row in enumerate(board):
        if all(row):
            board[y][rng.randrange(COLS)] = 0

    return board


def create_recovery_game(seed: Any | None = None, severity: str = "medium") -> Game:
    game = create_game(seed)
    game.board = make_recovery_board(seed, severity)
    if collides(game, game.active_piece):
        game.game_over = True
        game.status = "GAME OVER"
    else:
        game.status = "PLAY"
    return game


def recovery_summary(seed: Any, severity: str = "medium") -> dict[str, int | str]:
    resolved = resolve_recovery_severity(seed, severity)
    metrics = board_metrics(make_recovery_board(seed, resolved))
    return {"severity": resolved, **metrics}
