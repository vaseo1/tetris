from __future__ import annotations

from dataclasses import dataclass
from typing import Any

COLS = 10
ROWS = 20

ACTIONS = {
    "left": "left",
    "right": "right",
    "down": "down",
    "rotate": "rotate",
    "hardDrop": "hardDrop",
    "pause": "pause",
    "restart": "restart",
    "noop": "noop",
}

SHAPES = [
    {"name": "I", "cells": ((0, 1), (1, 1), (2, 1), (3, 1))},
    {"name": "J", "cells": ((0, 0), (0, 1), (1, 1), (2, 1))},
    {"name": "L", "cells": ((2, 0), (0, 1), (1, 1), (2, 1))},
    {"name": "O", "cells": ((1, 0), (2, 0), (1, 1), (2, 1))},
    {"name": "S", "cells": ((1, 0), (2, 0), (0, 1), (1, 1))},
    {"name": "T", "cells": ((1, 0), (0, 1), (1, 1), (2, 1))},
    {"name": "Z", "cells": ((0, 0), (1, 0), (1, 1), (2, 1))},
]


@dataclass
class Piece:
    name: str
    cells: list[list[int]]
    x: int
    y: int


@dataclass
class Game:
    cols: int
    rows: int
    board: list[list[int]]
    active_piece: Piece
    next_piece: Piece
    score: int
    lines_cleared: int
    game_over: bool
    paused: bool
    status: str
    seed: int
    rng_state: int
    gravity_ticks: int
    last_cleared: int = 0


def normalize_seed(seed: Any | None = None) -> int:
    if seed is None:
        import random

        return random.randrange(0x100000000)

    if isinstance(seed, (int, float)) and seed == seed and seed not in (float("inf"), float("-inf")):
        return int(seed) & 0xFFFFFFFF

    text = str(seed)
    hash_value = 2166136261
    for char in text:
        hash_value ^= ord(char)
        hash_value = (hash_value * 16777619) & 0xFFFFFFFF
    return hash_value


def next_random(game: Game) -> float:
    game.rng_state = ((game.rng_state * 1664525) + 1013904223) & 0xFFFFFFFF
    return game.rng_state / 0x100000000


def make_board() -> list[list[int]]:
    return [[0 for _ in range(COLS)] for _ in range(ROWS)]


def clone_piece(piece: Piece) -> Piece:
    return Piece(piece.name, [cell[:] for cell in piece.cells], piece.x, piece.y)


def clone_game(game: Game) -> Game:
    return Game(
        cols=game.cols,
        rows=game.rows,
        board=[row[:] for row in game.board],
        active_piece=clone_piece(game.active_piece),
        next_piece=clone_piece(game.next_piece),
        score=game.score,
        lines_cleared=game.lines_cleared,
        game_over=game.game_over,
        paused=game.paused,
        status=game.status,
        seed=game.seed,
        rng_state=game.rng_state,
        gravity_ticks=game.gravity_ticks,
        last_cleared=game.last_cleared,
    )


def create_piece(game: Game) -> Piece:
    shape = SHAPES[int(next_random(game) * len(SHAPES))]
    return Piece(
        name=shape["name"],
        cells=[[x, y] for x, y in shape["cells"]],
        x=(COLS // 2) - 2,
        y=0,
    )


def create_game(seed: Any | None = None) -> Game:
    normalized_seed = normalize_seed(seed)
    game = Game(
        cols=COLS,
        rows=ROWS,
        board=make_board(),
        active_piece=Piece("", [], 0, 0),
        next_piece=Piece("", [], 0, 0),
        score=0,
        lines_cleared=0,
        game_over=False,
        paused=False,
        status="READY",
        seed=normalized_seed,
        rng_state=normalized_seed,
        gravity_ticks=0,
    )
    game.active_piece = create_piece(game)
    game.next_piece = create_piece(game)
    return game


def can_play(game: Game) -> bool:
    return not game.game_over and not game.paused


def collides(
    game: Game,
    piece: Piece,
    offset_x: int = 0,
    offset_y: int = 0,
    cells: list[list[int]] | None = None,
) -> bool:
    check_cells = piece.cells if cells is None else cells
    for cell_x, cell_y in check_cells:
        x = piece.x + cell_x + offset_x
        y = piece.y + cell_y + offset_y
        if x < 0 or x >= COLS or y >= ROWS:
            return True
        if y >= 0 and game.board[y][x]:
            return True
    return False


def set_status(game: Game, status: str) -> None:
    game.status = status


def clear_rows(game: Game) -> int:
    cleared = 0
    kept_rows = []
    for row in game.board:
        if all(row):
            cleared += 1
        else:
            kept_rows.append(row)

    while len(kept_rows) < ROWS:
        kept_rows.insert(0, [0 for _ in range(COLS)])

    game.board = kept_rows
    game.last_cleared = cleared
    if cleared:
        game.score += cleared
        game.lines_cleared += cleared
    return cleared


def move(game: Game, dx: int, dy: int) -> bool:
    if not can_play(game) or collides(game, game.active_piece, dx, dy):
        return False
    game.active_piece.x += dx
    game.active_piece.y += dy
    set_status(game, "PLAY")
    return True


def rotate(game: Game) -> bool:
    if not can_play(game) or game.active_piece.name == "O":
        return False

    rotated = [[3 - y, x] for x, y in game.active_piece.cells]
    for kick in (0, -1, 1, -2, 2):
        if not collides(game, game.active_piece, kick, 0, rotated):
            game.active_piece.cells = rotated
            game.active_piece.x += kick
            set_status(game, "PLAY")
            return True
    return False


def lock_piece(game: Game) -> None:
    for cell_x, cell_y in game.active_piece.cells:
        x = game.active_piece.x + cell_x
        y = game.active_piece.y + cell_y
        if 0 <= y < ROWS and 0 <= x < COLS:
            game.board[y][x] = 1

    clear_rows(game)
    game.active_piece = game.next_piece
    game.next_piece = create_piece(game)

    if collides(game, game.active_piece):
        game.game_over = True
        set_status(game, "GAME OVER")
    else:
        set_status(game, "PLAY")


def soft_drop(game: Game) -> bool:
    if not can_play(game):
        return False
    if move(game, 0, 1):
        return True
    lock_piece(game)
    return True


def drop_distance(game: Game, piece: Piece) -> int:
    bottom_cells: dict[int, int] = {}
    for cell_x, cell_y in piece.cells:
        board_x = piece.x + cell_x
        board_y = piece.y + cell_y
        previous = bottom_cells.get(board_x)
        if previous is None or board_y > previous:
            bottom_cells[board_x] = board_y

    max_drop: int | None = None
    for board_x, board_y in bottom_cells.items():
        drop = 0
        next_y = board_y + 1
        while next_y < ROWS and not game.board[next_y][board_x]:
            drop += 1
            next_y += 1
        if max_drop is None or drop < max_drop:
            max_drop = drop
    return 0 if max_drop is None else max_drop


def hard_drop(game: Game) -> bool:
    if not can_play(game):
        return False
    game.active_piece.y += drop_distance(game, game.active_piece)
    lock_piece(game)
    return True


def step_game(
    game: Game,
    action: str = ACTIONS["noop"],
    capture_state: bool = True,
) -> dict[str, Any] | None:
    game.last_cleared = 0
    if action == ACTIONS["left"]:
        move(game, -1, 0)
    elif action == ACTIONS["right"]:
        move(game, 1, 0)
    elif action == ACTIONS["down"]:
        soft_drop(game)
    elif action == ACTIONS["rotate"]:
        rotate(game)
    elif action == ACTIONS["hardDrop"]:
        hard_drop(game)
    elif action == ACTIONS["pause"]:
        if not game.game_over:
            game.paused = not game.paused
            set_status(game, "PAUSED" if game.paused else "PLAY")
    elif action == ACTIONS["restart"]:
        restarted = create_game(game.seed)
        game.__dict__.update(restarted.__dict__)
    elif action == ACTIONS["noop"]:
        if can_play(game):
            set_status(game, "PLAY")
    else:
        raise ValueError(f"Unknown Tetris action: {action}")
    if capture_state:
        return get_state(game)
    return None


def advance_gravity(game: Game, ticks: int = 1, capture_state: bool = True) -> dict[str, Any] | None:
    count = max(0, int(ticks))
    game.last_cleared = 0
    for _ in range(count):
        if not can_play(game):
            break
        game.gravity_ticks += 1
        soft_drop(game)
    if capture_state:
        return get_state(game)
    return None


def get_state(game: Game) -> dict[str, Any]:
    return {
        "cols": game.cols,
        "rows": game.rows,
        "board": [row[:] for row in game.board],
        "activePiece": piece_to_state(game.active_piece),
        "nextPiece": piece_to_state(game.next_piece),
        "score": game.score,
        "linesCleared": game.lines_cleared,
        "gameOver": game.game_over,
        "paused": game.paused,
        "status": game.status,
        "seed": game.seed,
        "gravityTicks": game.gravity_ticks,
    }


def piece_to_state(piece: Piece) -> dict[str, Any]:
    return {
        "name": piece.name,
        "cells": [cell[:] for cell in piece.cells],
        "x": piece.x,
        "y": piece.y,
    }
