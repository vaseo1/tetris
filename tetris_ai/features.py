from __future__ import annotations

from .engine import COLS, ROWS

PIECES = ("I", "J", "L", "O", "S", "T", "Z")
FEATURE_SIZE = ROWS * COLS + len(PIECES) + 8


def column_heights(board: list[list[int]]) -> list[int]:
    heights = []
    for x in range(COLS):
        height = 0
        for y in range(ROWS):
            if board[y][x]:
                height = ROWS - y
                break
        heights.append(height)
    return heights


def count_holes(board: list[list[int]]) -> int:
    holes = 0
    for x in range(COLS):
        seen_block = False
        for y in range(ROWS):
            if board[y][x]:
                seen_block = True
            elif seen_block:
                holes += 1
    return holes


def bumpiness(heights: list[int]) -> int:
    return sum(abs(heights[index] - heights[index + 1]) for index in range(COLS - 1))


def aggregate_height(heights: list[int]) -> int:
    return sum(heights)


def complete_lines(board: list[list[int]]) -> int:
    return sum(1 for row in board if all(row))


def well_depth(board: list[list[int]]) -> int:
    heights = column_heights(board)
    depth = 0
    for index, height in enumerate(heights):
        left = heights[index - 1] if index > 0 else ROWS
        right = heights[index + 1] if index < COLS - 1 else ROWS
        rim = min(left, right)
        if rim > height:
            depth += rim - height
    return depth


def board_metrics(board: list[list[int]]) -> dict[str, int]:
    heights = column_heights(board)
    return {
        "holes": count_holes(board),
        "maxHeight": max(heights),
        "aggregateHeight": aggregate_height(heights),
        "bumpiness": bumpiness(heights),
        "completeLines": complete_lines(board),
        "wells": well_depth(board),
        "filledCells": sum(sum(row) for row in board),
    }


def feature_vector(
    board: list[list[int]],
    next_piece_name: str,
    cleared: int = 0,
    metrics: dict[str, int] | None = None,
) -> list[float]:
    flat = [float(cell) for row in board for cell in row]
    one_hot = [1.0 if piece == next_piece_name else 0.0 for piece in PIECES]
    resolved_metrics = board_metrics(board) if metrics is None else metrics
    extras = [
        cleared / 4.0,
        resolved_metrics["holes"] / 80.0,
        resolved_metrics["maxHeight"] / ROWS,
        resolved_metrics["aggregateHeight"] / (ROWS * COLS),
        resolved_metrics["bumpiness"] / (ROWS * COLS),
        resolved_metrics["completeLines"] / 4.0,
        resolved_metrics["wells"] / (ROWS * COLS),
        resolved_metrics["filledCells"] / (ROWS * COLS),
    ]
    return flat + one_hot + extras
