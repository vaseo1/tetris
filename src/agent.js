import { ACTIONS, COLS, ROWS } from './engine.js';

export const PIECES = ['I', 'J', 'L', 'O', 'S', 'T', 'Z'];
export const FEATURE_SIZE = ROWS * COLS + PIECES.length + 8;

function cloneBoard(board) {
  return board.map((row) => [...row]);
}

function clonePiece(piece) {
  return {
    name: piece.name,
    cells: piece.cells.map(([x, y]) => [x, y]),
    x: piece.x,
    y: piece.y,
  };
}

function collides(board, piece, offsetX = 0, offsetY = 0, cells = piece.cells) {
  return cells.some(([cellX, cellY]) => {
    const x = piece.x + cellX + offsetX;
    const y = piece.y + cellY + offsetY;
    return x < 0 || x >= COLS || y >= ROWS || (y >= 0 && board[y][x]);
  });
}

function rotatePiece(board, piece) {
  if (piece.name === 'O') return false;

  const rotated = piece.cells.map(([x, y]) => [3 - y, x]);
  const kicks = [0, -1, 1, -2, 2];

  for (const kick of kicks) {
    if (!collides(board, piece, kick, 0, rotated)) {
      piece.cells = rotated;
      piece.x += kick;
      return true;
    }
  }

  return false;
}

function lockPiece(board, piece) {
  const nextBoard = cloneBoard(board);
  piece.cells.forEach(([cellX, cellY]) => {
    const x = piece.x + cellX;
    const y = piece.y + cellY;
    if (y >= 0 && y < ROWS && x >= 0 && x < COLS) {
      nextBoard[y][x] = 1;
    }
  });

  let cleared = 0;
  const kept = nextBoard.filter((row) => {
    if (row.every(Boolean)) {
      cleared += 1;
      return false;
    }
    return true;
  });

  while (kept.length < ROWS) {
    kept.unshift(Array(COLS).fill(0));
  }

  return { board: kept, cleared };
}

export function columnHeights(board) {
  return Array.from({ length: COLS }, (_, x) => {
    for (let y = 0; y < ROWS; y += 1) {
      if (board[y][x]) return ROWS - y;
    }
    return 0;
  });
}

export function boardMetrics(board) {
  const heights = columnHeights(board);
  let holes = 0;
  for (let x = 0; x < COLS; x += 1) {
    let seenBlock = false;
    for (let y = 0; y < ROWS; y += 1) {
      if (board[y][x]) {
        seenBlock = true;
      } else if (seenBlock) {
        holes += 1;
      }
    }
  }

  const bumpiness = heights
    .slice(0, -1)
    .reduce((total, height, index) => total + Math.abs(height - heights[index + 1]), 0);

  let wells = 0;
  heights.forEach((height, index) => {
    const left = index > 0 ? heights[index - 1] : ROWS;
    const right = index < COLS - 1 ? heights[index + 1] : ROWS;
    const rim = Math.min(left, right);
    if (rim > height) wells += rim - height;
  });

  return {
    holes,
    maxHeight: Math.max(...heights),
    aggregateHeight: heights.reduce((total, height) => total + height, 0),
    bumpiness,
    completeLines: board.filter((row) => row.every(Boolean)).length,
    wells,
    filledCells: board.flat().filter(Boolean).length,
  };
}

export function featureVector(board, nextPieceName, cleared = 0) {
  const metrics = boardMetrics(board);
  return [
    ...board.flat().map((cell) => Number(Boolean(cell))),
    ...PIECES.map((piece) => (piece === nextPieceName ? 1 : 0)),
    cleared / 4,
    metrics.holes / 80,
    metrics.maxHeight / ROWS,
    metrics.aggregateHeight / (ROWS * COLS),
    metrics.bumpiness / (ROWS * COLS),
    metrics.completeLines / 4,
    metrics.wells / (ROWS * COLS),
    metrics.filledCells / (ROWS * COLS),
  ];
}

function rewardFor(board, cleared, done) {
  const metrics = boardMetrics(board);
  const lineReward = [0, 1, 3, 5, 8][Math.min(4, cleared)];
  return (
    0.08 +
    lineReward -
    0.035 * metrics.holes -
    0.018 * metrics.maxHeight -
    0.01 * metrics.bumpiness -
    (done ? 8 : 0)
  );
}

export function enumeratePlacements(state) {
  if (state.gameOver || state.paused) return [];

  const placements = [];
  const seen = new Set();

  for (let rotationCount = 0; rotationCount < 4; rotationCount += 1) {
    const rotatedPiece = clonePiece(state.activePiece);
    const rotationActions = [];
    for (let index = 0; index < rotationCount; index += 1) {
      if (rotatePiece(state.board, rotatedPiece)) {
        rotationActions.push(ACTIONS.rotate);
      }
    }

    for (let targetX = -4; targetX < COLS + 4; targetX += 1) {
      const piece = clonePiece(rotatedPiece);
      const horizontalActions = [];

      while (piece.x > targetX) {
        if (collides(state.board, piece, -1, 0)) break;
        piece.x -= 1;
        horizontalActions.push(ACTIONS.left);
      }

      while (piece.x < targetX) {
        if (collides(state.board, piece, 1, 0)) break;
        piece.x += 1;
        horizontalActions.push(ACTIONS.right);
      }

      if (piece.x !== targetX || collides(state.board, piece)) continue;

      while (!collides(state.board, piece, 0, 1)) {
        piece.y += 1;
      }

      const { board, cleared } = lockPiece(state.board, piece);
      const key = board.map((row) => row.join('')).join('|');
      if (seen.has(key)) continue;
      seen.add(key);

      const done = state.nextPiece.cells.some(([cellX, cellY]) => {
        const x = state.nextPiece.x + cellX;
        const y = state.nextPiece.y + cellY;
        return y >= 0 && board[y][x];
      });

      placements.push({
        actions: [...rotationActions, ...horizontalActions, ACTIONS.hardDrop],
        rotationCount,
        targetX,
        board,
        cleared,
        done,
        nextPiece: state.nextPiece.name,
        vector: featureVector(board, state.nextPiece.name, cleared),
        reward: rewardFor(board, cleared, done),
      });
    }
  }

  return placements.sort((a, b) => a.rotationCount - b.rotationCount || a.targetX - b.targetX);
}

function dense(vector, layer, useRelu) {
  return layer.bias.map((bias, rowIndex) => {
    const value = layer.weight[rowIndex].reduce((total, weight, index) => total + weight * vector[index], bias);
    return useRelu ? Math.max(0, value) : value;
  });
}

export function evaluateModel(model, vector) {
  if (!model || model.inputSize !== vector.length) {
    throw new Error(`Model input size mismatch: expected ${vector.length}`);
  }

  let current = vector;
  model.layers.forEach((layer, index) => {
    current = dense(current, layer, index < model.layers.length - 1);
  });
  return current[0];
}

export function choosePlacement(state, model = null) {
  const placements = enumeratePlacements(state);
  if (!placements.length) return null;

  return placements.reduce((best, placement) => {
    const value = model ? evaluateModel(model, placement.vector) : placement.reward;
    const bestValue = model ? evaluateModel(model, best.vector) : best.reward;
    return value > bestValue ? placement : best;
  }, placements[0]);
}

export function parseAgentModel(text) {
  const model = JSON.parse(text);
  if (model.type !== 'afterstate-value-mlp' || model.inputSize !== FEATURE_SIZE) {
    throw new Error('Unsupported Tetris agent model');
  }
  return model;
}

export async function loadModelFromUrl(url) {
  const response = await fetch(url, { cache: 'no-store' });
  if (!response.ok) {
    throw new Error(`Could not load model from ${url}: ${response.status}`);
  }
  return parseAgentModel(await response.text());
}

export function createReplayController(replay) {
  let index = 0;
  const frames = Array.isArray(replay?.frames) ? replay.frames : [];

  return {
    get length() {
      return frames.length;
    },
    get index() {
      return index;
    },
    reset() {
      index = 0;
      return frames[index] ?? null;
    },
    next() {
      if (!frames.length) return null;
      const frame = frames[index];
      index = Math.min(frames.length - 1, index + 1);
      return frame;
    },
  };
}
