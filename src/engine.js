export const COLS = 10;
export const ROWS = 20;

export const ACTIONS = {
  left: 'left',
  right: 'right',
  down: 'down',
  rotate: 'rotate',
  hardDrop: 'hardDrop',
  pause: 'pause',
  restart: 'restart',
  noop: 'noop',
};

export const SHAPES = [
  { name: 'I', cells: [[0, 1], [1, 1], [2, 1], [3, 1]] },
  { name: 'J', cells: [[0, 0], [0, 1], [1, 1], [2, 1]] },
  { name: 'L', cells: [[2, 0], [0, 1], [1, 1], [2, 1]] },
  { name: 'O', cells: [[1, 0], [2, 0], [1, 1], [2, 1]] },
  { name: 'S', cells: [[1, 0], [2, 0], [0, 1], [1, 1]] },
  { name: 'T', cells: [[1, 0], [0, 1], [1, 1], [2, 1]] },
  { name: 'Z', cells: [[0, 0], [1, 0], [1, 1], [2, 1]] },
];

function normalizeSeed(seed) {
  if (seed === undefined || seed === null) {
    return Math.floor(Math.random() * 0xffffffff);
  }

  if (typeof seed === 'number' && Number.isFinite(seed)) {
    return seed >>> 0;
  }

  const text = String(seed);
  let hash = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function nextRandom(game) {
  game.rngState = (Math.imul(game.rngState, 1664525) + 1013904223) >>> 0;
  return game.rngState / 0x100000000;
}

function makeBoard() {
  return Array.from({ length: ROWS }, () => Array(COLS).fill(0));
}

function cloneCells(cells) {
  return cells.map(([x, y]) => [x, y]);
}

function clonePiece(piece) {
  return {
    name: piece.name,
    cells: cloneCells(piece.cells),
    x: piece.x,
    y: piece.y,
  };
}

function createPiece(game) {
  const shape = SHAPES[Math.floor(nextRandom(game) * SHAPES.length)];
  return {
    name: shape.name,
    cells: cloneCells(shape.cells),
    x: Math.floor(COLS / 2) - 2,
    y: 0,
  };
}

function canPlay(game) {
  return !game.gameOver && !game.paused;
}

function collides(game, piece, offsetX = 0, offsetY = 0, cells = piece.cells) {
  return cells.some(([cellX, cellY]) => {
    const x = piece.x + cellX + offsetX;
    const y = piece.y + cellY + offsetY;
    return x < 0 || x >= COLS || y >= ROWS || (y >= 0 && game.board[y][x]);
  });
}

function setStatus(game, status) {
  game.status = status;
}

function clearRows(game) {
  let cleared = 0;
  game.board = game.board.filter((row) => {
    if (row.every(Boolean)) {
      cleared += 1;
      return false;
    }
    return true;
  });

  while (game.board.length < ROWS) {
    game.board.unshift(Array(COLS).fill(0));
  }

  if (cleared > 0) {
    game.score += cleared;
    game.linesCleared += cleared;
  }

  return cleared;
}

function move(game, dx, dy) {
  if (!canPlay(game) || collides(game, game.activePiece, dx, dy)) {
    return false;
  }

  game.activePiece.x += dx;
  game.activePiece.y += dy;
  setStatus(game, 'PLAY');
  return true;
}

function rotate(game) {
  if (!canPlay(game) || game.activePiece.name === 'O') {
    return false;
  }

  const rotated = game.activePiece.cells.map(([x, y]) => [3 - y, x]);
  const kicks = [0, -1, 1, -2, 2];

  for (const kick of kicks) {
    if (!collides(game, game.activePiece, kick, 0, rotated)) {
      game.activePiece.cells = rotated;
      game.activePiece.x += kick;
      setStatus(game, 'PLAY');
      return true;
    }
  }

  return false;
}

function lockPiece(game) {
  game.activePiece.cells.forEach(([cellX, cellY]) => {
    const x = game.activePiece.x + cellX;
    const y = game.activePiece.y + cellY;
    if (y >= 0 && y < ROWS && x >= 0 && x < COLS) {
      game.board[y][x] = 1;
    }
  });

  clearRows(game);
  game.activePiece = game.nextPiece;
  game.nextPiece = createPiece(game);

  if (collides(game, game.activePiece)) {
    game.gameOver = true;
    setStatus(game, 'GAME OVER');
  } else {
    setStatus(game, 'PLAY');
  }
}

function softDrop(game) {
  if (!canPlay(game)) {
    return false;
  }

  if (move(game, 0, 1)) {
    return true;
  }

  lockPiece(game);
  return true;
}

function hardDrop(game) {
  if (!canPlay(game)) {
    return false;
  }

  while (!collides(game, game.activePiece, 0, 1)) {
    game.activePiece.y += 1;
  }

  lockPiece(game);
  return true;
}

export function createGame({ seed } = {}) {
  const normalizedSeed = normalizeSeed(seed);
  const game = {
    cols: COLS,
    rows: ROWS,
    board: makeBoard(),
    activePiece: null,
    nextPiece: null,
    score: 0,
    linesCleared: 0,
    gameOver: false,
    paused: false,
    status: 'READY',
    seed: normalizedSeed,
    rngState: normalizedSeed,
    gravityTicks: 0,
  };

  game.activePiece = createPiece(game);
  game.nextPiece = createPiece(game);
  return game;
}

export function stepGame(game, action = ACTIONS.noop) {
  switch (action) {
    case ACTIONS.left:
      move(game, -1, 0);
      break;
    case ACTIONS.right:
      move(game, 1, 0);
      break;
    case ACTIONS.down:
      softDrop(game);
      break;
    case ACTIONS.rotate:
      rotate(game);
      break;
    case ACTIONS.hardDrop:
      hardDrop(game);
      break;
    case ACTIONS.pause:
      if (!game.gameOver) {
        game.paused = !game.paused;
        setStatus(game, game.paused ? 'PAUSED' : 'PLAY');
      }
      break;
    case ACTIONS.restart: {
      const restarted = createGame();
      Object.assign(game, restarted);
      break;
    }
    case ACTIONS.noop:
      if (canPlay(game)) {
        setStatus(game, 'PLAY');
      }
      break;
    default:
      throw new Error(`Unknown Tetris action: ${action}`);
  }

  return getGameState(game);
}

export function advanceGravity(game, ticks = 1) {
  const count = Math.max(0, Math.trunc(ticks));

  for (let index = 0; index < count; index += 1) {
    if (!canPlay(game)) {
      break;
    }
    game.gravityTicks += 1;
    softDrop(game);
  }

  return getGameState(game);
}

export function getGameState(game) {
  return {
    cols: game.cols,
    rows: game.rows,
    board: game.board.map((row) => [...row]),
    activePiece: clonePiece(game.activePiece),
    nextPiece: clonePiece(game.nextPiece),
    score: game.score,
    linesCleared: game.linesCleared,
    gameOver: game.gameOver,
    paused: game.paused,
    status: game.status,
    seed: game.seed,
    gravityTicks: game.gravityTicks,
  };
}
