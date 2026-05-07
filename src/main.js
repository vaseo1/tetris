import './style.css';
import {
  ACTIONS,
  COLS,
  ROWS,
  advanceGravity,
  createGame,
  getGameState,
  stepGame,
} from './engine.js';

const CELL = 30;
const GRAVITY_MS = 700;

const COLORS = {
  empty: '#a7b9af',
  ghost: '#80948b',
  ink: '#1c3638',
  dark: '#213d40',
  light: '#d0ddd6',
  mid: '#8fa39a',
};

const boardCanvas = document.querySelector('#board');
const nextCanvas = document.querySelector('#next');
const scoreNode = document.querySelector('#score');
const statusNode = document.querySelector('#status');
const ctx = boardCanvas.getContext('2d');
const nextCtx = nextCanvas.getContext('2d');

let game = createGame();
let state = getGameState(game);
let lastGravityAt = performance.now();

function resetGravityClock() {
  lastGravityAt = performance.now();
}

function renderCurrentState() {
  state = getGameState(game);
  updateScore();
  setStatus(state.status);
  render();
}

function resetGame(options = {}) {
  game = createGame(options);
  resetGravityClock();
  renderCurrentState();
  return state;
}

function runAction(action) {
  state = stepGame(game, action);
  if (
    action === ACTIONS.down ||
    action === ACTIONS.hardDrop ||
    action === ACTIONS.pause ||
    action === ACTIONS.restart
  ) {
    resetGravityClock();
  }
  return state;
}

function applyAction(action) {
  runAction(action);
  updateScore();
  setStatus(state.status);
  render();
  return state;
}

function setStatus(text) {
  statusNode.textContent = text;
}

function updateScore() {
  scoreNode.textContent = String(state.score).padStart(4, '0');
}

function drawCell(context, x, y, size, filled, ghost = false) {
  const originX = x * size;
  const originY = y * size;
  const gap = Math.max(2, Math.floor(size * 0.08));
  const inset = Math.max(6, Math.floor(size * 0.27));

  context.fillStyle = filled ? COLORS.mid : COLORS.empty;
  context.fillRect(originX + gap, originY + gap, size - gap * 2, size - gap * 2);

  context.strokeStyle = filled || ghost ? COLORS.ink : 'rgba(28, 54, 56, 0.2)';
  context.lineWidth = filled || ghost ? 3 : 1;
  context.strokeRect(originX + gap + 1, originY + gap + 1, size - gap * 2 - 2, size - gap * 2 - 2);

  if (filled || ghost) {
    context.fillStyle = ghost ? COLORS.ghost : COLORS.dark;
    context.fillRect(originX + inset, originY + inset, size - inset * 2, size - inset * 2);
    context.strokeStyle = COLORS.light;
    context.lineWidth = 1;
    context.strokeRect(originX + inset, originY + inset, size - inset * 2, size - inset * 2);
  }
}

function drawBoard() {
  ctx.clearRect(0, 0, boardCanvas.width, boardCanvas.height);
  ctx.fillStyle = COLORS.empty;
  ctx.fillRect(0, 0, boardCanvas.width, boardCanvas.height);

  for (let y = 0; y < ROWS; y += 1) {
    for (let x = 0; x < COLS; x += 1) {
      drawCell(ctx, x, y, CELL, Boolean(state.board[y][x]));
    }
  }

  state.activePiece.cells.forEach(([cellX, cellY]) => {
    const x = state.activePiece.x + cellX;
    const y = state.activePiece.y + cellY;
    if (y >= 0) {
      drawCell(ctx, x, y, CELL, true);
    }
  });

  if (state.paused || state.gameOver) {
    ctx.fillStyle = 'rgba(167, 185, 175, 0.78)';
    ctx.fillRect(0, 0, boardCanvas.width, boardCanvas.height);
    ctx.fillStyle = COLORS.ink;
    ctx.font = 'bold 26px ui-monospace, Menlo, Consolas, monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(state.paused ? 'PAUSED' : 'GAME OVER', boardCanvas.width / 2, boardCanvas.height / 2);
  }
}

function drawNext() {
  const size = 24;
  nextCtx.clearRect(0, 0, nextCanvas.width, nextCanvas.height);
  nextCtx.fillStyle = COLORS.empty;
  nextCtx.fillRect(0, 0, nextCanvas.width, nextCanvas.height);

  for (let y = 0; y < 4; y += 1) {
    for (let x = 0; x < 4; x += 1) {
      drawCell(nextCtx, x + 0.5, y + 0.5, size, false);
    }
  }

  state.nextPiece.cells.forEach(([x, y]) => {
    drawCell(nextCtx, x + 0.5, y + 0.5, size, true);
  });
}

function render() {
  drawBoard();
  drawNext();
}

function tick(time) {
  if (!state.gameOver && !state.paused && time - lastGravityAt >= GRAVITY_MS) {
    const ticks = Math.floor((time - lastGravityAt) / GRAVITY_MS);
    state = advanceGravity(game, ticks);
    lastGravityAt += ticks * GRAVITY_MS;
    updateScore();
    setStatus(state.status);
    render();
  }

  requestAnimationFrame(tick);
}

window.addEventListener('keydown', (event) => {
  const keyActions = {
    ArrowLeft: ACTIONS.left,
    ArrowRight: ACTIONS.right,
    ArrowDown: ACTIONS.down,
    ArrowUp: ACTIONS.rotate,
    ' ': ACTIONS.hardDrop,
    p: ACTIONS.pause,
    P: ACTIONS.pause,
    r: ACTIONS.restart,
    R: ACTIONS.restart,
  };

  const action = keyActions[event.key];
  if (!action) return;

  event.preventDefault();

  if (action === ACTIONS.restart) {
    resetGame({ seed: game.seed });
  } else {
    applyAction(action);
  }
});

window.tetrisAgent = {
  actions: { ...ACTIONS },
  reset(options = {}) {
    return resetGame(options);
  },
  step(action = ACTIONS.noop) {
    return applyAction(action);
  },
  stepMany(actions = []) {
    let latest = state;
    for (const action of actions) {
      latest = runAction(action);
    }
    updateScore();
    setStatus(latest.status);
    render();
    return latest;
  },
  getState() {
    state = getGameState(game);
    return state;
  },
};

renderCurrentState();
requestAnimationFrame(tick);
