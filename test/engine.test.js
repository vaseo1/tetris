import assert from 'node:assert/strict';
import test from 'node:test';
import {
  ACTIONS,
  COLS,
  ROWS,
  advanceGravity,
  createGame,
  getGameState,
  stepGame,
} from '../src/engine.js';

function pieceSignature(piece) {
  return `${piece.name}:${piece.x},${piece.y}:${piece.cells.map((cell) => cell.join(',')).join('|')}`;
}

test('seeded games produce the same piece sequence', () => {
  const first = createGame({ seed: 42 });
  const second = createGame({ seed: 42 });
  const firstSequence = [];
  const secondSequence = [];

  for (let index = 0; index < 8; index += 1) {
    firstSequence.push(pieceSignature(getGameState(first).activePiece));
    secondSequence.push(pieceSignature(getGameState(second).activePiece));
    stepGame(first, ACTIONS.hardDrop);
    stepGame(second, ACTIONS.hardDrop);
  }

  assert.deepEqual(firstSequence, secondSequence);
});

test('actions move, rotate, pause, and restart through the engine', () => {
  const game = createGame({ seed: 7 });
  const start = getGameState(game);

  stepGame(game, ACTIONS.left);
  assert.equal(getGameState(game).activePiece.x, start.activePiece.x - 1);

  stepGame(game, ACTIONS.right);
  assert.equal(getGameState(game).activePiece.x, start.activePiece.x);

  const beforeRotate = pieceSignature(getGameState(game).activePiece);
  stepGame(game, ACTIONS.rotate);
  assert.notEqual(pieceSignature(getGameState(game).activePiece), beforeRotate);

  stepGame(game, ACTIONS.pause);
  assert.equal(getGameState(game).paused, true);

  const pausedY = getGameState(game).activePiece.y;
  advanceGravity(game, 3);
  assert.equal(getGameState(game).activePiece.y, pausedY);

  stepGame(game, ACTIONS.restart);
  const restarted = getGameState(game);
  assert.equal(restarted.score, 0);
  assert.equal(restarted.paused, false);
  assert.equal(restarted.status, 'READY');
  assert.equal(pieceSignature(restarted.activePiece), pieceSignature(start.activePiece));
});

test('soft drop and gravity lock pieces when they reach the floor', () => {
  const game = createGame({ seed: 12 });

  for (let index = 0; index < ROWS + 4; index += 1) {
    stepGame(game, ACTIONS.down);
  }

  const state = getGameState(game);
  assert.equal(state.board.flat().some(Boolean), true);
  assert.equal(state.status, 'PLAY');
});

test('hard drop clears one point per filled row', () => {
  const game = createGame({ seed: 1 });
  game.board[ROWS - 2] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0];
  game.board[ROWS - 1] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0];
  game.activePiece = {
    name: 'O',
    cells: [[1, 0], [2, 0], [1, 1], [2, 1]],
    x: 7,
    y: ROWS - 2,
  };

  const state = stepGame(game, ACTIONS.hardDrop);

  assert.equal(state.score, 2);
  assert.equal(state.linesCleared, 2);
  assert.deepEqual(state.board[ROWS - 1], Array(COLS).fill(0));
});

test('failed spawn stops the game until restart', () => {
  const game = createGame({ seed: 4 });
  game.board[0][4] = 1;
  game.nextPiece = {
    name: 'O',
    cells: [[1, 0], [2, 0], [1, 1], [2, 1]],
    x: 3,
    y: 0,
  };

  const gameOver = stepGame(game, ACTIONS.hardDrop);
  assert.equal(gameOver.gameOver, true);
  assert.equal(gameOver.status, 'GAME OVER');

  const yAfterGameOver = gameOver.activePiece.y;
  stepGame(game, ACTIONS.down);
  assert.equal(getGameState(game).activePiece.y, yAfterGameOver);

  const restarted = stepGame(game, ACTIONS.restart);
  assert.equal(restarted.gameOver, false);
  assert.equal(restarted.status, 'READY');
});

test('many programmatic steps run synchronously without rendering', () => {
  const game = createGame({ seed: 'agent-smoke' });
  let state = getGameState(game);

  for (let index = 0; index < 1000 && !state.gameOver; index += 1) {
    state = stepGame(game, index % 5 === 0 ? ACTIONS.rotate : ACTIONS.hardDrop);
  }

  assert.equal(state.cols, COLS);
  assert.equal(state.rows, ROWS);
  assert.equal(state.board.length, ROWS);
  assert.equal(state.board.every((row) => row.length === COLS), true);
});
