import {
  ACTIONS,
  COLS,
  ROWS,
  createGame,
  getGameState,
  stepGame,
} from '../src/engine.js';

const [scenario] = process.argv.slice(2);
const game = createGame({ seed: 1 });

if (scenario === 'line-clear') {
  game.board[ROWS - 2] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0];
  game.board[ROWS - 1] = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0];
  game.activePiece = {
    name: 'O',
    cells: [[1, 0], [2, 0], [1, 1], [2, 1]],
    x: 7,
    y: ROWS - 2,
  };
  stepGame(game, ACTIONS.hardDrop);
} else if (scenario === 'top-out') {
  game.board[0][4] = 1;
  game.nextPiece = {
    name: 'O',
    cells: [[1, 0], [2, 0], [1, 1], [2, 1]],
    x: 3,
    y: 0,
  };
  stepGame(game, ACTIONS.hardDrop);
} else {
  throw new Error(`Unknown scenario: ${scenario}`);
}

console.log(JSON.stringify(getGameState(game)));
