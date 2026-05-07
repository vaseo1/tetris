import {
  ACTIONS,
  createGame,
  getGameState,
  stepGame,
} from '../src/engine.js';

const [seedText = 'agent-parity', actionText = ''] = process.argv.slice(2);
const seed = /^-?\d+$/.test(seedText) ? Number(seedText) : seedText;
const game = createGame({ seed });
const actions = actionText ? actionText.split(',') : [];

for (const action of actions) {
  stepGame(game, ACTIONS[action] ?? action);
}

console.log(JSON.stringify(getGameState(game)));
