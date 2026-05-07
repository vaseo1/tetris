import assert from 'node:assert/strict';
import test from 'node:test';
import {
  FEATURE_SIZE,
  choosePlacement,
  createReplayController,
  enumeratePlacements,
  featureVector,
} from '../src/agent.js';
import { createGame, getGameState, stepGame } from '../src/engine.js';

function replayActions(seed, actions) {
  const game = createGame({ seed });
  for (const action of actions) {
    stepGame(game, action);
  }
  return getGameState(game);
}

test('agent enumerates replayable afterstate placements', () => {
  const seed = 'agent-js';
  const game = createGame({ seed });
  const state = getGameState(game);
  const placements = enumeratePlacements(state);

  assert.ok(placements.length > 0);
  const placement = placements[0];
  const replayed = replayActions(seed, placement.actions);

  assert.deepEqual(replayed.board, placement.board);
  assert.equal(placement.vector.length, FEATURE_SIZE);
  assert.equal(featureVector(placement.board, placement.nextPiece, placement.cleared).length, FEATURE_SIZE);
});

test('heuristic placement selector returns a legal action list', () => {
  const state = getGameState(createGame({ seed: 71 }));
  const placement = choosePlacement(state);

  assert.ok(placement);
  assert.ok(placement.actions.length > 0);
  assert.equal(placement.actions.at(-1), 'hardDrop');
});

test('replay controller steps through saved frames', () => {
  const frames = [
    { state: { score: 0 }, actions: ['left'] },
    { state: { score: 1 }, actions: ['hardDrop'] },
  ];
  const replay = createReplayController({ frames });

  assert.equal(replay.length, 2);
  assert.equal(replay.reset().state.score, 0);
  assert.equal(replay.next().state.score, 0);
  assert.equal(replay.next().state.score, 1);
});
