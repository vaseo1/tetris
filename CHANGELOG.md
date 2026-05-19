# Changelog

## Training phases

- Phase 1: survival-first training. Learn stable placement and board hygiene against the 60s milestone.

```bash
uv run python -m tetris_ai.train --episodes 5000 --milestone-seconds 60
```

- Phase 2: resume from the phase-1 checkpoint, reset replay/optimizer state, and fine-tune for higher score while requiring held-out survival.

```bash
uv run python -m tetris_ai.train --resume --episodes 3000 --reward-profile phase2-score --reset-replay-on-resume --reset-optimizer-on-resume --learning-rate 1e-4 --milestone-seconds 120 --best-model-objective score --score-survival-floor 0.70
```

## Best-model workflow

- Use `--resume` to continue the latest checkpoint exactly after an interrupt or crash.
- Use `--resume-best` to continue from `checkpoint-best.pt.gz` with optimizer, replay, RNG, and best tracking preserved.
- Use `--init-model runs/tetris-agent/best-model.json` to start a fresh training phase from exported best weights when changing the milestone, reward profile, or recovery schedule.
- `--init-model` starts at a mature epsilon decay step by default so fine-tuning does not restart with fully random exploration. Use `--init-model-step 0` only when deliberate fresh exploration is wanted.
- `--init-model` evaluates the source model before training and uses that result as the best baseline, so a new phase only overwrites `best-model.json` after beating the parent.
- `--init-model` writes protected experiment checkpoints to `checkpoint-init-model.pt.gz` instead of replacing the main `checkpoint.pt.gz`.
- `tetris_ai.evaluate` supports `--eval-workers`, `--episodes-output`, and `--failures-output` for long validation runs.
- Action selection now refuses terminal placements when any non-terminal placement exists, which protects long survival runs from value misrankings in near-top-out states.
- `--reward-profile survival-v2` adds stronger penalties for covered holes, deep wells, top-zone occupancy, and high-stack pressure without changing the exported model shape.
- `--warmup-replay-steps` collects replay before optimizer updates, useful when changing reward profiles from exported best weights.
- `--optimizer-update-interval` throttles optimizer updates after warmup, so long episodes cannot apply thousands of updates before the next eval.
- `--eval-regression-tolerance` stops an `--init-model` phase if held-out success falls too far below the source model baseline.
- `--source-anchor-weight` regularizes init-model fine-tuning toward the source model's Q-values to reduce catastrophic drift.
- `--safety-profile safety-v1` enables an inference-time placement reranker for risky states; `safety-v1` at weight `0.08` is the current confirmed baseline.
- `safety-v2` adds nonlinear penalties for critical top/height/hole afterstates, but broad held-out sweeps regressed versus `safety-v1`; keep it experimental rather than recommended.
- `--safety-profile safety-v3` keeps the `safety-v1` afterstate penalty and adds a capped recovery bonus for risky-state placements that clear lines, reduce max height, or reduce danger-zone pressure. A 50-seed, 4h sweep did not beat `safety-v1`, so do not promote it without further changes.
- `tetris_ai.evaluate --safety-sweep-weights` can test overlay weights without training.
- New best evaluations write both `best-model.json` and `checkpoint-best.pt.gz`.

Safety-overlay findings:

- Confirmed baseline: `safety-v1` at weight `0.08` had the best 200-seed, 4h result seen so far (`successRate=0.665`) and should remain the default candidate for long 6h checks.
- `safety-v1` 50-seed, 4h sweep previously reached `successRate=0.86` at weight `0.07`, `0.84` at weight `0.10`, and `0.82` at weight `0.16`.
- `safety-v3` 50-seed, 4h sweep underperformed: best was weight `0.05` at `successRate=0.76`; weights `0.07`, `0.08`, `0.10`, and `0.12` scored `0.68`, `0.60`, `0.64`, and `0.58`.
- Conclusion: the v3 recovery bonus targets the right catastrophic-failure idea, but this formula is too disruptive broadly. Do not run the 200-seed v3 confirmation unless the scoring changes.

Current recommended validation command:

```bash
.venv/bin/python -m tetris_ai.evaluate runs/tetris-agent/best-model.json \
  --seeds 100 \
  --seconds 21600 \
  --eval-workers 0 \
  --safety-profile safety-v1 \
  --safety-weight 0.08 \
  --failures-output runs/tetris-agent/failures-6h-100-safety-v1-w008.json
```
