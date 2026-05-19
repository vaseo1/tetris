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
- New best evaluations write both `best-model.json` and `checkpoint-best.pt.gz`.

Next guarded long-survival stability probe:

```bash
.venv/bin/python -m tetris_ai.train \
  --init-model runs/tetris-agent/best-model.json \
  --episodes 12 \
  --milestone-seconds 14400 \
  --max-pieces 23000 \
  --eval-seeds 50 \
  --eval-interval 1 \
  --reward-profile survival-v2 \
  --eps-start 0 \
  --eps-end 0 \
  --warmup-replay-steps 30000 \
  --optimizer-update-interval 5000 \
  --eval-regression-tolerance 0.04 \
  --source-anchor-weight 0.50 \
  --best-model-objective survival \
  --recovery-start-rate 0.00 \
  --learning-rate 0.00000001
```
