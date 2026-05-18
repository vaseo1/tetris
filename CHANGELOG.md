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
- New best evaluations write both `best-model.json` and `checkpoint-best.pt.gz`.

Next long-survival stability probe:

```bash
.venv/bin/python -m tetris_ai.train \
  --init-model runs/tetris-agent/best-model.json \
  --episodes 100 \
  --milestone-seconds 14400 \
  --max-pieces 23000 \
  --eval-seeds 50 \
  --eval-interval 10 \
  --reward-profile phase2-score \
  --best-model-objective survival \
  --recovery-start-rate 0.00 \
  --learning-rate 0.0000005
```
