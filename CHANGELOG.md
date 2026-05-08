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
