from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import best_device, make_value_net, require_torch
from .recovery import RECOVERY_SEVERITIES
from .train import HELD_OUT_SEEDS, evaluate, load_exported_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an exported Tetris agent model.")
    parser.add_argument("model", type=Path)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--start-mode", choices=("clean", "recovery"), default="clean")
    parser.add_argument("--recovery-severity", choices=RECOVERY_SEVERITIES, default="medium")
    args = parser.parse_args()

    torch, _ = require_torch()
    device = best_device(torch)
    model = make_value_net().to(device)
    load_exported_model(model, torch, args.model)
    model.to(device)
    model.eval()
    result = evaluate(
        model,
        torch,
        device,
        HELD_OUT_SEEDS[: args.seeds],
        args.seconds,
        bool(args.replay),
        start_mode=args.start_mode,
        recovery_severity=args.recovery_severity,
    )
    if args.replay and result["replay"]:
        args.replay.parent.mkdir(parents=True, exist_ok=True)
        args.replay.write_text(json.dumps(result["replay"]), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("episodes", "replay")}, indent=2))


if __name__ == "__main__":
    main()
