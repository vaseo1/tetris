from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import best_device, make_value_net, require_torch
from .recovery import RECOVERY_SEVERITIES
from .train import HELD_OUT_SEEDS, SAFETY_PROFILES, evaluate, load_exported_model, safety_config_from_args


def parse_sweep_weights(value: str) -> list[float]:
    weights = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not weights:
        raise argparse.ArgumentTypeError("provide at least one comma-separated weight")
    if any(weight < 0.0 for weight in weights):
        raise argparse.ArgumentTypeError("safety sweep weights must be >= 0")
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an exported Tetris agent model.")
    parser.add_argument("model", type=Path)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--eval-workers", type=int, default=1, help="Evaluation worker processes. Use 0 for auto.")
    parser.add_argument("--episodes-output", type=Path, help="Write per-seed evaluation episodes to this JSON file.")
    parser.add_argument("--failures-output", type=Path, help="Write failed episodes, sorted by survival time, to this JSON file.")
    parser.add_argument("--safety-profile", choices=SAFETY_PROFILES, default="none")
    parser.add_argument("--safety-weight", type=float, default=1.0)
    parser.add_argument("--safety-trigger-height", type=int, default=14)
    parser.add_argument("--safety-trigger-holes", type=int, default=8)
    parser.add_argument("--safety-trigger-covered-holes", type=int, default=40)
    parser.add_argument("--safety-trigger-wells", type=int, default=8)
    parser.add_argument(
        "--safety-sweep-weights",
        type=parse_sweep_weights,
        help="Comma-separated safety weights to evaluate in one run.",
    )
    parser.add_argument("--start-mode", choices=("clean", "recovery"), default="clean")
    parser.add_argument("--recovery-severity", choices=RECOVERY_SEVERITIES, default="medium")
    args = parser.parse_args()
    if args.safety_sweep_weights is not None and args.safety_profile == "none":
        parser.error("--safety-sweep-weights requires --safety-profile")

    torch, _ = require_torch()
    device = best_device(torch)
    model = make_value_net().to(device)
    load_exported_model(model, torch, args.model)
    model.to(device)
    model.eval()
    if args.safety_sweep_weights is not None:
        sweep_results = []
        for weight in args.safety_sweep_weights:
            args.safety_weight = weight
            result = evaluate(
                model,
                torch,
                device,
                HELD_OUT_SEEDS[: args.seeds],
                args.seconds,
                capture_replay=False,
                eval_workers=args.eval_workers,
                start_mode=args.start_mode,
                recovery_severity=args.recovery_severity,
                safety_config=safety_config_from_args(args),
            )
            summary = {key: value for key, value in result.items() if key not in ("episodes", "replay")}
            summary["safetyProfile"] = args.safety_profile
            summary["safetyWeight"] = weight
            sweep_results.append(summary)
        print(json.dumps(sweep_results, indent=2))
        return

    result = evaluate(
        model,
        torch,
        device,
        HELD_OUT_SEEDS[: args.seeds],
        args.seconds,
        bool(args.replay),
        eval_workers=args.eval_workers,
        start_mode=args.start_mode,
        recovery_severity=args.recovery_severity,
        safety_config=safety_config_from_args(args),
    )
    if args.replay and result["replay"]:
        args.replay.parent.mkdir(parents=True, exist_ok=True)
        args.replay.write_text(json.dumps(result["replay"]), encoding="utf-8")
    if args.episodes_output:
        args.episodes_output.parent.mkdir(parents=True, exist_ok=True)
        args.episodes_output.write_text(json.dumps(result["episodes"]), encoding="utf-8")
    if args.failures_output:
        failures = [episode for episode in result["episodes"] if episode["survivalSeconds"] < args.seconds]
        failures.sort(key=lambda episode: episode["survivalSeconds"])
        args.failures_output.parent.mkdir(parents=True, exist_ok=True)
        args.failures_output.write_text(json.dumps(failures), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in ("episodes", "replay")}, indent=2))


if __name__ == "__main__":
    main()
