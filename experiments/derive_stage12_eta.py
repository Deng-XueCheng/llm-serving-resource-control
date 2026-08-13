from __future__ import annotations

"""Derive conservative Stage 12 ETA constants from a dedicated calibration run."""

import argparse
import json
import math
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def percentile90(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("calibration values must be finite and non-negative")
    ordered = sorted(values)
    return ordered[math.ceil(0.9 * len(ordered)) - 1]


def derive(*, steps: list[dict[str, Any]], timings: list[dict[str, Any]]) -> dict[str, float | int]:
    timing_by_index = {item["step_index"]: item for item in timings}
    if len(timing_by_index) != len(timings):
        raise ValueError("duplicate phase timing index")
    prefill_ms: list[float] = []
    decode_per_token_ms: list[float] = []
    overhead_ms: list[float] = []
    for step in steps:
        timing = timing_by_index.get(step["step_index"])
        if timing is None:
            raise ValueError("missing phase timing")
        cuda_ms = timing.get("model_runner_cuda_ms")
        wall_ms = timing.get("step_wall_ms")
        if not isinstance(cuda_ms, (int, float)) or not isinstance(wall_ms, (int, float)):
            raise ValueError("invalid CUDA timing")
        if step["phase"] == "prefill":
            prefill_ms.append(float(cuda_ms))
        elif step["phase"] == "decode" and step["num_scheduled_tokens"] > 0:
            decode_per_token_ms.append(float(cuda_ms) / step["num_scheduled_tokens"])
        overhead_ms.append(max(0.0, float(wall_ms) - float(cuda_ms)))
    if not prefill_ms or not decode_per_token_ms:
        raise ValueError("calibration must contain both prefill and decode timing")
    return {
        "schema_version": 1,
        "prefill_samples": len(prefill_ms),
        "decode_samples": len(decode_per_token_ms),
        "eta_prefill_seconds": percentile90(prefill_ms) / 1000.0,
        "eta_decode_seconds_per_token": percentile90(decode_per_token_ms) / 1000.0,
        "eta_safety_margin_seconds": percentile90(overhead_ms) / 1000.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=Path, required=True)
    parser.add_argument("--phase-timings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = derive(steps=read_jsonl(args.steps), timings=read_jsonl(args.phase_timings))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
