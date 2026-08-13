from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.final_characterization import make_trace, write_trace
from experiments.run_multi_replica import run
from experiments.stage18_rebaseline import calibration_config


LOADS = (4.8, 6.0)
SEEDS = (1, 2, 3)
ROUTERS = ("round_robin", "resource_aware")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    cells = []
    for load_index, offered_rps in enumerate(LOADS):
        for seed in SEEDS:
            trace_path = Path(
                f"experiments/data/post_fix_stage18_formal_{offered_rps:g}rps_seed{seed}.json"
            )
            write_trace(trace_path, make_trace(offered_rps=offered_rps, seed=seed))
            for router_index, router in enumerate(ROUTERS):
                config = calibration_config(
                    trace_path,
                    model_path=args.model_path,
                    port=27600 + load_index * 100 + seed * 10 + router_index * 2,
                    router=router,
                )
                config["sampling"]["seed"] = 1900 + seed
                cell_id = f"load{offered_rps:g}_seed{seed}_{router}"
                result = run(config)
                result["formal"] = {
                    "cell_id": cell_id, "offered_rps": offered_rps,
                    "seed": seed, "router": router,
                }
                output = args.output_directory / f"{cell_id}.json"
                output.write_text(json.dumps(result, indent=2) + "\n")
                cells.append({
                    **result["formal"], "artifact": str(output),
                    "summary": result["summary"],
                    "routing_summary": result["routing_summary"],
                    "gpu_provenance_status": result["gpu_provenance"]["cell_status"],
                })
    (args.output_directory / "matrix_summary.json").write_text(
        json.dumps({
            "schema_version": 1,
            "evidence_generation": "POST_GPU_BINDING_FIX",
            "loads_selected_from": "post-fix Stage 18 calibration",
            "cells": cells,
        }, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
