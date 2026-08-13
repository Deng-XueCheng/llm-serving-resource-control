from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from experiments.final_characterization import make_trace, write_trace
from experiments.run_multi_replica import run
from experiments.snapshot_preflight import model_contract


LOADS = (1.2, 2.4, 3.6, 4.8, 6.0, 7.2)
ROUTERS = ("round_robin", "resource_aware")


def calibration_config(
    trace_path: Path,
    *,
    model_path: Path,
    port: int,
    router: str,
) -> dict:
    return {
        **model_contract(model_path),
        "trace_path": str(trace_path),
        "token_id_upper_bound": 10000,
        "router": router,
        "prefix_aware": False,
        "replicas": {"gpu_ids": [0, 1], "distributed_init_port_base": port},
        "engine": {
            "enforce_eager": True, "tensor_parallel_size": 1,
            "max_model_len": 1024, "max_num_batched_tokens": 256,
            "max_num_seqs": 8, "gpu_memory_utilization": 0.35,
            "kvcache_block_size": 256, "num_kvcache_blocks": 8,
            "scheduler_policy": "mixed_token_budget", "decode_token_budget": 8,
            "incremental_kv_allocation": False,
        },
        "sampling": {"temperature": 0.6, "ignore_eos": True, "seed": 1810},
        "measurement_start": 0.0, "measurement_end": 20.0,
        "max_run_seconds": 90.0, "ttft_slo_ms": 5000.0,
        "itl_slo_ms": 1000.0, "require_itl": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--loads", type=float, nargs="+")
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    cells = []
    for load_index, offered_rps in enumerate(args.loads or LOADS):
        trace_path = Path(
            f"experiments/data/post_fix_stage18_calibration_{offered_rps:g}rps.json"
        )
        write_trace(trace_path, make_trace(offered_rps=offered_rps, seed=1))
        for router_index, router in enumerate(ROUTERS):
            config = calibration_config(
                trace_path,
                model_path=args.model_path,
                port=27400 + load_index * 10 + router_index * 2,
                router=router,
            )
            cell_id = f"load{offered_rps:g}_{router}"
            result = run(deepcopy(config))
            result["calibration"] = {"offered_rps": offered_rps, "cell_id": cell_id}
            output = args.output_directory / f"{cell_id}.json"
            output.write_text(json.dumps(result, indent=2) + "\n")
            cells.append({
                "cell_id": cell_id,
                "offered_rps": offered_rps,
                "router": router,
                "artifact": str(output),
                "summary": result["summary"],
                "routing_summary": result["routing_summary"],
                "gpu_provenance_status": result["gpu_provenance"]["cell_status"],
            })
    (args.output_directory / "summary.json").write_text(
        json.dumps({"schema_version": 1, "cells": cells}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
