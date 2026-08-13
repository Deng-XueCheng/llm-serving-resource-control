from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
from math import isfinite
from pathlib import Path
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def generate_trace(
    *,
    duration_seconds: float,
    request_count: int,
    interactive_per_cycle: int,
    long_per_cycle: int,
    interactive: dict[str, int],
    long_request: dict[str, int],
    seed: int,
    prompt_seed_base: int,
    description: str,
) -> dict[str, Any]:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if request_count <= 0:
        raise ValueError("request_count must be positive")
    cycle_size = interactive_per_cycle + long_per_cycle
    if cycle_size <= 0 or request_count % cycle_size != 0:
        raise ValueError("request_count must contain complete class cycles")

    generator = random.Random(seed)
    arrivals = sorted(
        generator.random() * duration_seconds
        for _ in range(request_count)
    )
    classes = (
        ["interactive"] * interactive_per_cycle
        + ["long"] * long_per_cycle
    ) * (request_count // cycle_size)
    generator.shuffle(classes)

    class_counters = {"interactive": 0, "long": 0}
    requests = []
    for index, (arrival_time, request_class) in enumerate(
        zip(arrivals, classes, strict=True)
    ):
        class_index = class_counters[request_class]
        class_counters[request_class] += 1
        profile = interactive if request_class == "interactive" else long_request
        requests.append(
            {
                "request_id": f"{request_class}-{class_index:03d}",
                "request_class": request_class,
                "arrival_time": arrival_time,
                "prompt_length": profile["prompt_length"],
                "max_output_tokens": profile["max_output_tokens"],
                "seed": prompt_seed_base + seed * 1000 + index,
            }
        )

    return {
        "schema_version": 1,
        "description": description,
        "time_unit": "seconds",
        "requests": requests,
    }


def build_benchmark_config(
    template: dict[str, Any],
    *,
    trace_path: str,
    run_id: str,
    inference_seed: int,
    duration_seconds: float,
) -> dict[str, Any]:
    config = copy.deepcopy(template)
    config["sampling"]["seed"] = inference_seed
    config["workload"]["trace_path"] = trace_path
    config["measurement"]["start_seconds"] = 0.0
    config["measurement"]["end_seconds"] = duration_seconds
    config["output"]["run_id"] = run_id
    return config


def write_json_idempotent(path: Path, value: Any) -> str:
    serialized = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(
                f"Refusing to replace non-identical generated file: {path}"
            )
        return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
        newline="\n",
    ) as file:
        temporary_path = Path(file.name)
        file.write(serialized)
        file.flush()
        os.fsync(file.fileno())
    try:
        os.link(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return "created"


def load_matrix_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = json.load(file)
    expected = {
        "schema_version",
        "description",
        "duration_seconds",
        "class_cycle",
        "profiles",
        "loads",
        "seeds",
        "prompt_seed_base",
        "inference_seed_base",
        "template_config_path",
        "trace_output_directory",
        "config_output_directory",
        "run_id_prefix",
    }
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError(f"Matrix config must contain exactly {sorted(expected)}")
    if (
        not isinstance(config["schema_version"], int)
        or isinstance(config["schema_version"], bool)
        or config["schema_version"] != 1
    ):
        raise ValueError("schema_version must be 1")
    if not isinstance(config["description"], str) or not config["description"]:
        raise ValueError("description must be a non-empty string")
    duration = config["duration_seconds"]
    if (
        not isinstance(duration, (int, float))
        or isinstance(duration, bool)
        or not isfinite(duration)
        or duration <= 0
    ):
        raise ValueError("duration_seconds must be finite and positive")

    cycle = config["class_cycle"]
    if not isinstance(cycle, dict) or set(cycle) != {"interactive", "long"}:
        raise ValueError("class_cycle must contain interactive and long")
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
        for value in cycle.values()
    ):
        raise ValueError("class_cycle values must be positive integers")

    profiles = config["profiles"]
    if (
        not isinstance(profiles, dict)
        or set(profiles) != {"interactive", "long"}
    ):
        raise ValueError("profiles must contain interactive and long")
    for name, profile in profiles.items():
        if (
            not isinstance(profile, dict)
            or set(profile) != {"prompt_length", "max_output_tokens"}
        ):
            raise ValueError(f"Invalid profile schema: {name}")
        if any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in profile.values()
        ):
            raise ValueError(f"Profile values must be positive: {name}")

    loads = config["loads"]
    if (
        not isinstance(loads, dict)
        or set(loads) != {"low", "critical", "overload"}
    ):
        raise ValueError("loads must contain low, critical, and overload")
    cycle_size = cycle["interactive"] + cycle["long"]
    cycle_output_tokens = (
        cycle["interactive"] * profiles["interactive"]["max_output_tokens"]
        + cycle["long"] * profiles["long"]["max_output_tokens"]
    )
    offered_values = []
    for name in ("low", "critical", "overload"):
        load = loads[name]
        if (
            not isinstance(load, dict)
            or set(load) != {"request_count", "offered_output_tps"}
        ):
            raise ValueError(f"Invalid load schema: {name}")
        request_count = load["request_count"]
        offered_output_tps = load["offered_output_tps"]
        if (
            not isinstance(request_count, int)
            or isinstance(request_count, bool)
            or request_count <= 0
            or request_count % cycle_size != 0
        ):
            raise ValueError(
                f"{name}.request_count must contain complete class cycles"
            )
        if (
            not isinstance(offered_output_tps, (int, float))
            or isinstance(offered_output_tps, bool)
            or not isfinite(offered_output_tps)
            or offered_output_tps <= 0
        ):
            raise ValueError(f"{name}.offered_output_tps must be positive")
        expected_tps = (
            request_count / cycle_size * cycle_output_tokens / duration
        )
        if abs(offered_output_tps - expected_tps) > 1e-9:
            raise ValueError(
                f"{name}.offered_output_tps={offered_output_tps} "
                f"does not match trace definition {expected_tps}"
            )
        offered_values.append(offered_output_tps)
    if offered_values != sorted(offered_values) or len(set(offered_values)) != 3:
        raise ValueError("load offered_output_tps must increase strictly")

    seeds = config["seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != 3
        or len(seeds) != len(set(seeds))
        or any(
            not isinstance(seed, int)
            or isinstance(seed, bool)
            or seed < 0
            for seed in seeds
        )
    ):
        raise ValueError("seeds must contain exactly 3 unique non-negative integers")
    for field in ("prompt_seed_base", "inference_seed_base"):
        value = config[field]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
        ):
            raise ValueError(f"{field} must be a non-negative integer")
    for field in (
        "template_config_path",
        "trace_output_directory",
        "config_output_directory",
    ):
        if not isinstance(config[field], str) or not config[field]:
            raise ValueError(f"{field} must be a non-empty string")
    validate_safe_component("run_id_prefix", config["run_id_prefix"])
    return config


def validate_safe_component(name: str, value: Any) -> None:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value)
        or ".." in value
    ):
        raise ValueError(f"{name} contains unsafe characters: {value!r}")


def resolve_repo_directory(value: str, allowed_root: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    root = (REPO_ROOT / allowed_root).resolve()
    if not path.is_relative_to(root):
        raise ValueError(
            f"Path must stay inside {allowed_root}: {value}"
        )
    return path


def generate_matrix(config: dict[str, Any]) -> list[dict[str, str]]:
    template_path = (REPO_ROOT / config["template_config_path"]).resolve()
    if not template_path.is_relative_to(REPO_ROOT) or not template_path.is_file():
        raise ValueError("template_config_path must be a repository file")
    template = json.loads(template_path.read_text(encoding="utf-8"))
    trace_directory = resolve_repo_directory(
        config["trace_output_directory"],
        "experiments/data",
    )
    benchmark_directory = resolve_repo_directory(
        config["config_output_directory"],
        "experiments/configs",
    )
    duration = config["duration_seconds"]
    cycle = config["class_cycle"]
    profiles = config["profiles"]
    if template["measurement"]["max_run_seconds"] < duration:
        raise ValueError("Template max_run_seconds is shorter than matrix duration")
    max_model_len = template["engine"]["max_model_len"]
    for name, profile in profiles.items():
        if (
            profile["prompt_length"] + profile["max_output_tokens"]
            > max_model_len
        ):
            raise ValueError(f"{name} profile exceeds template max_model_len")
    planned = []

    for load_name, load in config["loads"].items():
        validate_safe_component("load name", load_name)
        request_count = load["request_count"]
        for seed in config["seeds"]:
            stem = f"{config['run_id_prefix']}_{load_name}_seed{seed}"
            validate_safe_component("generated stem", stem)
            trace_relative = (
                Path(config["trace_output_directory"]) / f"{stem}.json"
            )
            trace = generate_trace(
                duration_seconds=duration,
                request_count=request_count,
                interactive_per_cycle=cycle["interactive"],
                long_per_cycle=cycle["long"],
                interactive=profiles["interactive"],
                long_request=profiles["long"],
                seed=seed,
                prompt_seed_base=config["prompt_seed_base"],
                description=(
                    f"{config['description']} load={load_name} seed={seed}"
                ),
            )
            benchmark = build_benchmark_config(
                template,
                trace_path=trace_relative.as_posix(),
                run_id=stem,
                inference_seed=config["inference_seed_base"] + seed,
                duration_seconds=duration,
            )
            trace_path = trace_directory / f"{stem}.json"
            benchmark_path = benchmark_directory / f"{stem}.json"
            if not trace_path.resolve().is_relative_to(trace_directory):
                raise ValueError("Generated trace path leaves trace directory")
            if not benchmark_path.resolve().is_relative_to(benchmark_directory):
                raise ValueError("Generated config path leaves config directory")
            planned.append(
                {
                    "load": load_name,
                    "seed": str(seed),
                    "trace": str(trace_path),
                    "trace_value": trace,
                    "config": str(benchmark_path),
                    "config_value": benchmark,
                }
            )
    target_paths = [
        Path(record[field])
        for record in planned
        for field in ("trace", "config")
    ]
    if len(target_paths) != len(set(target_paths)):
        raise ValueError("Matrix generated duplicate target paths")
    for record in planned:
        for path_field, value_field in (
            ("trace", "trace_value"),
            ("config", "config_value"),
        ):
            path = Path(record[path_field])
            serialized = (
                json.dumps(record[value_field], ensure_ascii=False, indent=2)
                + "\n"
            )
            if path.exists() and path.read_text(encoding="utf-8") != serialized:
                raise FileExistsError(
                    f"Matrix preflight found conflicting file: {path}"
                )

    records = []
    for record in planned:
        records.append(
            {
                "load": record["load"],
                "seed": record["seed"],
                "trace": record["trace"],
                "trace_status": write_json_idempotent(
                    Path(record["trace"]),
                    record["trace_value"],
                ),
                "config": record["config"],
                "config_status": write_json_idempotent(
                    Path(record["config"]),
                    record["config_value"],
                ),
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    matrix_config = load_matrix_config(args.config.resolve())
    print(json.dumps(generate_matrix(matrix_config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
