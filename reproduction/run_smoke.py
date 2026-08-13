from __future__ import annotations

import atexit
import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import flash_attn
import torch
import transformers
import triton
from nanovllm import LLM, SamplingParams
from experiments.snapshot_preflight import verify_source_snapshot
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "reproduction/results"
REQUIREMENTS_PATH = REPO_ROOT / "reproduction/requirements.lock.txt"
PYLOCK_PATH = REPO_ROOT / "reproduction/pylock.toml"
LEGACY_RELEASE_COMMIT = "e8b5bf1cd2b982632d763b93448250c1157efba1"
TOP_LEVEL_KEYS = {
    "upstream_commit",
    "model_path",
    "model_repo_id",
    "model_revision",
    "model_hashes",
    "engine",
    "sampling",
    "prompts",
    "use_tqdm",
    "output_path",
}
ENGINE_KEYS = {
    "enforce_eager",
    "tensor_parallel_size",
    "max_model_len",
    "max_num_batched_tokens",
    "max_num_seqs",
    "gpu_memory_utilization",
    "kvcache_block_size",
}
SAMPLING_KEYS = {"temperature", "max_tokens", "ignore_eos"}


def load_config(
    path: Path,
    *,
    model_path_override: Path | None = None,
) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = json.load(file)

    if model_path_override is not None:
        config["model_path"] = str(model_path_override.resolve())

    validate_exact_keys("top-level config", config, TOP_LEVEL_KEYS)
    validate_exact_keys("engine", config["engine"], ENGINE_KEYS)
    validate_exact_keys("sampling", config["sampling"], SAMPLING_KEYS)

    model_path = Path(config["model_path"])
    if not model_path.is_absolute():
        raise ValueError("model_path must be absolute")
    for field in (
        "upstream_commit",
        "model_repo_id",
        "model_revision",
        "output_path",
    ):
        if not isinstance(config[field], str) or not config[field]:
            raise ValueError(f"{field} must be a non-empty string")
    upstream_commit = config["upstream_commit"]
    if (
        len(upstream_commit) != 40
        or any(character not in "0123456789abcdef" for character in upstream_commit)
    ):
        raise ValueError("upstream_commit must be a lowercase 40-character Git SHA")
    if not isinstance(config["model_hashes"], dict) or not config["model_hashes"]:
        raise ValueError("model_hashes must be a non-empty object")
    for filename, digest in config["model_hashes"].items():
        if Path(filename).name != filename:
            raise ValueError(f"model_hashes contains an unsafe filename: {filename}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"Invalid SHA-256 for model file: {filename}")

    prompts = config["prompts"]
    if (
        not isinstance(prompts, list)
        or not prompts
        or any(not isinstance(prompt, str) or not prompt for prompt in prompts)
    ):
        raise ValueError("prompts must be a non-empty list of non-empty strings")
    if not isinstance(config["use_tqdm"], bool):
        raise ValueError("use_tqdm must be a boolean")

    engine = config["engine"]
    if not isinstance(engine["enforce_eager"], bool):
        raise ValueError("engine.enforce_eager must be a boolean")
    for field in (
        "tensor_parallel_size",
        "max_model_len",
        "max_num_batched_tokens",
        "max_num_seqs",
        "kvcache_block_size",
    ):
        if (
            not isinstance(engine[field], int)
            or isinstance(engine[field], bool)
            or engine[field] <= 0
        ):
            raise ValueError(f"engine.{field} must be a positive integer")
    gpu_utilization = engine["gpu_memory_utilization"]
    if (
        not isinstance(gpu_utilization, (int, float))
        or isinstance(gpu_utilization, bool)
        or not 0 < gpu_utilization <= 1
    ):
        raise ValueError("engine.gpu_memory_utilization must be in (0, 1]")

    sampling = config["sampling"]
    temperature = sampling["temperature"]
    if (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or temperature <= 0
    ):
        raise ValueError("sampling.temperature must be positive")
    if (
        not isinstance(sampling["max_tokens"], int)
        or isinstance(sampling["max_tokens"], bool)
        or sampling["max_tokens"] <= 0
    ):
        raise ValueError("sampling.max_tokens must be a positive integer")
    if not isinstance(sampling["ignore_eos"], bool):
        raise ValueError("sampling.ignore_eos must be a boolean")

    resolve_output_path(config["output_path"])
    return config


def validate_exact_keys(
    name: str,
    value: Any,
    expected: set[str],
) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    missing = sorted(expected - value.keys())
    unknown = sorted(value.keys() - expected)
    if missing or unknown:
        raise ValueError(
            f"{name} keys mismatch; missing={missing}, unknown={unknown}"
        )


def resolve_output_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise ValueError("output_path must be relative to the repository")
    resolved = (REPO_ROOT / path).resolve()
    if not resolved.is_relative_to(RESULTS_DIR.resolve()):
        raise ValueError("output_path must be inside reproduction/results")
    if resolved.suffix != ".json":
        raise ValueError("output_path must end in .json")
    return resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model_files(
    model_path: Path,
    expected_hashes: dict[str, str],
) -> dict[str, str]:
    actual_hashes = {}
    for filename, expected_hash in expected_hashes.items():
        path = model_path / filename
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        actual_hash = file_sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Model hash mismatch for {filename}: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        actual_hashes[filename] = actual_hash
    return actual_hashes


def repository_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def git_status() -> list[str]:
    output = subprocess.check_output(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        text=True,
    )
    return output.splitlines()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as file:
        temporary_path = Path(file.name)
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())
    temporary_path.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "reproduction/configs/smoke_eager.json",
    )
    parser.add_argument("--model-path", type=Path)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path, model_path_override=args.model_path)
    verify_source_snapshot()
    model_path = Path(config["model_path"])
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")
    model_hashes = verify_model_files(model_path, config["model_hashes"])

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in config["prompts"]
    ]
    sampling_params = SamplingParams(**config["sampling"])

    llm = None
    try:
        init_started = time.perf_counter()
        llm = LLM(str(model_path), **config["engine"])
        initialization_seconds = time.perf_counter() - init_started

        generation_started = time.perf_counter()
        outputs = llm.generate(
            prompts,
            sampling_params,
            use_tqdm=config["use_tqdm"],
        )
        generation_seconds = time.perf_counter() - generation_started
    finally:
        if llm is not None:
            atexit.unregister(llm.exit)
            llm.exit()
    if len(outputs) != len(prompts):
        raise RuntimeError(
            f"Expected {len(prompts)} outputs, received {len(outputs)}"
        )
    if any(not output["token_ids"] for output in outputs):
        raise RuntimeError("Smoke test produced an empty completion")

    result = {
        "status": "passed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "upstream_commit": config["upstream_commit"],
        "legacy_release_commit": LEGACY_RELEASE_COMMIT,
        "repository_head": repository_head(),
        "repository_status": git_status(),
        "config_path": str(config_path),
        "harness": {
            "run_smoke.py": file_sha256(Path(__file__)),
            str(config_path.relative_to(REPO_ROOT)): file_sha256(config_path),
            str(REQUIREMENTS_PATH.relative_to(REPO_ROOT)): file_sha256(
                REQUIREMENTS_PATH
            ),
            str(PYLOCK_PATH.relative_to(REPO_ROOT)): file_sha256(PYLOCK_PATH),
        },
        "model": {
            "repo_id": config["model_repo_id"],
            "revision": config["model_revision"],
            "path": str(model_path),
            "sha256": model_hashes,
        },
        "engine": config["engine"],
        "sampling": config["sampling"],
        "runtime": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "triton": triton.__version__,
            "flash_attn": flash_attn.__version__,
        },
        "timing": {
            "initialization_seconds": initialization_seconds,
            "generation_seconds": generation_seconds,
        },
        "outputs": [
            {
                "prompt": source_prompt,
                "text": output["text"],
                "generated_tokens": len(output["token_ids"]),
            }
            for source_prompt, output in zip(config["prompts"], outputs, strict=True)
        ],
    }

    output_path = resolve_output_path(config["output_path"])
    write_json_atomic(output_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
