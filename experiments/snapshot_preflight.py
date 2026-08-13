from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST_PATH = REPO_ROOT / "docs/SOURCE_SNAPSHOT.sha256"
LEGACY_UPSTREAM_COMMIT = "bb823b3e06983d71485a8e1f23715ebd87d98ef8"
MODEL_REPO_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
MODEL_HASHES = {
    "model.safetensors": (
        "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e"
        "382306f42996874b"
    ),
    "config.json": (
        "660db3b73d788119c04535e48cf9be5f55bc3100841a718637"
        "ae695b442f27dd"
    ),
    "tokenizer.json": (
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe"
        "794b4492dae4"
    ),
}
_DIGEST = re.compile(r"[0-9a-f]{64}")
_PROTECTED_DIRECTORIES = (
    "nanovllm",
    "experiments",
    "reproduction",
    "tests",
    "scripts",
)
_PROTECTED_ROOT_FILES = ("bench.py", "example.py", "pyproject.toml")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protected_source_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for directory_name in _PROTECTED_DIRECTORIES:
        directory = REPO_ROOT / directory_name
        for path in directory.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(REPO_ROOT).as_posix()
            if relative.startswith(
                ("experiments/results/", "reproduction/results/")
            ):
                continue
            paths[relative] = path
    for filename in _PROTECTED_ROOT_FILES:
        path = REPO_ROOT / filename
        if path.is_file():
            paths[filename] = path
    return paths


def parse_source_manifest() -> dict[str, str]:
    if not SOURCE_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Missing source manifest: {SOURCE_MANIFEST_PATH}"
        )
    entries: dict[str, str] = {}
    root = REPO_ROOT.resolve()
    for line_number, raw_line in enumerate(
        SOURCE_MANIFEST_PATH.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as error:
            raise ValueError(
                f"Invalid source manifest line {line_number}: {raw_line}"
            ) from error
        pure_path = PurePosixPath(relative)
        if (
            not _DIGEST.fullmatch(digest)
            or pure_path.is_absolute()
            or ".." in pure_path.parts
            or relative in entries
        ):
            raise ValueError(
                f"Unsafe source manifest line {line_number}: {raw_line}"
            )
        resolved = (REPO_ROOT / pure_path).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(
                f"Source manifest path escapes repository: {relative}"
            )
        entries[relative] = digest
    return entries


def verify_source_snapshot() -> dict[str, Any]:
    entries = parse_source_manifest()
    protected = protected_source_paths()
    if set(entries) != set(protected):
        missing = sorted(set(protected) - set(entries))
        extra = sorted(set(entries) - set(protected))
        raise RuntimeError(
            "Source manifest coverage differs: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    for relative, expected in entries.items():
        actual = file_sha256(protected[relative])
        if actual != expected:
            raise RuntimeError(
                f"Snapshot hash mismatch for {relative}: "
                f"expected {expected}, got {actual}"
            )
    return {
        "manifest_path": SOURCE_MANIFEST_PATH.relative_to(REPO_ROOT).as_posix(),
        "manifest_sha256": file_sha256(SOURCE_MANIFEST_PATH),
        "verified_files": len(entries),
    }


def model_contract(model_path: Path) -> dict[str, Any]:
    return {
        "model_path": str(model_path.resolve()),
        "model_repo_id": MODEL_REPO_ID,
        "model_revision": MODEL_REVISION,
        "model_hashes": dict(MODEL_HASHES),
    }


def verify_model_identity(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("model_repo_id") != MODEL_REPO_ID:
        raise ValueError("model_repo_id differs from the frozen model contract")
    if config.get("model_revision") != MODEL_REVISION:
        raise ValueError("model_revision differs from the frozen model contract")
    if config.get("model_hashes") != MODEL_HASHES:
        raise ValueError("model_hashes differ from the frozen model contract")
    model_path = Path(config.get("model_path", ""))
    if not model_path.is_absolute() or not model_path.is_dir():
        raise FileNotFoundError(
            f"Model directory is missing or not absolute: {model_path}"
        )
    actual: dict[str, str] = {}
    for filename, expected in MODEL_HASHES.items():
        path = model_path / filename
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        digest = file_sha256(path)
        if digest != expected:
            raise RuntimeError(
                f"Model hash mismatch for {filename}: "
                f"expected {expected}, got {digest}"
            )
        actual[filename] = digest
    return {
        "path": str(model_path.resolve()),
        "repo_id": MODEL_REPO_ID,
        "revision": MODEL_REVISION,
        "sha256": actual,
    }
