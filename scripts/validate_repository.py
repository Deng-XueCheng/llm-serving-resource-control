from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.snapshot_preflight import verify_source_snapshot


MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
DIGEST = re.compile(r"[0-9a-f]{64}")
SECRET_PATTERNS = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
DEVELOPMENT_PATHS = (
    "E:\\forajob",
    "/root/autodl",
    "/mnt/e/forajob",
    "/opt/models",
)
REQUIRED_PATHS = (
    "README.md",
    "ARCHITECTURE.md",
    "NOTICE.md",
    "docs/FINAL_RESULTS.md",
    "docs/PROVENANCE.md",
    "docs/REPRODUCIBILITY.md",
    "experiments/results/final/MANIFEST.sha256",
    "experiments/results/final/multi_replica/stage18/evidence_review.json",
    "experiments/results/final/characterization/p0_1/evidence_review.json",
    "experiments/results/final/characterization/p0_2/evidence_review.json",
    "experiments/results/final/pd/stage19/pd.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(
    path: Path,
    *,
    expected_root: Path | None = None,
) -> list[str]:
    errors: list[str] = []
    if not path.is_file():
        return [f"missing manifest: {path.relative_to(ROOT)}"]
    seen: set[str] = set()
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"invalid manifest line {path}:{line_number}")
            continue
        pure_path = PurePosixPath(relative)
        if (
            not DIGEST.fullmatch(expected)
            or pure_path.is_absolute()
            or ".." in pure_path.parts
            or relative in seen
        ):
            errors.append(f"unsafe manifest line {path}:{line_number}")
            continue
        seen.add(relative)
        target = (ROOT / pure_path).resolve()
        if not target.is_relative_to(ROOT.resolve()):
            errors.append(f"manifest path escapes repository: {relative}")
            continue
        if not target.is_file():
            errors.append(f"manifest target missing: {relative}")
        elif sha256(target) != expected:
            errors.append(f"manifest hash mismatch: {relative}")
    if expected_root is not None:
        actual = {
            item.relative_to(ROOT).as_posix()
            for item in expected_root.rglob("*")
            if item.is_file() and item.resolve() != path.resolve()
        }
        if seen != actual:
            errors.append(
                "manifest coverage differs: "
                f"missing={sorted(actual - seen)[:5]}, "
                f"extra={sorted(seen - actual)[:5]}"
            )
    return errors


def validate_markdown_links() -> list[str]:
    errors: list[str] = []
    for markdown in ROOT.rglob("*.md"):
        text = markdown.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().strip("<>")
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            path_part = unquote(target.split("#", 1)[0])
            if not path_part:
                continue
            resolved = (markdown.parent / path_part).resolve()
            if not resolved.exists():
                errors.append(
                    f"broken markdown link: {markdown.relative_to(ROOT)} -> {target}"
                )
    return errors


def validate_secrets_and_paths() -> list[str]:
    errors: list[str] = []
    text_suffixes = {".md", ".py", ".json", ".jsonl", ".toml", ".txt", ".csv"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.stat().st_size > 100 * 1024 * 1024:
            errors.append(f"file exceeds 100 MiB: {path.relative_to(ROOT)}")
        if path.suffix.lower() not in text_suffixes:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            errors.append(f"secret-like content: {path.relative_to(ROOT)}")
        if path.suffix.lower() == ".md" and any(
            marker in text for marker in DEVELOPMENT_PATHS
        ):
            errors.append(f"development absolute path in docs: {path.relative_to(ROOT)}")
    return errors


def main() -> int:
    errors: list[str] = []
    for relative in REQUIRED_PATHS:
        if not (ROOT / relative).exists():
            errors.append(f"required path missing: {relative}")

    forbidden = (
        ROOT / "experiments/results/historical_invalid",
        ROOT / "experiments/results/pre_gpu_binding_fix",
    )
    for path in forbidden:
        if path.exists():
            errors.append(f"invalid evidence directory present: {path.relative_to(ROOT)}")

    errors.extend(validate_markdown_links())
    errors.extend(validate_secrets_and_paths())
    try:
        verify_source_snapshot()
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        errors.append(f"source snapshot validation failed: {error}")
    errors.extend(
        validate_manifest(
            ROOT / "experiments/results/final/MANIFEST.sha256",
            expected_root=ROOT / "experiments/results/final",
        )
    )

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        print(f"VALIDATION_FAILED: {len(errors)} issue(s)")
        return 1
    print("MARKDOWN_LINKS_OK")
    print("FINAL_EVIDENCE_PATHS_OK")
    print("MANIFESTS_OK")
    print("SECRET_SCAN_OK")
    print("ABSOLUTE_DOC_PATH_SCAN_OK")
    print("LARGE_FILE_GATE_OK")
    print("CLEAN_REPOSITORY_VALIDATION_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
