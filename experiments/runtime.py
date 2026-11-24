"""Local workspace and integrity checks for the ported protocols."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PACKAGE_ROOT / "configs"
MANIFEST = CONFIG_ROOT / "integrity.json"
RUN_ROOT = Path(os.environ.get("PMIS_WORK_DIR", "runs")).expanduser().resolve()


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read {path.name}: {exc.strerror if isinstance(exc, OSError) else 'invalid JSON'}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


@lru_cache(maxsize=1)
def verify_package() -> dict:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("The recorded environment uses Python 3.11")
    manifest = read_json(MANIFEST)
    if manifest.get("schema") != 1 or not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise ValueError("Invalid package integrity manifest")
    for key, expected in manifest["files"].items():
        rel = Path(key)
        if rel.is_absolute() or ".." in rel.parts or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("Invalid file entry in package integrity manifest")
        if file_sha256(PACKAGE_ROOT / rel) != expected:
            raise ValueError(f"Portable package file differs from its recorded SHA-256: {rel}")
    return manifest


def port_identity() -> str:
    verify_package()
    return file_sha256(MANIFEST)


@lru_cache
def reference_receipts() -> dict:
    """Read optional, user-supplied historical digests from the local workspace."""
    path = RUN_ROOT / "reference_receipts.json"
    if not path.exists():
        return {}
    return read_json(path)


def reference_matches(actual: str | None, expected: str | None) -> bool:
    """Compare with a historical reference when one is supplied."""
    return expected is None or actual == expected


def load_protocol(path: Path) -> dict:
    inventory = verify_package()["files"]
    if path.parent != CONFIG_ROOT or f"configs/{path.name}" not in inventory:
        raise ValueError("Use a protocol distributed with the package")
    protocol = read_json(path)
    if protocol.get("status") != "FROZEN" or not isinstance(protocol.get("protocol_id"), str):
        raise ValueError(f"Invalid frozen protocol record: {path.name}")

    def restore(node, prefix=""):
        for key, value in node.items():
            field = prefix + key
            if isinstance(value, dict):
                restore(value, field + "/")
            elif value is None and field in references:
                digest = references[field]
                if (not isinstance(digest, str) or len(digest) != 64
                        or any(c not in "0123456789abcdef" for c in digest)):
                    raise ValueError(f"Invalid SHA-256 receipt for {path.name}, {field}")
                node[key] = digest

    references = reference_receipts().get("protocols", {}).get(path.name, {})
    restore(protocol)
    return protocol


def input_path(key: str) -> Path:
    """Resolve a user-supplied path from the runtime workspace."""
    cfg = read_json(RUN_ROOT / "inputs.json")
    value = cfg.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Set '{key}' to a local path in the workspace inputs.json")
    path = (RUN_ROOT / Path(value).expanduser()).resolve()
    if not path.exists():
        raise FileNotFoundError(f"The configured input '{key}' does not exist")
    return path


def write_json(path: Path, data) -> None:
    """Replace one completed JSON output without exposing a partial document."""
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".", delete=False
    ) as stream:
        tmp = Path(stream.name)
        try:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.close()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def npz_content_sha256(path: Path) -> str:
    """Hash ordered field names, shapes, dtypes and bytes independently of ZIP packaging."""
    import numpy as np

    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as arrays:
        for name in sorted(arrays.files):
            array = arrays[name]
            header = json.dumps([name, array.shape, array.dtype.str]).encode()
            digest.update(len(header).to_bytes(8, "big"))
            digest.update(header)
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def verify_recorded_input(path: Path, original: str | None) -> bool:
    """Require the input file and compare bytes or ordered arrays when a receipt is supplied."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing workflow input: {path.name}")
    if original is None:
        return True
    if file_sha256(path) == original:
        return True
    if path.suffix != ".npz":
        return False
    receipts = reference_receipts().get("array_receipts", {})
    expected = receipts.get(original)
    return expected is not None and expected == npz_content_sha256(path)
