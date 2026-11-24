"""Local source geometry for the producer's Dreyer2023 GDF release."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path

from experiments.runtime import RUN_ROOT, input_path

RECORD_MD5 = "c8c6f5f09f1882666eec10e29155fdcb"
RUNS = ("R1", "R2", "R3", "R4", "R5", "R6")
GDF_TAG = {
    "R1": "R1_acquisition", "R2": "R2_acquisition", "R3": "R3_onlineT",
    "R4": "R4_onlineT", "R5": "R5_onlineT", "R6": "R6_onlineT",
}
GROUPS = {"A": range(1, 61), "B": range(61, 82), "C": range(82, 88)}


def group_of(n: int) -> str:
    for group, participants in GROUPS.items():
        if n in participants:
            return group
    raise ValueError("Participant index is outside the producer release")


def producer_id(n: int) -> str:
    return f"{group_of(n)}{n}"


def member_path(n: int, run: str) -> Path:
    pid = producer_id(n)
    return Path("BCI Database") / "Signals" / f"DATA {pid[0]}" / pid / f"{pid}_{GDF_TAG[run]}.gdf"


def raw_root() -> Path:
    source = input_path("dreyer2023_raw")
    if source.is_dir():
        if not (source / "BCI Database" / "Signals").is_dir():
            raise ValueError("Dreyer2023 root must contain the producer's BCI Database/Signals tree")
        return source
    return RUN_ROOT / "data" / "dreyer_producer"


def index() -> tuple[Path, dict]:
    """Index or prepare only the six documented run files. Other member types stay untouched."""
    source = input_path("dreyer2023_raw")
    root = raw_root()
    paths = {n: {r: member_path(n, r) for r in RUNS} for ns in GROUPS.values() for n in ns}
    if source.is_dir():
        return root, {n: {r: (root / f).stat().st_size for r, f in runs.items()}
                      for n, runs in paths.items() if all((root / f).is_file() for f in runs.values())}

    with source.open("rb") as f:
        if hashlib.file_digest(f, "md5").hexdigest() != RECORD_MD5:
            raise ValueError("Dreyer2023 archive differs from the Zenodo 8089820 checksum")
    result = {}
    with zipfile.ZipFile(source) as archive:
        entries = {i.filename: i for i in archive.infolist()}
        for n, runs in paths.items():
            if not all(f.as_posix() in entries for f in runs.values()):
                continue
            result[n] = {}
            for r, f in runs.items():
                info = entries[f.as_posix()]
                out = root / f
                out.parent.mkdir(parents=True, exist_ok=True)
                part = out.with_suffix(".gdf.part")
                with archive.open(info) as orig, part.open("wb") as dest:
                    shutil.copyfileobj(orig, dest, length=2**20)
                part.replace(out)
                result[n][r] = info.file_size
    return root, result
