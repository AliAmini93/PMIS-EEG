"""Use a scheduler-assigned CUDA device and lock a run workspace."""

from __future__ import annotations

import contextlib
import fcntl
import os

from .runtime import RUN_ROOT


class AlreadyRunning(RuntimeError):
    """Another process holds the lock for this stage and workspace."""


@contextlib.contextmanager
def JobLock(job: str):
    locks = RUN_ROOT / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"{job}.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AlreadyRunning(f"{job} is already running in this workspace") from exc
        yield


def select_device():
    """Select the current process's CUDA device without altering visibility or other jobs."""
    import torch

    index = int(os.environ.get("PMIS_CUDA_DEVICE", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this recorded neural training stage")
    if not 0 <= index < torch.cuda.device_count():
        raise ValueError("Selected CUDA device is outside the visible device list")
    torch.cuda.set_device(index)
    return torch.device("cuda", index)
