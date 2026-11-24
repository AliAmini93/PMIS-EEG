"""R12-X resource safety and monitoring (Amendment 1, owner GPU policy of 2026-11-07).

Replaces the G1 GpuBudget (6-GPU-h ledger + SIGALRM unit stop), the 900-s unit cap, the maximum wait and the fixed
7.5-GiB per-process GPU fraction. There is NO time limit of any kind:
  * exclusive GPU use is coordinated machine-wide by the external GPU scheduler (FIFO queue + flock lease, no timeout);
  * before every unit, wait_safe() polls (default every 45 s, no timeout) until free GPU memory >= the measured
    need, host MemAvailable >= 8 GiB and memory-pressure stall (PSI full avg10) <= 5 -- availability checks, not caps;
  * UnitMeter records wall time, GPU peak allocated / reserved bytes and host RSS for reporting only.
Legitimate stops remain: a genuine OOM (identical recompute after waiting for safe availability fails
OOM_RETRIES times), hardware / CUDA errors, corrupt inputs, failed scientific controls (A/A, firewall, hashes).
"""

from __future__ import annotations

import os
import time

OOM_RETRIES = 3
MIN_MEM_AVAILABLE = 8 * 2**30
MAX_PSI_FULL_AVG10 = 5.0
POLL_S = 45.0


class ResourceStop(RuntimeError):
    pass


def host_state() -> dict:
    m = {ln.split(":")[0]: int(ln.split()[1]) * 1024 for ln in open("/proc/meminfo")
         if ln.split(":")[0] in ("MemAvailable", "SwapFree", "SwapTotal")}
    try:
        full = open("/proc/pressure/memory").read().split("\n")[1]
        m["psi_full_avg10"] = float(full.split()[1].split("=")[1])
    except (OSError, IndexError, ValueError):
        m["psi_full_avg10"] = 0.0
    return m


def rss_bytes() -> int:
    with open(f"/proc/{os.getpid()}/statm") as fh:
        return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def wait_safe(torch=None, need_free_bytes: int = 0, log=None, poll_s: float = POLL_S, sleep=time.sleep,
              state=host_state, status_every: int = 10) -> float:
    """Block (no timeout) until resources are safely available; returns the seconds waited."""
    t0, n = time.time(), 0
    while True:
        h = state()
        gpu_free = torch.cuda.mem_get_info()[0] if torch is not None else need_free_bytes
        ok = gpu_free >= need_free_bytes and h["MemAvailable"] >= MIN_MEM_AVAILABLE and h["psi_full_avg10"] <= MAX_PSI_FULL_AVG10
        if ok:
            return time.time() - t0
        if log and n % status_every == 0:
            log(f"waiting for safe resources: gpu_free {gpu_free} need {need_free_bytes} mem_available "
                f"{h['MemAvailable']} psi {h['psi_full_avg10']}")
        n += 1
        sleep(poll_s)


class UnitMeter:
    """Context manager measuring one unit (reporting only; never stops anything)."""

    def __init__(self, torch=None):
        self.torch, self.rec = torch, {}

    def __enter__(self):
        if self.torch is not None:
            self.torch.cuda.reset_peak_memory_stats()
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.rec = {"wall_s": time.time() - self.t0, "host_rss_bytes": rss_bytes()}
        if self.torch is not None:
            self.rec["gpu_peak_allocated_bytes"] = int(self.torch.cuda.max_memory_allocated())
            self.rec["gpu_peak_reserved_bytes"] = int(self.torch.cuda.max_memory_reserved())
        return False


def with_oom_retry(fn, is_oom, recover, log=None, retries: int = OOM_RETRIES):
    """Run fn(); on an OOM error call recover() (empty cache + wait for safe availability) and recompute
    identically. A persistent OOM after `retries` attempts is a genuine OOM: ResourceStop."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if not is_oom(e):
                raise
            if log:
                log(f"OOM on attempt {attempt + 1}: identical recompute after waiting for safe availability")
            recover()
    raise ResourceStop(f"genuine out-of-memory: {retries} identical attempts failed")
