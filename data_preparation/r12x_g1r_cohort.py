"""Verify the source recordings and reconstruct the outcome-blind Dreyer2023 eligibility rules."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor

from experiments.runtime import reference_matches, CONFIG_ROOT, RUN_ROOT, load_protocol, json_sha256, write_json
from . import dreyer_sources as source
from .r12x_g1r_check import check_participant

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "1"


def check_job(args):
    n, root, sizes = args
    return n, check_participant(root, n, sizes)


def duplicates(checks, field):
    owners = collections.defaultdict(list)
    for n, participant in checks.items():
        for run in participant["runs"]:
            items = [run[field]] if field == "sha256" else run[field]
            for item in items:
                owners[item].append((n, run["run"]))
    return {digest: runs for digest, runs in owners.items() if len(runs) > 1}


def cohort_receipt(checks: dict, proto: dict) -> tuple[dict, dict]:
    file_repeats = duplicates(checks, "sha256")
    full_window_repeats = duplicates(checks, "window_hashes")
    cue_repeats = duplicates(checks, "cue_window_hashes")
    involved = {n for v in (*file_repeats.values(), *full_window_repeats.values()) for n, _ in v}
    extreme = {n for n, status in checks.items() if any(r["extreme_calibration_header"] for r in status["runs"])}
    ratios = {n: status["scaling_ratio"] for n, status in checks.items()}
    reference = [ratio for n, ratio in ratios.items() if n not in extreme]
    lo, hi = min(reference), max(reference)
    revised = sorted(n for n, c in checks.items() if n not in involved
                     and not (set(c["fails"]) - {"G7"}) and (n not in extreme or lo <= ratios[n] <= hi))
    declared = sorted(n for n, c in checks.items() if n not in involved and not c["fails"])
    table = {f"{source.producer_id(n)}_{r['run']}": r["sha256"] for n in revised for r in checks[n]["runs"]}
    membership = proto["sensitivity_membership"]
    revised_sha = hashlib.sha256(json.dumps(revised).encode()).hexdigest()
    declared_sha = hashlib.sha256(json.dumps(declared).encode()).hexdigest()
    table_sha = json_sha256(table)
    if (len(revised) != proto["n_participants"] or not reference_matches(revised_sha, proto["participants_sha256"]) or not reference_matches(table_sha, proto["cohort"]["raw_sha256_table_sha256"])
            or len(declared) != membership["count"] or not reference_matches(declared_sha, membership["sha256"])):
        raise ValueError("Audited eligibility or raw file bytes differ from the frozen Dreyer2023 protocol")
    if {n for v in cue_repeats.values() for n, _ in v} - involved:
        raise ValueError("A scored cue-window identity occurs outside the frozen exclusions")
    runs = [r for check in checks.values() for r in check["runs"]]
    public = {
        "participants_file_eligible": len(checks),
        "eligible_participants": len(revised),
        "eligible_by_group": dict(collections.Counter(source.group_of(n) for n in revised)),
        "runs_checked": len(runs),
        "events_equal_runs": sum(bool(r["G2_events_equal"]) for r in runs),
        "max_rel_diff_own_vs_mne": max(r["G2_max_rel_diff"] for r in runs),
        "min_cue_spacing_s": min(r["G5_min_spacing_s"] for r in runs),
        "cross_participant_duplicate_windows": sum(len({n for n, _ in v}) > 1 for v in cue_repeats.values()),
        "duplicate_files": len(file_repeats),
        "runs_without_end_event": sum(not r["has_800"] for r in runs),
        "sensitivity_participants": len(declared),
        "eligible_set_sha256": revised_sha,
        "raw_sha256_table_sha256": table_sha,
    }
    return {"eligible": revised, "sensitivity_G7_as_declared": declared, "raw_sha256": table}, public


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    args = ap.parse_args()
    proto = load_protocol(CONFIG_ROOT / "R12X_PROTOCOL.json")
    root, roster = source.index()
    if len(roster) != proto["cohort"]["eligible_by_files"]:
        raise ValueError("Expected all 86 file-eligible participants in the original Dreyer2023 release")
    jobs = [(n, root, sizes) for n, sizes in roster.items()]
    checks = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, check in pool.map(check_job, jobs):
            checks[n] = check
            if len(checks) % 10 == 0:
                print(f"{len(checks)}/{len(roster)} recording sets checked", flush=True)
    private, public = cohort_receipt(checks, proto)
    write_json(RUN_ROOT / "data" / "r12x" / "G1R_eligible_PRIVATE.json", private)
    write_json(RUN_ROOT / "results" / "R12X_G1R_cohort_validity.json", public)
    print(json.dumps(public))


if __name__ == "__main__":
    main()
