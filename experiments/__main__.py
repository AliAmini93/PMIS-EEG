"""Command line for the recorded PMIS-EEG protocols."""

from __future__ import annotations

import argparse
import os
import runpy
import sys

COMMANDS = {
    "build-hs-stroke": ("data_preparation.hs_stroke", "HS-Stroke releases 1 and 3, features and folds"),
    "build-shin2022": ("data_preparation.shin2022", "Shin2022 run identity and feedback windows"),
    "build-lee2019": ("data_preparation.r6_c3_build_lee2019", "Lee2019 original and coarse-window features"),
    "build-lee-shifts": ("data_preparation.r7_c5_build_shifts", "Lee2019 fine shifts"),
    "build-lee-sliding": ("data_preparation.r10_b_build_windows", "Lee2019 sliding windows"),
    "build-lee-cbramod": ("data_preparation.r8_e_build_lee2019_cbramod", "CBraMod motor montage features"),
    "build-lee-pretraining-format": ("data_preparation.r9_b_build_lee2019_tueglike", "CBraMod pretraining-format features"),
    "audit-dreyer": ("data_preparation.r12x_g1r_cohort", "Dreyer2023 recording and eligibility checks"),
    "build-dreyer": ("data_preparation.r12x_build_features", "Dreyer2023 EEG and control features"),
    "validate-dreyer": ("data_preparation.r12x_g1r_feature_validation", "Dreyer2023 feature integrity and split isolation"),
    "hs-swap-tslr": ("experiments.run_g3", "HS-Stroke TS+LR identity swap"),
    "hs-newton": ("experiments.r6_c1_run", "One-step approximation and copy doses"),
    "hs-swap-eegnet": ("experiments.r6_c2_run", "HS-Stroke EEGNet identity swap"),
    "hs-karcher-audit": ("experiments.r7_c1b_run", "Independent reference-mean audit"),
    "shin-swap": ("experiments.r8_a_run", "Shin2022 identity swap"),
    "lee-swap-tslr": ("experiments.r6_c3a_run", "Lee2019 TS+LR identity swap"),
    "lee-swap-eegnet": ("experiments.r6_c3b_run", "Lee2019 EEGNet identity swap"),
    "lee-coarse-tslr": ("experiments.r6_c4_run", "Lee2019 coarse temporal shifts, TS+LR"),
    "lee-coarse-eegnet": ("experiments.r6_c4b_run", "Lee2019 coarse temporal shifts, EEGNet"),
    "lee-fine-shifts": ("experiments.r7_c5_run", "Lee2019 fine temporal shifts"),
    "lee-convnet-comparison": ("experiments.r7_c6_run", "Lee2019 validation-split decoder comparison"),
    "lee-transformers": ("experiments.r8_de_run", "Lee2019 TCFormer and CBraMod"),
    "lee-cbramod-validity": ("experiments.r9_b_run", "CBraMod input and pretraining controls"),
    "hs-legacy-reference": ("experiments.hs_legacy_reference", "Generate the earlier HS-Stroke benchmark reference"),
    "hs-benchmark-tslr": ("experiments.r7_c7_run", "HS-Stroke TS+LR benchmark"),
    "hs-benchmark-eegnet": ("experiments.r8_b_run", "HS-Stroke EEGNet benchmark"),
    "hs-split-eegnet": ("experiments.r9_c_run", "HS-Stroke EEGNet split-half decomposition"),
    "hs-split-tslr": ("experiments.r9_c2_run", "HS-Stroke TS+LR split-half decomposition"),
    "hs-ranking": ("experiments.r10_a_run", "HS-Stroke decoder ranking"),
    "lee-sliding": ("experiments.r10_b_run", "Lee2019 sliding-window forecast"),
    "lee-sliding-audit": ("evaluation.r11_a_c0_erratum_check", "Verify the 500-ms-stride overlap correction"),
    "lee-dose-priors": ("evaluation.r11_b_g0_existing", "Recompute the single-copy inputs to the dose forecast"),
    "lee-doses": ("experiments.r11_b_run", "Lee2019 controlled doses and ranking forecast"),
    "ceiling-audit": ("experiments.r8_c_run", "Baseline dependence across identity-swap experiments"),
    "dreyer-calibration": ("experiments.r12x_g1r_calibration", "Pre-outcome synthetic inference calibration"),
    "dreyer-selection": ("experiments.r12x_run", "Dreyer2023 held-out-run selection and controls"),
    "dreyer-descriptives": ("evaluation.r12x_descriptives", "Dreyer2023 paired and seed-level descriptives"),
    "dreyer-audit": ("evaluation.r12x_audit_recompute", "Separate recomputation from Dreyer2023 saved fits"),
    "dreyer-tie-sensitivity": ("evaluation.r17p1_e2_tie_sensitivity", "Secondary mean-scale tie-rule correction"),
    "build-feasibility": ("data_preparation.r15b_build_cache", "Dreyer2023 fitting-run features for the feasibility screen"),
    "feasibility": ("experiments.r15b_feasibility", "ST-EEGFormer, LaBraM and CBraMod feasibility screen"),
    "simulate": ("experiments.r15c0_sims", "Post hoc synthetic operating-characteristics study"),
    "summarize-simulation": ("evaluation.simulation_summary", "Extract the reported synthetic study summaries"),
}


def main() -> None:
    width = max(map(len, COMMANDS))
    listing = "\n".join(f"  {name:<{width}}  {label}" for name, (_, label) in COMMANDS.items())
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Commands\n{listing}",
    )
    parser.add_argument("--work-dir", help="workspace holding inputs.json and generated data")
    parser.add_argument("--cuda-device", metavar="N", type=int,
                        help="select CUDA device N within the devices visible to this process")
    parser.add_argument("command", metavar="COMMAND", choices=COMMANDS)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.work_dir:
        os.environ["PMIS_WORK_DIR"] = args.work_dir
    if args.cuda_device is not None:
        if args.cuda_device < 0:
            parser.error("--cuda-device must be nonnegative")
        os.environ["PMIS_CUDA_DEVICE"] = str(args.cuda_device)
    module, _ = COMMANDS[args.command]
    help_requested = any(arg in {"-h", "--help"} for arg in args.arguments)
    if not help_requested:
        from .runtime import RUN_ROOT, verify_package

        try:
            verify_package()
            (RUN_ROOT / "results").mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
            for var, name in (("NUMBA_CACHE_DIR", "numba"), ("MPLCONFIGDIR", "matplotlib")):
                cache = RUN_ROOT / "cache" / name
                cache.mkdir(parents=True, exist_ok=True)
                os.environ.setdefault(var, str(cache))
        except (OSError, ValueError, RuntimeError) as exc:
            parser.exit(2, f"PMIS-EEG: {exc}\n")
    sys.argv = [f"python -m experiments {args.command}", *args.arguments]
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except (OSError, ValueError, ImportError, RuntimeError) as exc:
        parser.exit(2, f"PMIS-EEG: {exc}\n")


if __name__ == "__main__":
    main()
