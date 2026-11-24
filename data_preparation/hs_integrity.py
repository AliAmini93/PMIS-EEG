"""HS-Stroke file integrity and optional historical receipts."""

from experiments.runtime import file_sha256 as sha256_file, reference_receipts

# These two source files ship with the release and can be checked publicly.
FROZEN_R4_CODE_SHA256 = {
    "r4_models.py": "7b12bef9c672c0316a31ce94661e2f4c739cc20868b9e267f27f55560d89566f",
    "r4_stats.py": "8a9714e44e5c1bb26e356d5bca5d3d0a64c9e055d78a75dacb97683e9569760a",
}

FORBIDDEN_PRIVATE_FILES = ("G2_covs_8_30.npy", "G2_eegnet_4_40_128hz.npy", "G3_tslr_predictions_PRIVATE.npz")
METADATA_FILES = ("G2_instances.npz", "G2_distinct.npz", "G2_manifest.npz")


def historical_sha256(filename: str) -> str | None:
    return reference_receipts().get("hs_stroke", {}).get(filename)
