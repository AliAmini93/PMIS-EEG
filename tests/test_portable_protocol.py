import json

import numpy as np
import pytest

from data_preparation import dreyer_sources, feasibility
from evaluation import r12x_analysis
from experiments import runtime


def test_reject_invalid_workspace_input(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "RUN_ROOT", tmp_path)
    (tmp_path / "inputs.json").write_text(json.dumps({"lee2019_mi": []}))
    with pytest.raises(ValueError, match="local path"):
        runtime.input_path("lee2019_mi")
    (tmp_path / "inputs.json").write_text(json.dumps({"lee2019_mi": "absent"}))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        runtime.input_path("lee2019_mi")


def test_reject_protocol_outside_inventory(tmp_path):
    with pytest.raises(ValueError, match="distributed"):
        runtime.load_protocol(tmp_path / "R12X_PROTOCOL.json")


def test_local_receipt_restores_only_historical_digests(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "RUN_ROOT", tmp_path)
    runtime.reference_receipts.cache_clear()
    name = "G3_PROTOCOL_FROZEN.json"
    key = "plan_commitment_sha256"
    path = runtime.CONFIG_ROOT / name
    artifact = tmp_path / "generated.json"
    artifact.write_text("{}")
    digest = runtime.file_sha256(artifact)
    try:
        assert runtime.load_protocol(path)["commitments"][key] is None
        assert runtime.verify_recorded_input(artifact, None)
        with pytest.raises(FileNotFoundError, match="Missing workflow input"):
            runtime.verify_recorded_input(tmp_path / "missing.json", None)

        receipt = {"protocols": {name: {f"commitments/{key}": digest, "status": digest}}}
        (tmp_path / "reference_receipts.json").write_text(json.dumps(receipt))
        runtime.reference_receipts.cache_clear()
        protocol = runtime.load_protocol(path)
        assert protocol["commitments"][key] == digest and protocol["status"] == "FROZEN"
        assert runtime.verify_recorded_input(artifact, digest)
        artifact.write_text('{"modified": true}')
        assert not runtime.verify_recorded_input(artifact, digest)
    finally:
        runtime.reference_receipts.cache_clear()


def test_fitting_only_reader_refuses_holdout_before_opening(tmp_path):
    for run in ("R5", "R6"):
        with pytest.raises(feasibility.OutcomeRunRefused):
            feasibility.gdf_path(tmp_path, 2, run)
    assert feasibility.gdf_path(tmp_path, 2, "R3") == tmp_path / dreyer_sources.member_path(2, "R3")


def test_corrected_e2_tolerance_applies_to_means_in_every_resample():
    # Model 2 wins by 0.006 points, within the mean-scale tie tolerance.
    # The historically implemented sum scale resolves the same gap differently.
    clean = np.tile([70.0, 70.006, 60.0], (83, 1))
    contam = np.tile([70.0, 72.0, 60.0], (83, 1))
    evaluate = np.tile([81.0, 78.0, 70.0], (83, 1))
    corrected = r12x_analysis.bootstrap(clean, contam, evaluate, "E2", n_boot=20)
    archived = r12x_analysis.bootstrap(clean, contam, evaluate, "E2", n_boot=20, legacy_e2_sums=True)
    np.testing.assert_array_equal(corrected["dist"], np.full(20, 3.0))
    np.testing.assert_array_equal(archived["dist"], np.zeros(20))
    assert r12x_analysis.e2(clean, contam, evaluate)["regret"] == 3.0
    assert corrected["p_any_flip"] == 1.0 and archived["p_any_flip"] == 0.0


def test_final_epoch_only_training_matches_with_validation():
    import torch
    from models.deep_group import EEGNetG, train_generic

    gen = np.random.default_rng(19)
    n_trial = 44
    x = torch.as_tensor(gen.standard_normal((n_trial, 4, 128)), dtype=torch.float32)
    y = torch.as_tensor(np.arange(n_trial) % 2)
    rows = np.tile(np.arange(32), (2, 1))
    args = (EEGNetG, x, y, rows, np.zeros(2, dtype=int), [13], 7, torch.device("cpu"), np.arange(36, 44))
    with_validation = train_generic(*args, np.arange(32, 36), epochs=2)
    final_only = train_generic(*args, None, epochs=2)
    np.testing.assert_array_equal(with_validation["p_final"], final_only["p_final"])
    np.testing.assert_array_equal(final_only["p_final"][0], final_only["p_final"][1])
    assert set(final_only) == {"p_final"}
