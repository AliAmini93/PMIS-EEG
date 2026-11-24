# Running the PMIS-EEG workflows

## Environment

Use Linux on a little-endian machine and Python 3.11. The recorded environment used Python 3.11.15, NumPy 1.26.4, SciPy 1.16.0, scikit-learn 1.6.1, PyTorch 2.9.1 with CUDA 12.8 on an RTX 5090, MNE 1.11.0, einops 0.8.1 and timm 1.0.22. Install the matching PyTorch and torchvision wheels if you intend to run the neural protocols. [PyTorch wheel instructions](https://pytorch.org/get-started/previous-versions/#v291)

```text
python -m pip install -r requirements.txt
python -m experiments --help
python -m pip install pytest==9.1.1
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q
```

Use `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` for the CPU experiments. The neural runners set deterministic Torch and cuBLAS options. For CUDA runs use a GPU assigned by your site's scheduler or lease mechanism. Restrict the job's visible devices with the scheduler, then use `--cuda-device 0` for the first visible device. The runners retain the campaign's free-memory checks, resource limits and A/A checks. Shards are zero-indexed, for example `--shard 0/2` and `--shard 1/2`. Keep only one process per shard and workspace. Finish all shards before analysis.

## Inputs

Obtain the original files under the terms of their producer records. The code works from local paths in `work/inputs.json`. Paths may be absolute or relative to the work directory. Only the keys needed by the current command are required.

```json
{
  "hs_stroke_v1": "../originals/hs-stroke-v1/sourcedata",
  "hs_stroke_v3": "../originals/hs-stroke-v3/sourcedata",
  "shin2022_zip": "../originals/study-2022-data-anonymized.zip",
  "lee2019_mi": "../originals/openbmi/mi",
  "dreyer2023_raw": "../originals/BCI Database.zip"
}
```

| Input | Original release and layout |
|---|---|
| HS-Stroke | Figshare 30747806 v1, `sourcedata.zip` file 60064148, MD5 `d5a6c27706356cc9f6f68eec22cd4821`. Extract the `sourcedata/sub-…/ses-…/eeg/*.mat` tree. Repeat with v3, file 66772196, MD5 `a079fa4a6f3073e0ebb92102ade23731`. The v1 rows retain the natural copies. The matched v3 trials identify their original acquisition runs. Both versions carry CC BY 4.0. [v1 record](https://api.figshare.com/v2/articles/30747806/versions/1), [v3 record](https://api.figshare.com/v2/articles/30747806/versions/3) |
| Shin2022 | Figshare 20383716 v1, `study-2022-data-anonymized.zip`, file 36438960, CC BY 4.0. Pass the original ZIP. The builder verifies the publisher's MD5, inspects the live BCI2000 recordings, reconstructs the recording-identity mask and builds the first 0.96 s of feedback with the recorded per-recording causal filters. [Figshare version record](https://api.figshare.com/v2/articles/20383716/versions/1) |
| Lee2019 / OpenBMI | GigaDB dataset 100542, DOI `10.5524/100542`, CC0 1.0. Place the 108 original motor-imagery MATLAB files named `sessNN_subjNN_EEG_MI.mat` in one directory. The builder checks the expected file layout and count. An optional historical receipt can also verify the file manifest. [GigaDB dataset](https://gigadb.org/dataset/100542), [DataCite rights record](https://api.datacite.org/dois/10.5524/100542) |
| Dreyer2023 | Producer GDF release, Zenodo record 8089820, DOI `10.5281/zenodo.8089820`, CC BY 4.0. Pass the producer ZIP or a directory containing `BCI Database/Signals/DATA {A,B,C}/<participant>/<participant>_<run>.gdf`. The ZIP is checked against the published MD5 and only the six-run EEG members are extracted. The corrected local reader is checked against MNE, and the eligibility and raw-file receipt are regenerated locally. [Zenodo record](https://zenodo.org/records/8089820) |

The HS-Stroke producers credit Zhiwei Deng for discovering the duplicates. BCI Report previously documented the Shin2022 recording duplicates.

Generated features, original measurements, pair tables, training checkpoints, detailed predictions and analysis files live under `work/data` and `work/results`. Treat both directories as local research data. Several analysis outputs contain participant-level diagnostics. The release tree contains source and configuration only.

### Weights

Place the authenticated checkpoint files in `work/weights`. The commands validate the published or pinned SHA-256 before loading. Refer to the upstream weight sources for their terms.

| Model | Local filename and SHA-256 |
|---|---|
| CBraMod | `pretrained_weights.pth`, `0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178`. The pinned model card declares Apache 2.0. [Hugging Face, revision 500543c](https://huggingface.co/weighting666/CBraMod/tree/500543c7e30bda1b22bfd51a49301b238dee21fd) |
| ST-EEGFormer-small | `checkpoint-300.pth`, `98d2848abdaea7a143d399c07f67f8782e07dd66e7c309ddb058c7158a8a8b02`. See the upstream release for the checkpoint and its terms. [Upstream release](https://github.com/LiuyinYang1101/STEEGFormer/releases/tag/ST-EEGFormer-small) |
| LaBraM-Base and VQ-NSP | `labram-base.pth`, `7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c`. `vqnsp.pth`, `d8f14b4232f06a5c37b6386ab021c270dd4bfa12bfecdd60372dc8ac7b711101`. See the original project for the checkpoints and any applicable terms. [Pinned upstream checkpoints](https://github.com/935963004/LaBraM/tree/c431221e6cfd23dbfa9950e0180682fb322b0548/checkpoints) |

## Run order

Prefix each command below with `python -m experiments --work-dir work`. Add `--cuda-device 0` before the command for neural execution on an allocated device. The filenames shown are under `work/results/` unless indicated. Execute dependency rows in order. A planning command checks reconstruction of the experimental design. An analysis command requires every outcome used by that analysis.

### HS-Stroke

| Reported analysis | Commands in order | Main output |
|---|---|---|
| Cohort, folds and pairs | `build-hs-stroke`, `hs-swap-tslr --plan-only` | Features and numbered fold manifest under `data/hs_stroke/` |
| Identity swap, placebo, neighbour and bystanders | `hs-swap-tslr --execute`, `hs-swap-eegnet --execute` | `G3_results.json`, `R6_C2_results.json` |
| One-step law, copy doses and independent Karcher audit | `hs-newton --execute`, `hs-karcher-audit --execute` | `R6_C1_results.json`, `R7_C1b_results.json` |
| TS+LR natural benchmark, earlier contrast reproduction and benchmark influence law | `hs-legacy-reference`, `hs-benchmark-tslr --execute`. Use `hs-benchmark-tslr --analyse` to recompute from completed fits | Runtime legacy reference, `R7_C7_results.json` |
| EEGNet benchmark | `hs-benchmark-eegnet --execute` | `R8_B_results.json` |
| Split-half decomposition | `hs-split-eegnet --execute`, `hs-split-eegnet --analyse`, `hs-split-tslr --execute` | `R9_C_results.json`, `R9_C2_results.json` |
| Ranking and apparent choice gap | `hs-ranking --execute --part SH1`, then `SH2` and `EG2`, then `hs-ranking --analyse` | `R10_A_results.json` |

### Shin2022 and Lee2019

| Reported analysis | Commands in order | Main output |
|---|---|---|
| Shin2022 natural identity swap | `build-shin2022`, `shin-swap --execute-tslr`, `shin-swap --execute-eegnet`, `shin-swap --analyse` | `R8_A_results.json` |
| Lee2019 input, TS+LR and EEGNet identity swap | `build-lee2019`, `lee-swap-tslr --execute`, `lee-swap-eegnet --execute` | `R6_C3a_results.json`, `R6_C3b_results.json` |
| EEGNet and ShallowConvNet on matched validation folds | `lee-convnet-comparison --execute` | `R7_C6_results.json` |
| TCFormer and CBraMod | `build-lee-cbramod`, `lee-transformers --execute --model TCFormer`, `lee-transformers --execute --model CBraMod`, `lee-transformers --execute-probe`, `lee-transformers --analyse` | `R8_DE_results.json` |
| CBraMod input format, pretraining engagement and random initialization | `build-lee-pretraining-format`, `lee-cbramod-validity --execute-recon`, then `--execute-probe --variant random20`, `--execute-probe --variant pre19`, `--execute-ft --variant random20`, `--analyse` | `R9_B1_results.json`, `R9_B_results.json` |
| Coarse temporal shifts | `lee-coarse-tslr --execute`, `lee-coarse-eegnet --execute` | `R6_C4_results.json`, `R6_C4b_results.json` |
| Fine temporal shifts | `build-lee-shifts`, `lee-fine-shifts --execute-tslr`, `lee-fine-shifts --execute-eegnet`, `lee-fine-shifts --analyse` | `R7_C5_results.json` |
| Realistic and fixed-density sliding windows | `build-lee-sliding`, `lee-sliding --execute --model TSLR`, `lee-sliding --execute --model EEGNet`, `lee-sliding --analyse`, `lee-sliding-audit` | `R10_B_results.json`, `R11_R10B_C0_erratum_check.json` |
| Controlled doses and ranking forecast | `lee-dose-priors` after the Lee2019 identity-swap and convolutional fits; `lee-doses --execute --model TSLR --rep R1`; run `lee-doses --execute` for each combination of `--model EEGNet` or `--model Shallow` and `--rep R1` or `--rep R2`; then `lee-doses --analyse` | `R11_B_G0_existing.json`, `R11_B_results.json` |
| Baseline dependence on the probability and log-odds scales | `ceiling-audit --execute` after all identity-swap and transformer results | `R8_C_results.json` |

### Dreyer2023 and synthetic studies

| Reported analysis | Commands in order | Main output |
|---|---|---|
| Producer-file audit and independent reader agreement | `audit-dreyer`, `build-dreyer --all`, `validate-dreyer` | `R12X_G1R_cohort_validity.json`, `R12X_G1R_feature_validation.json` |
| Pre-outcome inference calibration | `dreyer-calibration` | `R12X_G1R_calibration.json` |
| Held-out-run selection, confound controls and sensitivity cohort | `dreyer-selection --worker`, `dreyer-selection --analyse`, `dreyer-descriptives`, `dreyer-audit`, `dreyer-tie-sensitivity`. The CPU fits can also be generated first with `dreyer-selection --execute-tslr`. | `R12X_results.json`, `R12X_descriptives.json`, `R12X_audit_recompute.json`, `R17P1_E2_tie_sensitivity.json` |
| Feasibility screen | `build-feasibility`, `feasibility --stages SAED` | `R15B_feasibility.json`. Features are built from fitting runs 1 through 4. |
| Post hoc operating characteristics | `simulate all`, `simulate S3b`, `summarize-simulation`. The full grids use 400 replicates per cell. Use `--quick` for reduced-size code verification. | `R15C0_S1.json` through `R15C0_S4.json`, `R15C0_S3b.json`, `R15G_simulation_extract.json` |
