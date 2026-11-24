# PMIS-EEG

Scientific code for the paired matched identity swap study of exact-copy exposure in motor-imagery EEG. The release includes input builders, frozen experiment designs, training and evaluation workflows, and simulations. It covers HS-Stroke, Shin2022, Lee2019 (OpenBMI) and Dreyer2023.

## Run the code

Use Linux and Python 3.11 with the dependencies in `requirements.txt`. Run `python -m experiments --help` for the command list. The workflow guide covers input sources, local workspace setup, model checkpoints and command order. [Workflow guide](experiments/WORKFLOWS.md)

Obtain the original datasets and any required weights from their upstream sources. Keep input files and generated research artifacts in your local workspace. This release excludes EEG, participant and trial records, weights, checkpoints, historical reference digests, research history, and figure and manuscript generation. [Porting and provenance](PROVENANCE.md)

## Verification

The reviewed candidate passed 216 tests and 43 command help checks. Seven CUDA-only tests were skipped. The verification included full selected input builds, CPU fit stages and simulations, plus bounded real-data checks and analysis replays from saved fits. Full fresh neural refits were outside this verification. Public workflows rebuild prerequisites locally. Exact comparisons to the historical artifacts require user-supplied receipts. [Historical verification](PROVENANCE.md#historical-verification)

## Citation and reuse

Cite the repository URL and tag `v0.1.0`. No project-wide reuse licence is granted. Applicable upstream component licences and attributions accompany their source, including CC BY-NC 4.0 terms for the Meta MAE-derived portions. [PMIS-EEG v0.1.0](https://github.com/AliAmini93/PMIS-EEG/tree/v0.1.0)
