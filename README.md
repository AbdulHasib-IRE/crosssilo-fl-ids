# Shared features, no shared signal: cross-silo federated intrusion detection between IoT and enterprise networks

Code, split definitions, per-run results and analysis scripts for the manuscript
"Shared features, no shared signal: cross-silo federated intrusion detection between IoT and enterprise networks"
(A. Hasib, S. C. Das, Md. Ashiqussalehin).

## What is here
| Folder | Contents |
|---|---|
| `paper_reproduction/` | Exported result files (`data/`) and the scripts that compute every reported statistic and figure. `run_all.sh` regenerates all statistics (`outputs/analysis/`) and all figures at 1000 dpi (`outputs/figures/`). |
| `results/` | Raw outputs of every run: 300 main-grid runs (within-domain and cross-silo, four rules, two splits, IID check, learning-rate swap), pilot runs, centralized references and transfer tests, sensitivity runs (Web family, Dirichlet 0.1), separate-head runs, robustness runs (global scaler, silo size, domain separability), standard-metric re-runs with confusion matrices, partial-personalization runs (`personal/`: output layer or attention and first dense layer kept silo-local, plus a check run with all layers shared that reproduces the main-grid run exactly), and domain-alignment runs (`align/`: CORAL and MMD penalties, shared and silo-local batch normalization (FedBN), batch-normalized within-silo baselines, their validation pilot, a smoke test and a check run that reproduces the main-grid run exactly). |
| `splits/` | Row indices and labels of the random and time-block splits of both silos, and the feature and family lists. No features or IP addresses are included. |
| `code/fl_core.py` | Federated simulation harness (the notebook patches it to a five-class head). |
| `code/align_core.py` | Cross-silo harness with CORAL / MMD alignment and shared or silo-local batch normalization. |
| `notebook/` | The Colab notebooks that ran every experiment (`CrossSilo_FL.ipynb`; `CrossSilo_Align.ipynb` for the alignment runs), with outputs cleared. |

## Reproduce the statistics and figures (no GPU, about 1 min)
pip install -r requirements.txt
bash paper_reproduction/run_all.sh
Expected output includes `27 panels, 0 failing` (the figure-panel consistency check). The scripts call `python3`; set `PYTHON=python` to use another interpreter.

## Changes in v1.0.1
Adds the domain-alignment and batch-normalization experiments (60 runs, 16 pilot runs) with their harness, notebook and statistics. All v1.0.0 result files are unchanged; the two analysis scripts gain the alignment statistics and figure panels.

## Data
CIC IoT-DIAD 2024 (https://www.unb.ca/cic/datasets/iot-diad-2024.html) and Improved CICIDS2017 (Engelen et al., 2021) are not redistributed.

## Citation
See `CITATION.cff`. License: MIT (code). Result files: CC BY 4.0.

## Contact
Saurav Chandra Das (corresponding author), saurav0001@uftb.ac.bd
