# qorder

Code and configuration files for the paper **"Feature Ordering by Distance to the Measured Qubit in Shallow Quantum Classifiers"** (M. de Paula Barbosa Souza, B. Kano, E. Fonseca - Inteli, 2026).

The paper proposes a search-free heuristic for shallow variational quantum classifiers with a local cost function:

1. Rank the features by mutual information with the label, estimated on the training set only.
2. Assign them to qubits in increasing order of graph distance to the measured qubit.

With nearest-neighbor CZ gates, the backward causal cone of a single-qubit measurement grows by one qubit per layer, so after L layers only L + 1 qubits (measuring at the edge of the chain) can influence the prediction. The heuristic makes sure those slots hold the most informative features.

Raw results (CSV files, trained models and figures) are in the [results folder on Google Drive](https://drive.google.com/drive/folders/1MwfTVtCP-QoG_rZf7RxPvmf9_mpnzx9w).

## Repository layout

| Path | Contents |
|---|---|
| `qorder/` | Library: datasets, orderings (`ordering.py`), shallow circuit and data re-uploading model, NumPy reference simulator, shot-noise trainer, hardware validation, statistical analysis |
| `configs/` | One YAML file per experiment (see the table below) |
| `scripts/` | Hardware and noisy-simulator validation, summaries, a test runner that does not need pytest |
| `tests/` | Unit and equivalence tests (PennyLane, Qiskit and the NumPy reference give the same values; causal-cone checks) |
| `notebooks/` | Colab notebooks used to run the experiments and to produce the figures and tables of the paper |
| `tools/` | Scripts that generate the notebooks |

## Running

On Google Colab, open a notebook from `notebooks/` and run the cells in order:

- `colab.ipynb` sets up the environment, runs the tests and the main experiments (rounds 1–2).
- `experiments_round3.ipynb` and `experiments_round4.ipynb` run the resampled-split, depth, within-cone and shot-noise experiments.
- `figures.ipynb` reads the results and produces Figs. 1–4 and Tables II–III.

Locally (Linux or macOS, Python ≥ 3.10):

```bash
pip install -r requirements.txt
python -m pytest -q                       # or: python scripts/run_tests_without_pytest.py
python -m qorder.run configs/exp0_smoke_offline.yaml          # NumPy only, ~1 min
python -m qorder.run configs/exp2_breast_cancer_shallow.yaml --quick
python -m qorder.analysis results/exp2_breast_cancer_shallow_quick
python -m qorder.analysis results/exp3_depth_cost --depth results/exp4_gradients   # required depth (H3)
```

The pins in `requirements.txt` (PennyLane 0.42 + JAX 0.6.2) are a combination known to work; do not upgrade one without the other. If a run is interrupted, run it again: completed models are skipped.

## Experiments

| Config | Hypothesis | Description |
|---|---|---|
| `exp0_smoke_offline` | - | Pipeline check with the NumPy simulator and scikit-learn `digits` |
| `exp1_adversarial_drc`, `exp7_adversarial_n1000` | H1 | Model and adversarial dataset of Salmenperä et al. (10 informative + 10 noise features), with 300 and 1000 training samples |
| `exp2_{mnist,fashion,breast_cancer}_shallow` | H1 | 8-qubit shallow circuit with local cost, L = 1, 2, 3 |
| `exp5_*`, `exp6_*_shuffled` | H1 | Tabular datasets in their original column order and PCA components in a fixed random order |
| `exp8_*_resplit` | H1 | The nine datasets with arbitrary default order, with a new train–test split per seed, including the within-cone baselines `cone_reversed` and `cone_random` |
| `exp3_depth_cost` | H2, H3 | L = 1 to 7 with local and global costs on Breast Cancer |
| `exp4_gradients` | H3 | Gradient variance as a function of L and n, local and global costs |
| `exp9_shots_breast_cancer`, `exp9b_shots_global` | H3 | Training with gradients estimated from 100, 10 and 2 shots (local cost) and a global-cost control |

`scripts/validate_hardware.py` evaluates trained models with shots, with the IBM Fez noise model or on IBM hardware (with a budget check). The IBM token is read from the `IBM_QUANTUM_TOKEN` environment variable (Colab Secrets) and is never stored in the repository.

Each run writes `results/<name>/runs.csv` (one row per trained model), exported models in `models/`, and the tables and plots produced by `qorder.analysis`.

## Shallow circuit

Each layer applies RY(xᵢ) (first layer only, unless `reupload: true`), then a trainable RY(θ) and a CZ on every edge of the chain. A final RY(θ) follows the last layer, and Z₀ (local cost) or Z₀Z₁…Z₇ (global cost) is measured. `tests/test_reference.py` checks that only qubits 0…L influence ⟨Z₀⟩.

## License and credits

Apache-2.0. Parts of `datasets.py` and `models.py` are adapted from [QMLPerm](https://github.com/Ilmosal/QMLPerm) (Salmenperä, Perkkola & Nurminen, 2025) and [qml-benchmarks](https://github.com/XanaduAI/qml-benchmarks) (Xanadu), both Apache-2.0; see `NOTICE`.
