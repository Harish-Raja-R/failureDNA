# FailureDNA: Reproducibility & Audit Protocol

This guide outlines the complete operational protocol to reproduce the environment, run verification tests, execute data engineering audits, and validate data leakage protections.

---

## 1. Environment Requirements

- **Python Version**: Python 3.11+ (verified on Python 3.13.1 64-bit AMD64).
- **Core Dependencies**:
  - `numpy >= 2.0.0`
  - `scipy >= 1.14.0`
  - `pandas >= 2.2.0`
  - `scikit-learn >= 1.5.0`
  - `torch >= 2.4.0`
  - `xgboost >= 2.1.0`
  - `pyyaml >= 6.0.2`
  - `pydantic >= 2.8.0`
  - `pytest >= 8.0.0`

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

---

## 2. Seed Control & Determinism Policy

To guarantee strict reproducibility across experimental runs:
1. All random processes in Python, NumPy, and PyTorch are initialized through `seed_everything(seed=42)` defined in `src/failuredna/utils/seed.py`.
2. PyTorch deterministic mode is enabled (`torch.backends.cudnn.deterministic = True`).
3. Shuffling of engine IDs in `src/failuredna/data/split.py` uses an isolated `np.random.RandomState(seed)` instance to prevent global RNG interference.

---

## 3. Dataset Setup Protocol

1. Place the official C-MAPSS FD004 text files into `data/raw/`:
   ```
   data/raw/
   ├── train_FD004.txt
   ├── test_FD004.txt
   └── RUL_FD004.txt
   ```
2. Verify file permissions and check that file names strictly match `train_FD004.txt`, `test_FD004.txt`, and `RUL_FD004.txt`.

---

## 4. Running the Verification & Audit Suite

### CLI Verification Command
To run the automated schema validation, trajectory length audit, engine-level partitioning, and sliding-window generation:
```bash
python experiments/verify_dataset.py --data-dir data/raw --variant FD004
```

- If raw files are present: Generates `reports/results/dataset_audit_FD004.json` containing exact trajectory length quantiles, near-constant sensor variance, operating condition distributions, and window counts.
- If raw files are missing: Runs the complete pipeline on synthetic C-MAPSS fixtures and logs `reports/results/synthetic_audit_report.json`, while explicitly reporting real-data verification as blocked pending data placement.

---

## 5. Running Automated Unit Tests

To run the unit test suite covering schema validation, malformed input rejection, missing value detection, engine-split disjointness, train-only scaler fitting, and temporal windowing:
```bash
pytest tests/ -v
```

---

## 6. Data Leakage Verification Checklist

| Leakage Gate | Implementation Verification | Test Coverage |
| :--- | :--- | :--- |
| **Engine Disjointness** | Partitioning occurs strictly by `unit_id`. $Train \cap Val = \emptyset$, $Train \cap Test = \emptyset$, $Val \cap Test = \emptyset$. | `tests/test_data.py::test_split_engines_by_unit_disjointness` |
| **No-Peeking Scaling** | Preprocessor is fit exclusively on `train_df`. Validation/Test transformed with frozen parameters. | `tests/test_data.py::test_preprocessor_train_only_fitting` |
| **Single-Engine Windows** | Sliding windows never cross engine boundaries. | `tests/test_windowing.py::test_windows_single_engine_isolation` |
| **Chronological Order** | Temporal window sequences preserve strictly increasing cycle order ($t_1 < t_2 < \dots < t_W$). | `tests/test_windowing.py::test_windows_strict_temporal_ordering` |
| **Label Exclusion** | RUL, cycle counts, and fault modes are strictly excluded from input feature matrices. | `src/failuredna/data/preprocessing.py` |
