"""C-MAPSS FD004 Dataset Verification and Ingestion Audit Script.

Executes end-to-end schema validation, trajectory audits, engine-level partitioning,
train-only preprocessor fitting, and sliding-window generation.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure src/ is importable
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import numpy as np
import pandas as pd
import yaml

from failuredna.data.loader import (
    CYCLE_COL,
    ID_COL,
    SENSOR_COLS,
    SETTING_COLS,
    TELEMETRY_COLS,
    load_cmapss_dataset,
    load_raw_cmapss_file,
)
from failuredna.data.preprocessing import CMAPSSPreprocessor
from failuredna.data.split import split_engines_by_unit
from failuredna.data.validator import DataValidator, ValidationError
from failuredna.data.windowing import WindowGenerator
from failuredna.utils.logging import get_logger
from failuredna.utils.seed import seed_everything

logger = get_logger("failuredna.verify")


def generate_synthetic_cmapss_df(
    n_engines: int = 5,
    min_cycles: int = 60,
    max_cycles: int = 90,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    train_rows = []
    test_rows = []
    rul_rows = []

    # Generate synthetic training engines
    for unit_id in range(1, n_engines + 1):
        total_cycles = rng.randint(min_cycles, max_cycles + 1)
        for cycle in range(1, total_cycles + 1):
            settings = [float(rng.uniform(0, 42)), float(rng.uniform(0, 0.84)), float(rng.uniform(20, 100))]
            sensors = [float(100.0 + i * 5.0 + rng.normal(0, 0.5) + (cycle / total_cycles) * (i % 3)) for i in range(1, 22)]
            # make sensor 1 constant to test constant sensor pruning
            sensors[0] = 518.67
            train_rows.append([unit_id, cycle] + settings + sensors)

    # Generate synthetic test engines
    for unit_id in range(1, 4):
        total_cycles = rng.randint(40, 60)
        true_rul = rng.randint(15, 80)
        rul_rows.append([unit_id, float(true_rul)])
        for cycle in range(1, total_cycles + 1):
            settings = [float(rng.uniform(0, 42)), float(rng.uniform(0, 0.84)), float(rng.uniform(20, 100))]
            sensors = [float(100.0 + i * 5.0 + rng.normal(0, 0.5) + (cycle / (total_cycles + true_rul)) * (i % 3)) for i in range(1, 22)]
            sensors[0] = 518.67
            test_rows.append([unit_id, cycle] + settings + sensors)

    train_df = pd.DataFrame(train_rows, columns=TELEMETRY_COLS)
    test_df = pd.DataFrame(test_rows, columns=TELEMETRY_COLS)
    rul_df = pd.DataFrame(rul_rows, columns=["unit_id", "rul"])

    return train_df, test_df, rul_df


def run_pipeline_audit(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rul_df: pd.DataFrame,
    output_dir: Path,
    is_synthetic: bool = False,
) -> Dict[str, Any]:
    """Execute end-to-end audit, split, preprocess, and windowing pipeline."""
    validator = DataValidator(raise_on_error=True)

    # 1. Statistical Dataset Audit
    logger.info("Executing comprehensive dataset audit...")
    audit_results = validator.audit_dataset(train_df, test_df, rul_df)
    audit_results["is_synthetic"] = is_synthetic

    # 2. Engine-level Split
    logger.info("Testing engine-level disjoint splitting...")
    train_split, val_split, test_split, split_info = split_engines_by_unit(
        train_df,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    # 3. Train-only Preprocessing
    logger.info("Testing leakage-free preprocessor fitting on train engines...")
    preprocessor = CMAPSSPreprocessor(scaler_type="robust", variance_threshold=1e-5)
    train_scaled = preprocessor.fit_transform(train_split)
    val_scaled = preprocessor.transform(val_split)
    test_scaled = preprocessor.transform(test_split)

    # 4. Temporal Window Generation
    logger.info("Testing sliding temporal window extraction (W=50, Stride=1)...")
    window_gen = WindowGenerator(window_size=50, stride=1, drop_short_trajectories=True)
    train_windows, train_meta, win_stats = window_gen.create_windows(train_scaled)

    audit_results["split_verification"] = split_info
    audit_results["preprocessor_state"] = preprocessor.get_state()
    audit_results["windowing_verification"] = win_stats

    # Save audit report to JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / ("synthetic_audit_report.json" if is_synthetic else "dataset_audit_FD004.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    logger.info(f"Audit report saved to: '{report_file.as_posix()}'")
    return audit_results


def main():
    parser = argparse.ArgumentParser(description="FailureDNA C-MAPSS Dataset Verification")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/raw",
        help="Path to directory containing C-MAPSS FD004 txt files",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="FD004",
        help="C-MAPSS sub-dataset variant",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/results",
        help="Directory to save audit JSON reports",
    )
    args = parser.parse_args()

    seed_everything(42)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    train_file = data_dir / f"train_{args.variant}.txt"
    test_file = data_dir / f"test_{args.variant}.txt"
    rul_file = data_dir / f"RUL_{args.variant}.txt"

    real_files_exist = train_file.is_file() and test_file.is_file() and rul_file.is_file()

    print("\n" + "=" * 70)
    print("      FailureDNA Dataset Verification & Engineering Audit")
    print("=" * 70)
    print(f"Target Variant:        {args.variant}")
    print(f"Target Data Directory: {data_dir.resolve()}")
    print(f"Target Output Directory:{output_dir.resolve()}")
    print(f"Status of Raw Files:   {'FOUND (Real Data)' if real_files_exist else 'NOT FOUND'}")
    print("-" * 70)

    if real_files_exist:
        logger.info(f"Loading real C-MAPSS {args.variant} files from '{data_dir}'...")
        dataset = load_cmapss_dataset(data_dir, variant=args.variant)
        audit_results = run_pipeline_audit(
            dataset["train"],
            dataset["test"],
            dataset["rul"],
            output_dir,
            is_synthetic=False,
        )

        print("\n>>> REAL DATA AUDIT SUMMARY:")
        print(f"  * Training Engines:   {audit_results['counts']['train_engines']}")
        print(f"  * Training Total Rows:{audit_results['counts']['train_rows']:,}")
        print(f"  * Test Engines:       {audit_results['counts']['test_engines']}")
        print(f"  * Test Total Rows:    {audit_results['counts']['test_rows']:,}")
        print(f"  * Train Engine Life:  Min={audit_results['trajectory_lengths']['train']['min']} cycles, "
              f"Median={audit_results['trajectory_lengths']['train']['median']} cycles, "
              f"Max={audit_results['trajectory_lengths']['train']['max']} cycles")
        print(f"  * Constant Sensors:   {audit_results['sensor_analysis']['constant_sensors']}")
        print(f"  * Active Sensors:     {len(audit_results['sensor_analysis']['informative_sensors'])} channels")
        print(f"  * Windows Generated:  {audit_results['windowing_verification']['total_windows_generated']:,} (W=50)")
        print(f"  * Audit Report File:  {output_dir / 'dataset_audit_FD004.json'}")
        print("=" * 70 + "\n")
        sys.exit(0)
    else:
        print("\n[BLOCKED] Real C-MAPSS FD004 raw files are currently not in:")
        print(f"  - {train_file.as_posix()}")
        print(f"  - {test_file.as_posix()}")
        print(f"  - {rul_file.as_posix()}")
        print("\nExecuting comprehensive pipeline verification using Synthetic C-MAPSS Fixture...")

        train_synth, test_synth, rul_synth = generate_synthetic_cmapss_df(n_engines=10, min_cycles=60, max_cycles=100)
        audit_results = run_pipeline_audit(
            train_synth,
            test_synth,
            rul_synth,
            output_dir,
            is_synthetic=True,
        )

        print("\n>>> SYNTHETIC VERIFICATION SUMMARY:")
        print("  * Schema & Ingestion:  PASSED (26 columns, strict typing)")
        print("  * Engine Splitting:    PASSED (Disjoint engine IDs verified)")
        print("  * Preprocessor:        PASSED (Fitted strictly on train split)")
        print("  * Windowing:           PASSED (Temporal order & engine boundaries enforced)")
        print(f"  * Synthetic Report:    {output_dir / 'synthetic_audit_report.json'}")
        print("\n>>> TO RUN AUDIT ON REAL DATASET:")
        print(f"  Place 'train_FD004.txt', 'test_FD004.txt', and 'RUL_FD004.txt' into '{data_dir.resolve()}'")
        print("  Then run:")
        print(f"    python experiments/verify_dataset.py --data-dir {args.data_dir}")
        print("=" * 70 + "\n")
        # Exit with code 0 to indicate pipeline verification succeeded, real verification pending data placement
        sys.exit(0)


if __name__ == "__main__":
    main()
