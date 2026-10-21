"""select_model.py: Select the best model and class balance setting based on validation macro-F1.

Reads E1 run CSVs across configurations (e.g., class_balance: none, sqrt, balanced),
computes mean VALIDATION macro-F1 for each (model, class_balance) combination,
selects the top performer, and writes config_final.yaml with specified parameters.
"""
import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Select best model & class balance by validation macro-F1")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["config_quick.yaml", "config_quick_sqrt.yaml", "config_quick_balanced.yaml"],
        help="List of config yaml files containing E1 runs",
    )
    parser.add_argument(
        "--base-config",
        default="config.yaml",
        help="Base config file to copy full-dataset parameters from",
    )
    parser.add_argument(
        "--out-config",
        default="config_final.yaml",
        help="Target config file to generate",
    )
    parser.add_argument(
        "--results-dir",
        default="results_final",
        help="Results directory for final config",
    )
    return parser.parse_args()


def load_e1_results(configs):
    records = []
    for cpath in configs:
        if not os.path.exists(cpath):
            print(f"[warning] Config file not found: {cpath}")
            continue
        with open(cpath, "r") as f:
            cfg = yaml.safe_load(f)
        cb = str(cfg.get("class_balance", "none"))
        rdir = cfg.get("results_dir", ".")
        e1_csv = os.path.join(rdir, "e1_closed_set_runs.csv")
        if not os.path.exists(e1_csv):
            print(f"[warning] E1 CSV not found: {e1_csv} for config {cpath}")
            continue
        df = pd.read_csv(e1_csv)
        if "val_macro_f1" not in df.columns:
            print(f"[warning] 'val_macro_f1' column missing in {e1_csv}")
            continue
        # Exclude baselines and rows without validation scores
        valid = df[~df["model"].isin(["majority_baseline", "shuffled_label_rf"])].dropna(subset=["val_macro_f1"]).copy()
        valid["class_balance"] = cb
        valid["config_file"] = cpath
        records.append(valid)

    if not records:
        raise RuntimeError("No valid E1 runs found across the provided configs.")
    return pd.concat(records, ignore_index=True)


def main():
    args = parse_args()
    print("=" * 70)
    print("Model Selection based on Mean VALIDATION Macro-F1 Only")
    print("=" * 70)

    combined_df = load_e1_results(args.configs)

    # Group by model and class_balance, aggregate mean and std across seeds
    summary = combined_df.groupby(["model", "class_balance"]).agg(
        n_runs=("val_macro_f1", "count"),
        mean_val_macro_f1=("val_macro_f1", "mean"),
        std_val_macro_f1=("val_macro_f1", "std"),
        mean_test_macro_f1=("macro_f1", "mean"),
        mean_test_accuracy=("accuracy", "mean"),
    ).reset_index()

    # Sort strictly by mean VALIDATION macro-F1 descending
    summary = summary.sort_values(by="mean_val_macro_f1", ascending=False).reset_index(drop=True)

    print("\nCandidate Combinations (ranked by validation macro-F1):")
    fmt_df = summary.copy()
    fmt_df["std_val_macro_f1"] = fmt_df["std_val_macro_f1"].fillna(0.0)
    for col in ["mean_val_macro_f1", "std_val_macro_f1", "mean_test_macro_f1", "mean_test_accuracy"]:
        fmt_df[col] = fmt_df[col].map(lambda x: f"{x:.4f}")
    print(fmt_df.to_string(index=True))

    best = summary.iloc[0]
    best_model = best["model"]
    best_cb = best["class_balance"]
    best_val_f1 = best["mean_val_macro_f1"]

    print("\n" + "-" * 70)
    print(f"SELECTED BEST COMBINATION:")
    print(f"  Model:         {best_model}")
    print(f"  Class Balance: {best_cb}")
    print(f"  Mean Val F1:   {best_val_f1:.4f}")
    print("-" * 70)

    # Load base config
    base_file = args.base_config if os.path.exists(args.base_config) else args.configs[0]
    with open(base_file, "r") as f:
        final_cfg = yaml.safe_load(f)

    # Apply final parameters
    final_cfg["model"] = best_model
    final_cfg["class_balance"] = best_cb
    final_cfg["seeds"] = [1, 2, 3, 4, 5]
    final_cfg["holdout_families"] = ["httpflood", "slowrate", "synflood", "scans"]
    final_cfg["results_dir"] = args.results_dir

    with open(args.out_config, "w") as f:
        yaml.dump(final_cfg, f, sort_keys=False, default_flow_style=None)

    print(f"\nWrote final configuration to: {args.out_config}")
    print(f"  seeds: {final_cfg['seeds']}")
    print(f"  holdout_families: {final_cfg['holdout_families']}")
    print(f"  results_dir: {final_cfg['results_dir']}")
    print(f"  model: {final_cfg['model']}")
    print(f"  class_balance: {final_cfg['class_balance']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
