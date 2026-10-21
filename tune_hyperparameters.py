"""tune_hyperparameters.py: Grouped-CV hyperparameter tuning strictly on training data.

Groups by __block (or group_col) so adjacent flows/blocks never leak across folds.
Evaluates candidate hyperparameters by mean CV validation macro-F1, fits the best
model on full training data, and evaluates on validation set.
"""
import argparse
import os
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

from src import data as D, pipeline as P
from src.experiment import prepare


def parse_args():
    parser = argparse.ArgumentParser(description="Grouped-CV hyperparameter tuning for rf/xgb/lgbm")
    parser.add_argument("--config", default="config_quick.yaml", help="Configuration file path")
    parser.add_argument("--models", nargs="+", default=["rf", "xgb", "lgbm"], help="Models to tune")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--folds", type=int, default=3, help="Number of grouped-CV folds on training data")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print("Grouped-CV Hyperparameter Tuning (Training Data Only)")
    print(f"Config: {args.config} | Models: {args.models} | Folds: {args.folds} | Seed: {args.seed}")
    print("=" * 75)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    df = D.load_clean(cfg)
    X = D.feature_columns(df, cfg)
    
    # Prepare data without holdout family to evaluate closed-set validation performance
    s = prepare(df, X, cfg, seed=args.seed, family=None)
    Xtr, ytr = s["Xtr"], s["ytr"]
    Xva, yva = s["Xva"], s["yva"]
    gtr = s.get("gtr")
    
    w = P.class_weights(ytr, cfg.get("class_balance"))
    
    print(f"\n[data] Training samples: {len(Xtr)}, Validation samples: {len(Xva)}")
    if gtr is not None:
        print(f"[data] Grouping by: '{cfg.get('group_col')}' with {len(np.unique(gtr))} unique training groups")
    else:
        print("[data] No group column found, falling back to stratified CV")

    all_cv_records = []
    comparison = []

    for name in args.models:
        if name not in P.TUNING_GRIDS:
            print(f"[skip] Model '{name}' has no defined tuning grid in pipeline.py")
            continue
            
        print(f"\n>>> Tuning {name.upper()} ({len(P.TUNING_GRIDS[name])} parameter sets, {args.folds}-fold grouped-CV)...")
        
        # 1. Evaluate baseline default model
        default_model = P.make_model(name, args.seed)
        default_model.fit(Xtr, ytr, sample_weight=w)
        def_val_pred = default_model.predict(Xva)
        def_val_f1 = f1_score(yva, def_val_pred, average="macro", zero_division=0)
        def_val_acc = accuracy_score(yva, def_val_pred)
        
        # 2. Run Grouped-CV strictly on training data
        tuned_model, best_params, cv_df = P.tune_model(
            name, Xtr, ytr, gtr, seed=args.seed, sample_weight=w, n_splits=args.folds
        )
        
        # 3. Evaluate tuned model on validation set
        tuned_val_pred = tuned_model.predict(Xva)
        tuned_val_f1 = f1_score(yva, tuned_val_pred, average="macro", zero_division=0)
        tuned_val_acc = accuracy_score(yva, tuned_val_pred)
        
        for _, row in cv_df.iterrows():
            rec = row.to_dict()
            rec["is_selected_best"] = (row["raw_params"] == best_params)
            all_cv_records.append(rec)
            
        print("\nFold CV Results on Training Data:")
        for idx, row in cv_df.iterrows():
            mark = " [BEST]" if row["raw_params"] == best_params else ""
            print(f"  Candidate {idx + 1}: CV F1 = {row['mean_cv_macro_f1']:.4f} ± {row['std_cv_macro_f1']:.4f} | Params: {row['params']}{mark}")
            
        print(f"\nSelected Best Parameters for {name}: {best_params}")
        print(f"  Default Model -> Val Macro-F1: {def_val_f1:.4f} | Accuracy: {def_val_acc:.4f}")
        print(f"  Tuned Model   -> Val Macro-F1: {tuned_val_f1:.4f} | Accuracy: {tuned_val_acc:.4f}")
        print(f"  Validation Macro-F1 Delta: {tuned_val_f1 - def_val_f1:+.4f}")
        
        comparison.append({
            "model": name,
            "best_params": str(best_params),
            "best_cv_macro_f1": cv_df.iloc[0]["mean_cv_macro_f1"],
            "default_val_macro_f1": def_val_f1,
            "tuned_val_macro_f1": tuned_val_f1,
            "val_f1_delta": tuned_val_f1 - def_val_f1,
        })

    # Save tuning summary
    rdir = cfg.get("results_dir", ".")
    os.makedirs(rdir, exist_ok=True)
    comp_df = pd.DataFrame(comparison)
    comp_path = os.path.join(rdir, "hyperparameter_tuning.csv")
    comp_df.to_csv(comp_path, index=False)
    
    print("\n" + "=" * 75)
    print("HYPERPARAMETER TUNING SUMMARY:")
    print("=" * 75)
    print(comp_df.to_string(index=False))
    print(f"\nSaved detailed summary to: {comp_path}")


if __name__ == "__main__":
    main()
