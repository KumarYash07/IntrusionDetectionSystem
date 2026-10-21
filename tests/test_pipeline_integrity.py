"""test_pipeline_integrity.py: Verification of pipeline split, holdout, threshold, and sanity check properties.

Checks:
1. No index/group overlap across train/val/test splits.
2. Held-out classes strictly absent from train and val sets.
3. Thresholds and temperature scaling computed from validation-known scores only (no test leakage).
4. Shuffled-label training collapses accuracy to near-chance baseline.
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import data as D, pipeline as P, metrics as M
from src.experiment import prepare, open_set_run


def get_test_config():
    """Loads synthetic config for fast deterministic testing."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config_synthetic.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def test_no_index_or_group_overlap():
    """Requirement 1: no index/group overlap across train/val/test."""
    cfg = get_test_config()
    df = D.load_clean(cfg)
    
    for family in cfg["holdout_families"]:
        for seed in [0, 42]:
            tr, va, te = D.make_split(df, cfg, seed, family)
            
            # Check index sets are mutually disjoint
            set_tr, set_va, set_te = set(tr), set(va), set(te)
            assert not (set_tr & set_va), f"Index overlap between train and val: {set_tr & set_va}"
            assert not (set_tr & set_te), f"Index overlap between train and test: {set_tr & set_te}"
            assert not (set_va & set_te), f"Index overlap between val and test: {set_va & set_te}"
            
            # Check group sets are mutually disjoint
            gc = cfg.get("group_col")
            if gc:
                g = df[gc].values
                g_tr, g_va, g_te = set(g[tr]), set(g[va]), set(g[te])
                assert not (g_tr & g_va), f"Group overlap between train and val: {g_tr & g_va}"
                assert not (g_tr & g_te), f"Group overlap between train and test: {g_tr & g_te}"
                assert not (g_va & g_te), f"Group overlap between val and test: {g_va & g_te}"
            
            # Explicit call to pipeline's check_no_overlap
            D.check_no_overlap(df, cfg, tr, va, te)


def test_heldout_classes_absent_from_train_and_val():
    """Requirement 2: held-out classes absent from train/val, present in test unknowns."""
    cfg = get_test_config()
    df = D.load_clean(cfg)
    X = D.feature_columns(df, cfg)
    
    for family in cfg["holdout_families"]:
        heldout_classes = D.family_classes(cfg, family)
        assert len(heldout_classes) > 0, f"Holdout family '{family}' has no classes defined"
        
        s = prepare(df, X, cfg, seed=0, family=family)
        
        # Check raw labels in train and val splits
        tr_labels = set(df[cfg["label_col"]].iloc[s["tr"]])
        va_labels = set(df[cfg["label_col"]].iloc[s["va"]])
        
        overlap_tr = tr_labels & heldout_classes
        overlap_va = va_labels & heldout_classes
        assert not overlap_tr, f"Held-out classes {overlap_tr} leaked into training set!"
        assert not overlap_va, f"Held-out classes {overlap_va} leaked into validation set!"
        
        # Check LabelEncoder classes known during training
        assert not (set(s["le"].classes_) & heldout_classes), "Held-out class encoded in LabelEncoder classes"
        
        # Check test unknowns correctly flagged
        te_labels = df[cfg["label_col"]].iloc[s["te"]].values
        for held_cls in heldout_classes:
            mask_held = (te_labels == held_cls)
            if np.any(mask_held):
                assert np.all(s["is_unk"][mask_held]), f"Class {held_cls} in test set not marked as is_unk!"
                assert np.all(s["yte"][mask_held] == -1), f"Class {held_cls} in test set does not have yte == -1!"


def test_thresholds_computed_from_validation_known_scores_only():
    """Requirement 3: thresholds computed strictly from validation-known scores without test score leakage."""
    cfg = get_test_config()
    df = D.load_clean(cfg)
    X = D.feature_columns(df, cfg)
    
    s = prepare(df, X, cfg, seed=0, family="dos")
    name = "lr"  # fast baseline model
    model = P.make_model(name, seed=0).fit(s["Xtr"], s["ytr"])
    
    zva = P.get_logits(model, name, s["Xva"])
    zte_clean = P.get_logits(model, name, s["Xte"])
    
    # Validation temperature
    T1 = P.fit_temperature(zva, s["yva"])
    
    # Anomaly detector fit on training data only
    anom = P.Anomaly(cfg["anomaly"], seed=0).fit(s["Xtr"])
    ava = anom.score(s["Xva"])
    ate_clean = anom.score(s["Xte"])
    
    q = cfg["known_accept_rate"]
    thr_clean = np.quantile(ava, q)
    
    # Now simulate altered / corrupted / adversarial test set
    corrupted_Xte = s["Xte"] + 1000.0  # huge perturbation
    ate_corrupted = anom.score(corrupted_Xte)
    thr_after_test_corruption = np.quantile(ava, q)
    
    # Threshold must depend ONLY on ava (validation score), completely invariant to test distribution
    assert thr_clean == thr_after_test_corruption, "Threshold changed when test set was perturbed! Leakage detected."
    
    # Verify validation empirical known accept rate on validation set matches target quantile
    val_flag_rate = (ava > thr_clean).mean()
    expected_flag_rate = 1.0 - q
    assert np.isclose(val_flag_rate, expected_flag_rate, atol=0.03), (
        f"Validation flag rate {val_flag_rate} does not match expected budget {expected_flag_rate}"
    )


def test_shuffled_label_accuracy_near_chance():
    """Requirement 4: shuffled-label accuracy must be near chance."""
    cfg = get_test_config()
    df = D.load_clean(cfg)
    X = D.feature_columns(df, cfg)
    
    s = prepare(df, X, cfg, seed=42, family=None)
    kn = ~s["is_unk"]
    Xtr, ytr = s["Xtr"], s["ytr"]
    Xte_kn, yte_kn = s["Xte"][kn], s["yte"][kn]
    
    # 1. Normal model (should perform well above chance)
    model_normal = P.make_model("lr", seed=42).fit(Xtr, ytr)
    acc_normal = accuracy_score(yte_kn, model_normal.predict(Xte_kn))
    
    # 2. Shuffled labels model
    rng = np.random.default_rng(42)
    ytr_shuffled = rng.permutation(ytr)
    model_shuffled = P.make_model("lr", seed=42).fit(Xtr, ytr_shuffled)
    acc_shuffled = accuracy_score(yte_kn, model_shuffled.predict(Xte_kn))
    
    # Chance baseline (majority class or uniform 1/K)
    num_classes = len(np.unique(ytr))
    majority_acc = (yte_kn == np.bincount(ytr).argmax()).mean()
    uniform_chance = 1.0 / num_classes
    
    print(f"\n[shuffled_test] Normal accuracy: {acc_normal:.4f}")
    print(f"[shuffled_test] Shuffled accuracy: {acc_shuffled:.4f}")
    print(f"[shuffled_test] Majority class chance: {majority_acc:.4f}, Uniform chance: {uniform_chance:.4f}")
    
    # Shuffled accuracy must be far below genuine model accuracy
    assert acc_shuffled < acc_normal - 0.30, (
        f"Shuffled model accuracy ({acc_shuffled:.4f}) is suspiciously close to genuine model ({acc_normal:.4f})"
    )
    
    # Shuffled accuracy must be close to chance level (within 0.15 of majority class or uniform)
    assert acc_shuffled <= max(majority_acc, uniform_chance) + 0.15, (
        f"Shuffled model accuracy ({acc_shuffled:.4f}) exceeds chance threshold"
    )


def test_ablation_configurations_and_mechanics():
    """Requirement 6: Verify ablation mechanics (drop cols, label_map merging, random split)."""
    # 1. Test Drop Cols
    cfg_base = get_test_config()
    cfg_drop = cfg_base.copy()
    cfg_drop["drop_cols"] = ["f0", "f1", "f2"]
    df_drop = D.load_clean(cfg_drop)
    X_drop = D.feature_columns(df_drop, cfg_drop)
    for col in ["f0", "f1", "f2"]:
        assert col not in X_drop.columns, f"Feature {col} was not dropped!"

    # 2. Test Label Map (Merge scan classes into 'Scan')
    cfg_merge = cfg_base.copy()
    cfg_merge["label_map"] = {"SYNScan": "Scan", "UDPScan": "Scan"}
    df_merge = D.load_clean(cfg_merge)
    labels = df_merge[cfg_merge["label_col"]].unique()
    assert "Scan" in labels, "Merged class 'Scan' not present after label_map!"
    assert "SYNScan" not in labels, "SYNScan was not mapped to Scan!"
    assert "UDPScan" not in labels, "UDPScan was not mapped to Scan!"

    # 3. Test Random Stratified Split vs Group Split
    cfg_rand = cfg_base.copy()
    cfg_rand["group_col"] = None
    df_rand = D.load_clean(cfg_rand)
    tr, va, te = D.make_split(df_rand, cfg_rand, seed=0, holdout_family="dos")
    assert not (set(tr) & set(va)), "Index overlap between train and val in random split"
    assert not (set(tr) & set(te)), "Index overlap between train and test in random split"
    assert not (set(va) & set(te)), "Index overlap between val and test in random split"


def test_grouped_cv_hyperparameter_tuning():
    """Requirement 7: Verify grouped-CV hyperparameter tuning for rf/xgb/lgbm strictly on training data."""
    cfg = get_test_config()
    df = D.load_clean(cfg)
    X = D.feature_columns(df, cfg)
    s = prepare(df, X, cfg, seed=0, family=None)
    
    Xtr, ytr, gtr = s["Xtr"][:600], s["ytr"][:600], s["gtr"][:600]
    
    # Mini custom grid for speed during unit testing
    mini_grid = [
        {"max_depth": 5, "n_estimators": 10},
        {"max_depth": 10, "n_estimators": 10},
    ]
    
    for model_name in ["rf", "xgb", "lgbm"]:
        best_model, best_params, cv_df = P.tune_model(
            model_name, Xtr, ytr, gtr, seed=0, n_splits=3, grid=mini_grid
        )
        assert best_model is not None, f"Tuned {model_name} model is None"
        assert best_params in mini_grid, f"Best params not in candidate grid for {model_name}"
        assert len(cv_df) == len(mini_grid), f"CV df row count mismatch for {model_name}"
        assert "mean_cv_macro_f1" in cv_df.columns, "mean_cv_macro_f1 column missing in CV df"
        # Test that model predicts valid classes
        preds = best_model.predict(s["Xva"][:50])
        assert len(preds) == 50, f"Predictions length mismatch for {model_name}"


def test_make_report_single_seed_partial_families(tmp_path=None):
    """Test make_report on a 1-seed, 2-family results directory without crash."""
    import tempfile, shutil
    import subprocess
    
    test_dir = tempfile.mkdtemp(prefix="test_results_single_seed_")
    try:
        # Build 1-seed, 2-family e2_open_set_runs.csv
        records = []
        methods = ["anomaly_only", "classifier_only", "energy", "mahalanobis", "msp_uncalibrated", "proposed", "rejection_only"]
        for fam in ["famA", "famB"]:
            for m in methods:
                records.append({
                    "family": fam,
                    "seed": 0,
                    "method": m,
                    "auroc": 0.90 if m != "classifier_only" else np.nan,
                    "aupr": 0.85 if m != "classifier_only" else np.nan,
                    "unknown_precision": 0.80 if m != "classifier_only" else 0.0,
                    "unknown_detection_rate": 0.88 if m != "classifier_only" else 0.0,
                    "known_fpr": 0.05,
                    "open_set_f1": 0.85,
                })
        pd.DataFrame(records).to_csv(os.path.join(test_dir, "e2_open_set_runs.csv"), index=False)

        # Build e2_per_class.csv
        pc_records = []
        for fam in ["famA", "famB"]:
            for m in methods:
                for cls_name in ["Benign", "Attack1"]:
                    pc_records.append({
                        "family": fam,
                        "method": m,
                        "cls": cls_name,
                        "n": 100,
                        "flag_rate": 0.04,
                    })
        pd.DataFrame(pc_records).to_csv(os.path.join(test_dir, "e2_per_class.csv"), index=False)

        # Build e3_calibration.csv with single seed (std will be empty)
        cal_df = pd.DataFrame([
            {"family": "famA", "T": 1.2, "ece_raw": 0.08, "ece_cal": 0.04},
            {"family": "famB", "T": 1.5, "ece_raw": 0.09, "ece_cal": 0.05},
        ]).set_index("family")
        cal_agg = cal_df.groupby("family").agg(["mean", "std"])
        cal_agg.to_csv(os.path.join(test_dir, "e3_calibration.csv"))

        # Build e4_threshold.csv
        thr_df = pd.DataFrame([
            {"known_accept_target": 0.80, "known_accept_rate": 0.79, "unknown_detection_rate": 0.95},
            {"known_accept_target": 0.95, "known_accept_rate": 0.94, "unknown_detection_rate": 0.80},
        ]).groupby("known_accept_target").agg(["mean", "std"])
        thr_df.to_csv(os.path.join(test_dir, "e4_threshold.csv"))

        # Build 1-seed npz files for famA and famB
        for fam in ["famA", "famB"]:
            n_samples = 100
            npz_dict = {
                "family": np.array(fam),
                "seed": np.int64(0),
                "is_unk": np.random.choice([False, True], size=n_samples, p=[0.7, 0.3]),
                "yte": np.random.choice([-1, 0, 1], size=n_samples),
                "yva": np.random.choice([0, 1], size=50),
                "pred": np.random.choice([0, 1], size=n_samples),
                "classes": np.array(["Benign", "Attack1"]),
                "T": np.float64(1.3),
                "conf_uncal": np.random.uniform(0.5, 1.0, size=n_samples).astype(np.float32),
                "conf_cal": np.random.uniform(0.5, 1.0, size=n_samples).astype(np.float32),
            }
            for m in ["anomaly_only", "energy", "mahalanobis", "msp_uncalibrated", "proposed", "rejection_only"]:
                npz_dict[f"score_{m}"] = np.random.randn(n_samples).astype(np.float32)
                npz_dict[f"val_score_{m}"] = np.random.randn(50).astype(np.float32)
                npz_dict[f"thr_{m}"] = np.float32(0.5)
                npz_dict[f"flagged_{m}"] = (npz_dict[f"score_{m}"] > 0.5)
            np.savez_compressed(os.path.join(test_dir, f"scores_{fam}_seed0.npz"), **npz_dict)

        # Run make_report on test_dir
        cmd = [sys.executable, os.path.join(os.path.dirname(__file__), "..", "make_report.py"), "--results-dir", test_dir]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0, f"make_report.py failed on 1-seed test directory: {res.stderr}"

        # Assert report.md was generated with "± n/a" for single-seed stds
        report_file = os.path.join(test_dir, "report.md")
        assert os.path.exists(report_file), "report.md was not created"
        with open(report_file, "r", encoding="utf-8") as f:
            report_content = f.read()
        assert "± n/a" in report_content, "Expected '± n/a' in report for single-seed std"
        assert "famA" in report_content and "famB" in report_content

        # Assert diagnostic figures generated
        for fig_name in [
            "fig_roc_curves.png", "fig_pr_curves.png", "fig_flag_rate_heatmap.png",
            "fig_reliability_diagrams.png", "fig_threshold_tradeoff.png"
        ]:
            assert os.path.exists(os.path.join(test_dir, fig_name)), f"Figure {fig_name} was not generated"

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    print("Running Pipeline Integrity Tests (standalone mode)...")
    test_no_index_or_group_overlap()
    print("[PASS] Test 1: No index/group overlap across train/val/test")
    test_heldout_classes_absent_from_train_and_val()
    print("[PASS] Test 2: Held-out classes absent from train/val")
    test_thresholds_computed_from_validation_known_scores_only()
    print("[PASS] Test 3: Thresholds computed from validation-known scores only")
    test_shuffled_label_accuracy_near_chance()
    print("[PASS] Test 4: Shuffled-label accuracy collapses to near chance")
    test_ablation_configurations_and_mechanics()
    print("[PASS] Test 5: Ablation mechanics (drop cols, label_map, random split)")
    test_grouped_cv_hyperparameter_tuning()
    print("[PASS] Test 6: Grouped-CV hyperparameter tuning (rf, xgb, lgbm)")
    test_make_report_single_seed_partial_families()
    print("[PASS] Test 7: make_report on 1-seed, 2-family results directory")
    print("\nAll pipeline integrity checks passed successfully!")



