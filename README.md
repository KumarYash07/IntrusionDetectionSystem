# Open-Set Intrusion Detection for 5G Networks

This repository implements leakage-controlled open-set intrusion detection for 5G networks evaluated on the **5G-NIDD** dataset. It includes:
- **Leakage-controlled group block splitting** across attack classes (preventing adjacent flow leaks).
- **Closed-set model benchmark (E1):** Random Forest (`rf`), Logistic Regression (`lr`), Histogram Gradient Boosting (`hgb`), XGBoost (`xgb`), LightGBM (`lgbm`).
- **Open-set leave-one-family-out evaluation (E2):** Testing detection of novel attack families (`httpflood`, `slowrate`, `synflood`, `scans`).
- **Post-hoc temperature scaling calibration (E3):** Expected Calibration Error (ECE) reduction.
- **Quantile rejection threshold study (E4):** Controlled false alarm rate on known classes.
- **Ablation studies (E5):** Feature dropping (`sTtl`, `dTtl`, `sHops`), scan class merging (`label_map`), and random vs. block splitting leakage analysis.
- **Grouped-CV hyperparameter tuning:** Cross-validation grouped by `__block` strictly within training data.
- **Automated report generator:** Producing markdown summary tables and high-resolution diagnostic figures from saved artifacts without retraining.

---

## 1. Environment Setup

Ensure Python 3.10+ is installed:

```bash
# Clone or navigate to the repository
cd paper1_ids

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

---

## 2. Pipeline Integrity Tests

Run the automated test suite to verify leak-free data splitting, holdout absence, validation-only thresholds, shuffled-label sanity checks, and grouped-CV tuning:

```bash
# Using pytest
pytest -v tests/test_pipeline_integrity.py

# Or run standalone without pytest runner
python tests/test_pipeline_integrity.py
```

### Verified Checks:
1. `test_no_index_or_group_overlap`: Asserts zero index or block group overlap across `train`, `val`, and `test`.
2. `test_heldout_classes_absent_from_train_and_val`: Asserts held-out attack classes never appear in training or validation sets.
3. `test_thresholds_computed_from_validation_known_scores_only`: Verifies decision thresholds and calibration temperatures are invariant to test data perturbations (zero test leakage).
4. `test_shuffled_label_accuracy_near_chance`: Confirms model accuracy collapses to near chance when trained on permuted labels.
5. `test_ablation_configurations_and_mechanics`: Tests feature dropping, class merging via `label_map`, and random splitting.
6. `test_grouped_cv_hyperparameter_tuning`: Confirms `GroupKFold` on training data without fold leakage across `rf`, `xgb`, and `lgbm`.

---

## 3. Fast Verification (Synthetic Data)

To verify the complete pipeline end-to-end in less than 30 seconds using synthetic data:

```bash
# 1. Generate synthetic dataset
python make_synthetic.py

# 2. Run closed-set baselines (E1)
python run_e1.py --config config_synthetic.yaml

# 3. Run open-set evaluation (E2-E5)
python run_e2.py --config config_synthetic.yaml

# 4. Generate report tables and diagnostic plots
python make_report.py --results-dir results_synthetic
```

---

## 4. Hyperparameter Tuning (Grouped-CV)

Perform grouped-CV tuning strictly on training data (`__block` groups):

```bash
# Run grouped-CV for RF, XGBoost, and LightGBM on training data
python tune_hyperparameters.py --config config_quick.yaml --models rf xgb lgbm --folds 3 --seed 0
```
This writes `hyperparameter_tuning.csv` in the results directory with mean $\pm$ std cross-validation macro-F1 for each parameter set and reports validation delta.

---

## 5. Model & Class Balance Selection

To compare all models across class weighting schemes (`none`, `sqrt`, `balanced`) and generate the final configuration:

```bash
# 1. Run quick sweeps across class balance schemes
python run_e1.py --config config_quick.yaml           # class_balance: none
python run_e1.py --config config_quick_sqrt.yaml      # class_balance: sqrt
python run_e1.py --config config_quick_balanced.yaml  # class_balance: balanced

# 2. Automatically select best combination by mean validation macro-F1 only
python select_model.py
```

`select_model.py` reads all E1 runs, ranks candidates strictly on **mean validation macro-F1**, and writes `config_final.yaml` configured with:
- Best model and class balance setting.
- `seeds: [1, 2, 3, 4, 5]`
- `holdout_families: [httpflood, slowrate, synflood, scans]`
- `results_dir: results_final`

---

## 6. Final Full Experiment Run

> **Note:** Do not start this run until you are ready for a full evaluation.

### Exact Reproduction Commands:

```bash
# Step 1: Execute the full leave-one-family-out open-set evaluation across 5 seeds
python run_e2.py --config config_final.yaml

# Step 2: Generate all publication-ready markdown tables and PNG diagnostic figures
python make_report.py --results-dir results_final
```

### Expected Runtime:
- **Hardware:** Modern 8-core CPU (e.g. AMD Ryzen 7 / Intel Core i7) with 16GB RAM.
- **Dataset Size:** 5G-NIDD Combined (~379,000 clean deduplicated flows).
- **Scale:** 4 holdout families (`httpflood`, `slowrate`, `synflood`, `scans`) $\times$ 5 seeds = 20 evaluation runs.
- **Per-run runtime:** ~1.5 to 2.5 minutes per (family, seed) run (training classifier, anomaly detector, Mahalanobis covariance, scoring, calibration, thresholding).
- **Total Expected Runtime:** **30 to 50 minutes**.

---

## 7. Ablation Studies

Run each ablation independently to evaluate specific architectural choices:

### Ablation (a): Drop Router / TTL Artifacts
Tests performance without router-hop features (`sTtl`, `dTtl`, `sHops`):
```bash
python run_e2.py --config config_ablation_drop_ttl_hops.yaml
python make_report.py --results-dir results_ablation_drop_ttl_hops
```

### Ablation (b): Merge Scan Classes into "Scan"
Evaluates whether separating or combining scan types affects open-set detection:
```bash
python run_e2.py --config config_ablation_merge_scans.yaml
python make_report.py --results-dir results_ablation_merge_scans
```

### Ablation (c): Random Split vs. Block Split (Leakage Measurement)
Measures the optimistic bias from flow/session leakage when block grouping is removed:
```bash
python run_e2.py --config config_ablation_random_split.yaml
python make_report.py --results-dir results_ablation_random_split
```

---

## 8. Saved Artifacts and Outputs

Each experiment run directory produces:
- `e1_closed_set_runs.csv`: Closed-set metrics, latency, model sizes.
- `e2_open_set_runs.csv`: Full open-set metrics per seed and family.
- `e2_open_set_summary.csv`: Aggregated method metrics across seeds.
- `e2_per_family.csv`: Family-stratified performance breakdown.
- `e2_per_class.csv`: False alarm flag rate on known classes.
- `e3_calibration.csv`: Temperature scaling parameter $T$ and raw/calibrated ECE.
- `e4_threshold.csv`: Quantile trade-off operating points.
- `scores_{family}_seed{seed}.npz`: Test predictions, continuous detection scores, calibrated confidences, and labels.
- `report.md`: Markdown tables formatted for papers/reports.
- Diagnostic Figures:
  - `fig_roc_curves.png`: Multi-method ROC curves with individual AUROC values.
  - `fig_pr_curves.png`: Precision-Recall curves.
  - `fig_flag_rate_heatmap.png`: Matrix heatmap of per-class false alarm rates.
  - `fig_reliability_diagrams.png`: Dual-panel reliability diagrams before/after calibration.
  - `fig_threshold_tradeoff.png`: Known acceptance vs unknown detection trade-off.
