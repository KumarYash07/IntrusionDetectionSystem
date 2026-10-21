"""make_report.py: Generate comprehensive markdown tables and PNG diagnostic figures from saved experiment files only.

Inputs (from results directory):
- e2_open_set_runs.csv, e2_per_family.csv, e2_per_class.csv, e3_calibration.csv, e4_threshold.csv
- scores_{family}_seed{seed}.npz

Outputs:
- report.md (with formatted GitHub markdown tables)
- fig_roc_curves.png
- fig_pr_curves.png
- fig_flag_rate_heatmap.png
- fig_reliability_diagrams.png
- fig_threshold_tradeoff.png
"""
import argparse
import glob
import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def parse_args():
    parser = argparse.ArgumentParser(description="Generate markdown report and figures from saved results.")
    parser.add_argument("--results-dir", default="results_final", help="Path to experiment results directory")
    parser.add_argument("--out-dir", default=None, help="Directory to save report and plots (defaults to results-dir)")
    parser.add_argument("--model", default=None, help="Model name (e.g. rf, hgb, xgb, lgbm). If not provided, inferred automatically from config.")
    return parser.parse_args()


def detect_model(results_dir, specified_model=None):
    if specified_model:
        return specified_model.lower().strip()
    res_norm = os.path.normpath(results_dir).replace("\\", "/").rstrip("/")
    res_base = os.path.basename(res_norm)
    for yf in glob.glob("*.yaml"):
        try:
            with open(yf, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            if cfg and isinstance(cfg, dict):
                r = cfg.get("results_dir")
                if r and os.path.normpath(r).replace("\\", "/").rstrip("/") in (res_norm, res_base):
                    m = cfg.get("model")
                    if m:
                        return str(m).lower().strip()
        except Exception:
            pass
    return None


def format_mean_std(mean, std, digits=4):
    if pd.isna(mean):
        return "—"
    if pd.isna(std):
        return f"{mean:.{digits}f} ± n/a"
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def build_markdown_tables(results_dir, out_dir, model=None):
    md_lines = []
    md_lines.append(f"# Experiment Evaluation Report\n")
    md_lines.append(f"Source Results Directory: `{results_dir}`\n")
    if model:
        md_lines.append(f"Model: `{model}`\n")

    # 1. Overall Open-Set Metrics Table
    runs_file = os.path.join(results_dir, "e2_open_set_runs.csv")
    if os.path.exists(runs_file):
        df_runs = pd.read_csv(runs_file)
        metric_cols = ["auroc", "aupr", "unknown_precision", "unknown_detection_rate", "known_fpr", "open_set_f1"]
        avail_cols = [c for c in metric_cols if c in df_runs.columns]
        
        summ = df_runs.groupby("method")[avail_cols].agg(["mean", "std"])
        
        md_lines.append("## 1. Overall Open-Set Detection Performance\n")
        md_lines.append("Averaged across all holdout families and random seeds.\n")
        
        header = ["Method"] + [col.replace("_", " ").title() for col in avail_cols]
        md_lines.append("| " + " | ".join(header) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        
        for method, row in summ.iterrows():
            row_vals = [f"**{method}**"]
            if method == "energy" and model in ["rf", "hgb"]:
                row_vals.extend(["n/a: no logits"] * len(avail_cols))
            else:
                for col in avail_cols:
                    row_vals.append(format_mean_std(row[(col, "mean")], row[(col, "std")]))
            md_lines.append("| " + " | ".join(row_vals) + " |")
        md_lines.append("\n")

        # 2. Per-Family Breakdown Table
        md_lines.append("## 2. Per-Family Open-Set Detection Breakdown\n")
        fam_summ = df_runs.groupby(["family", "method"])[avail_cols].agg(["mean", "std"])
        fam_header = ["Family", "Method"] + [col.replace("_", " ").title() for col in avail_cols]
        md_lines.append("| " + " | ".join(fam_header) + " |")
        md_lines.append("| " + " | ".join(["---"] * len(fam_header)) + " |")
        
        for (fam, method), row in fam_summ.iterrows():
            row_vals = [fam, f"`{method}`"]
            if method == "energy" and model in ["rf", "hgb"]:
                row_vals.extend(["n/a: no logits"] * len(avail_cols))
            else:
                for col in avail_cols:
                    row_vals.append(format_mean_std(row[(col, "mean")], row[(col, "std")]))
            md_lines.append("| " + " | ".join(row_vals) + " |")
        md_lines.append("\n")

        # 3. Unknown Precision Focus Table
        if "unknown_precision" in df_runs.columns:
            md_lines.append("## 3. Unknown Precision by Method and Family\n")
            up_pivot_mean = df_runs.pivot_table(index="method", columns="family", values="unknown_precision", aggfunc="mean")
            up_pivot_std = df_runs.pivot_table(index="method", columns="family", values="unknown_precision", aggfunc="std", dropna=False)
            
            fams = list(up_pivot_mean.columns)
            up_header = ["Method"] + [f"{fam} (mean ± std)" for fam in fams] + ["Overall Mean"]
            md_lines.append("| " + " | ".join(up_header) + " |")
            md_lines.append("| " + " | ".join(["---"] * len(up_header)) + " |")
            
            for m in up_pivot_mean.index:
                row_vals = [f"**{m}**"]
                if m == "energy" and model in ["rf", "hgb"]:
                    row_vals.extend(["n/a: no logits"] * (len(fams) + 1))
                else:
                    for f in fams:
                        m_val = up_pivot_mean.loc[m, f] if (m in up_pivot_mean.index and f in up_pivot_mean.columns) else np.nan
                        s_val = up_pivot_std.loc[m, f] if (m in up_pivot_std.index and f in up_pivot_std.columns) else np.nan
                        row_vals.append(format_mean_std(m_val, s_val))
                    overall_series = df_runs[df_runs["method"] == m]["unknown_precision"]
                    overall_m = overall_series.mean() if not overall_series.empty else np.nan
                    overall_s = overall_series.std() if len(overall_series) > 1 else np.nan
                    row_vals.append(format_mean_std(overall_m, overall_s))
                md_lines.append("| " + " | ".join(row_vals) + " |")
            md_lines.append("\n")

    # 4. Calibration (ECE & Temperature) Table
    cal_file = os.path.join(results_dir, "e3_calibration.csv")
    if os.path.exists(cal_file):
        md_lines.append("## 4. Temperature Calibration Effect\n")
        md_lines.append("Temperature scaling parameter $T$ and Expected Calibration Error (ECE) before/after scaling.\n")
        try:
            df_cal = pd.read_csv(cal_file, header=[0, 1], index_col=0)
            cal_header = ["Family", "Temperature T", "ECE Raw (Before)", "ECE Calibrated (After)"]
            md_lines.append("| " + " | ".join(cal_header) + " |")
            md_lines.append("| " + " | ".join(["---"] * len(cal_header)) + " |")
            for fam, row in df_cal.iterrows():
                t_str = format_mean_std(row[("T", "mean")], row[("T", "std")])
                raw_str = format_mean_std(row[("ece_raw", "mean")], row[("ece_raw", "std")])
                cal_str = format_mean_std(row[("ece_cal", "mean")], row[("ece_cal", "std")])
                md_lines.append(f"| **{fam}** | {t_str} | {raw_str} | {cal_str} |")
        except Exception:
            cal_plain = pd.read_csv(cal_file)
            md_lines.append(cal_plain.to_markdown(index=False))
        md_lines.append("\n")

    # 5. Threshold Trade-off Table
    thr_file = os.path.join(results_dir, "e4_threshold.csv")
    if os.path.exists(thr_file):
        md_lines.append("## 5. Threshold Rejection Trade-Off (E4)\n")
        md_lines.append("Empirical known acceptance rate and unknown detection rate across rejection thresholds.\n")
        try:
            df_thr = pd.read_csv(thr_file, header=[0, 1], index_col=0)
            thr_header = ["Target Known Accept", "Empirical Known Accept Rate", "Unknown Detection Rate"]
            md_lines.append("| " + " | ".join(thr_header) + " |")
            md_lines.append("| " + " | ".join(["---"] * len(thr_header)) + " |")
            for tgt, row in df_thr.iterrows():
                ka_str = format_mean_std(row[("known_accept_rate", "mean")], row[("known_accept_rate", "std")])
                ud_str = format_mean_std(row[("unknown_detection_rate", "mean")], row[("unknown_detection_rate", "std")])
                md_lines.append(f"| **{tgt}** | {ka_str} | {ud_str} |")
        except Exception:
            thr_plain = pd.read_csv(thr_file)
            md_lines.append(thr_plain.to_markdown(index=False))
        md_lines.append("\n")

    md_lines.append("## 6. Diagnostic Figures\n")
    md_lines.append("- ![ROC Curves](fig_roc_curves.png)\n")
    md_lines.append("- ![PR Curves](fig_pr_curves.png)\n")
    md_lines.append("- ![Per-Class Flag-Rate Heatmap](fig_flag_rate_heatmap.png)\n")
    md_lines.append("- ![Reliability Diagrams](fig_reliability_diagrams.png)\n")
    md_lines.append("- ![Threshold Trade-off](fig_threshold_tradeoff.png)\n")

    report_text = "\n".join(md_lines)
    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[report] Saved markdown report to: {report_path}")


def plot_flag_rate_heatmap(results_dir, out_dir, model=None):
    pc_file = os.path.join(results_dir, "e2_per_class.csv")
    if not os.path.exists(pc_file):
        print(f"[skip] {pc_file} not found for heatmap")
        return
    df = pd.read_csv(pc_file)
    if "flag_rate" not in df.columns:
        return
    if model in ["rf", "hgb"]:
        df = df[df["method"] != "energy"]
    # Pivot: rows = method, cols = known class, values = mean flag rate
    pivot = df.pivot_table(index="method", columns="cls", values="flag_rate", aggfunc="mean").fillna(0.0)
    
    # Calculate mean test-flow count per class
    counts = df.groupby("cls")["n"].mean() if "n" in df.columns else {}
    col_labels = []
    for c in pivot.columns:
        if c in counts and pd.notna(counts[c]):
            cnt_val = counts[c]
            col_labels.append(f"{c}\n(n={int(round(cnt_val)):,})")
        else:
            col_labels.append(str(c))
            
    plt.figure(figsize=(max(8.5, len(pivot.columns) * 1.3), max(5.5, len(pivot.index) * 0.65)))
    ax = plt.gca()
    cax = ax.matshow(pivot.values, cmap="YlOrRd", vmin=0.0, vmax=max(0.2, pivot.values.max()))
    plt.colorbar(cax, label="False Alarm Flag Rate on Known Classes")
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(col_labels, rotation=45, ha="left", fontsize=8.5)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    
    # Annotate values inside cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val > (cax.norm.vmax * 0.6) else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontsize=8)
            
    plt.title("Per-Class False Alarm Flag-Rate on Known Classes\n(with Mean Flow Count n per Class)", pad=35, fontsize=11, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_flag_rate_heatmap.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot] Saved flag rate heatmap: {out_path}")


def plot_reliability_diagrams(npz_files, out_dir, results_dir=None, bins=10):
    if not npz_files:
        print("[skip] No npz files found for reliability diagrams")
        return
    
    uncal_confs, cal_confs, corrects = [], [], []
    for f in npz_files:
        data = np.load(f)
        kn = ~data["is_unk"]
        if not kn.any():
            continue
        yte_kn = data["yte"][kn]
        pred_kn = data["pred"][kn]
        correct = (pred_kn == yte_kn)
        
        uncal_confs.append(data["conf_uncal"][kn])
        cal_confs.append(data["conf_cal"][kn])
        corrects.append(correct)
        
    if not corrects:
        return
    
    uncal_conf = np.concatenate(uncal_confs)
    cal_conf = np.concatenate(cal_confs)
    y_corr = np.concatenate(corrects)

    # Check e3_calibration.csv for per-run mean ± std ECE
    cal_summary = None
    if results_dir:
        cal_path = os.path.join(results_dir, "e3_calibration.csv")
        if os.path.exists(cal_path):
            try:
                df_cal = pd.read_csv(cal_path, header=[0, 1], index_col=0)
                raw_mean = df_cal[("ece_raw", "mean")].mean()
                raw_std = df_cal[("ece_raw", "std")].mean()
                cal_mean = df_cal[("ece_cal", "mean")].mean()
                cal_std = df_cal[("ece_cal", "std")].mean()
                cal_summary = (raw_mean, raw_std, cal_mean, cal_std)
            except Exception:
                pass
    
    def compute_diagram(conf, corr, n_bins=10):
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_acc, bin_conf, bin_counts = [], [], []
        ece = 0.0
        n_total = len(conf)
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (conf > lo) & (conf <= hi)
            if mask.any():
                acc_val = corr[mask].mean()
                conf_val = conf[mask].mean()
                bin_acc.append(acc_val)
                bin_conf.append(conf_val)
                bin_counts.append(mask.sum())
                ece += (mask.sum() / n_total) * abs(acc_val - conf_val)
            else:
                bin_acc.append(np.nan)
                bin_conf.append((lo + hi) / 2.0)
                bin_counts.append(0)
        return edges, np.array(bin_acc), np.array(bin_conf), np.array(bin_counts), ece

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    
    titles_meta = [
        ("Before Scaling (Raw Softmax)", 0),
        ("After Temperature Scaling", 2)
    ]
    
    for ax, conf, (title_base, cal_idx) in zip(axes, [uncal_conf, cal_conf], titles_meta):
        edges, b_acc, b_conf, b_cnt, ece = compute_diagram(conf, y_corr, bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        width = 1.0 / bins * 0.8
        
        valid = ~np.isnan(b_acc)
        ax.bar(centers[valid], b_acc[valid], width=width, color="#3498db", alpha=0.7, edgecolor="black", label="Outputs (Empirical Accuracy)")
        ax.plot([0, 1], [0, 1], "--", color="red", linewidth=1.5, label="Perfect Calibration")

        # Show sample count per bin on top of valid bars
        for c, a, cnt in zip(centers[valid], b_acc[valid], b_cnt[valid]):
            if cnt > 0:
                ax.text(c, a + 0.02, f"n={cnt:,}", ha="center", va="bottom", fontsize=7.5, rotation=35)

        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.25])
        ax.set_xlabel("Confidence", fontsize=10)
        ax.set_ylabel("Accuracy", fontsize=10)
        
        if cal_summary is not None:
            mean_v, std_v = cal_summary[cal_idx], cal_summary[cal_idx + 1]
            std_str = f" ± {std_v:.4f}" if pd.notna(std_v) else ""
            title_text = f"{title_base} [Pooled Across Runs]\nPooled ECE = {ece:.4f} | Mean Per-Run ECE = {mean_v:.4f}{std_str}"
        else:
            title_text = f"{title_base} [Pooled Across Runs]\nPooled ECE = {ece:.4f}"
            
        ax.set_title(title_text, fontsize=10, fontweight="bold", pad=8)
        ax.grid(True, linestyle=":", alpha=0.6)
        # Move legend outside the bars below the axis
        ax.legend(bbox_to_anchor=(0.5, -0.22), loc="upper center", ncol=2, fontsize=8.5, frameon=True)

    plt.suptitle("Reliability Diagrams (Pooled Across All Test Runs)", fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    out_path = os.path.join(out_dir, "fig_reliability_diagrams.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot] Saved reliability diagrams: {out_path}")


def plot_roc_and_pr_curves(npz_files, out_dir, results_dir=None, model=None):
    if not npz_files:
        print("[skip] No npz files found for ROC/PR curves")
        return

    # Load e2_per_family.csv if available to get exact family means for legends
    fam_df = None
    if results_dir:
        fam_csv = os.path.join(results_dir, "e2_per_family.csv")
        if os.path.exists(fam_csv):
            try:
                fam_df = pd.read_csv(fam_csv, header=[0, 1], index_col=[0, 1])
            except Exception:
                fam_df = None

    # Group npz files by family
    fam_files = {}
    for f in npz_files:
        try:
            d = np.load(f)
            fam = str(d["family"])
        except Exception:
            fam = os.path.basename(f).split("_")[1]
        fam_files.setdefault(fam, []).append(f)

    fams = sorted(list(fam_files.keys()))
    sample = np.load(npz_files[0])
    raw_methods = [k[6:] for k in sample.files if k.startswith("score_")]
    if model in ["rf", "hgb"]:
        raw_methods = [m for m in raw_methods if m != "energy"]

    preferred_order = ["proposed", "proposed_maha", "rejection_only", "mahalanobis", "energy", "msp_uncalibrated", "anomaly_only"]
    methods = [m for m in preferred_order if m in raw_methods] + [m for m in raw_methods if m not in preferred_order]

    # Color palette
    method_colors = {
        "proposed": "#1f77b4",        # blue
        "proposed_maha": "#2ca02c",   # green
        "rejection_only": "#ff7f0e",  # orange
        "mahalanobis": "#9467bd",     # purple
        "energy": "#d62728",          # red
        "msp_uncalibrated": "#8c564b",# brown
        "anomaly_only": "#7f7f7f",    # gray
    }

    n_fams = len(fams)
    ncols = 2 if n_fams > 1 else 1
    nrows = (n_fams + ncols - 1) // ncols

    # ---------------- 1. ROC Curves (4 Panels) ----------------
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    common_fpr = np.linspace(0.0, 1.0, 201)

    for idx, fam in enumerate(fams):
        ax = axes_flat[idx]
        files = fam_files[fam]
        fam_data = [np.load(f) for f in files]

        for m in methods:
            tpr_list = []
            for d in fam_data:
                y_true = d["is_unk"].astype(int)
                if f"score_{m}" not in d:
                    continue
                sc = np.nan_to_num(d[f"score_{m}"], nan=0.0, posinf=1e6, neginf=-1e6)
                fpr, tpr, _ = roc_curve(y_true, sc)
                interp_tpr = np.interp(common_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                interp_tpr[-1] = 1.0
                tpr_list.append(interp_tpr)

            if not tpr_list:
                continue

            mean_tpr = np.mean(tpr_list, axis=0)
            std_tpr = np.std(tpr_list, axis=0)

            # Retrieve exact mean AUROC from e2_per_family.csv if available
            val_auroc = None
            if fam_df is not None and (fam, m) in fam_df.index:
                val_auroc = fam_df.loc[(fam, m), ("auroc", "mean")]
            if val_auroc is None or pd.isna(val_auroc):
                val_auroc = auc(common_fpr, mean_tpr)

            lw = 2.4 if "proposed" in m else 1.5
            ls = "-" if "proposed" in m else ("-." if "rejection" in m else "--")
            color = method_colors.get(m, None)

            ax.plot(common_fpr, mean_tpr, label=f"{m} (AUROC = {val_auroc:.4f})", linewidth=lw, linestyle=ls, color=color)
            if len(tpr_list) > 1 and np.max(std_tpr) > 0.0001:
                ax.fill_between(common_fpr, np.clip(mean_tpr - std_tpr, 0, 1), np.clip(mean_tpr + std_tpr, 0, 1), color=color, alpha=0.15)

        ax.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Chance")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel("False Positive Rate (Knowns Flagged)", fontsize=9.5)
        ax.set_ylabel("True Positive Rate (Unknowns Detected)", fontsize=9.5)
        ax.set_title(f"Held-out Family: {fam}", fontsize=11, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

    # Hide any unused subplots
    for ax in axes_flat[n_fams:]:
        ax.set_visible(False)

    fig.suptitle("Receiver Operating Characteristic (ROC) Curves by Held-Out Family", fontsize=13, fontweight="bold", y=0.99)
    caption_text = (
        "Note: Curves show mean ± std across seeds on a common grid for each held-out family. Legend values match per-family means in e2_per_family.csv.\n"
        "(The pooled alternative concatenates raw test scores across all families into a single curve, but distorts evaluation due to cross-family prevalence and score shift)."
    )
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=8.5, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    roc_out = os.path.join(out_dir, "fig_roc_curves.png")
    plt.savefig(roc_out, dpi=200)
    plt.close()
    print(f"[plot] Saved ROC curves: {roc_out}")

    # ---------------- 2. PR Curves (4 Panels) ----------------
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    common_rec = np.linspace(0.0, 1.0, 201)

    for idx, fam in enumerate(fams):
        ax = axes_flat[idx]
        files = fam_files[fam]
        fam_data = [np.load(f) for f in files]

        for m in methods:
            prec_list = []
            for d in fam_data:
                y_true = d["is_unk"].astype(int)
                if f"score_{m}" not in d:
                    continue
                sc = np.nan_to_num(d[f"score_{m}"], nan=0.0, posinf=1e6, neginf=-1e6)
                prec, rec, _ = precision_recall_curve(y_true, sc)
                interp_prec = np.interp(common_rec, rec[::-1], prec[::-1])
                prec_list.append(interp_prec)

            if not prec_list:
                continue

            mean_prec = np.mean(prec_list, axis=0)
            std_prec = np.std(prec_list, axis=0)

            # Retrieve exact mean AUPR from e2_per_family.csv if available
            val_aupr = None
            if fam_df is not None and (fam, m) in fam_df.index:
                val_aupr = fam_df.loc[(fam, m), ("aupr", "mean")]
            if val_aupr is None or pd.isna(val_aupr):
                val_aupr = auc(common_rec, mean_prec)

            lw = 2.4 if "proposed" in m else 1.5
            ls = "-" if "proposed" in m else ("-." if "rejection" in m else "--")
            color = method_colors.get(m, None)

            ax.plot(common_rec, mean_prec, label=f"{m} (AUPR = {val_aupr:.4f})", linewidth=lw, linestyle=ls, color=color)
            if len(prec_list) > 1 and np.max(std_prec) > 0.0001:
                ax.fill_between(common_rec, np.clip(mean_prec - std_prec, 0, 1), np.clip(mean_prec + std_prec, 0, 1), color=color, alpha=0.15)

        mean_prev = np.mean([d["is_unk"].mean() for d in fam_data])
        ax.plot([0, 1], [mean_prev, mean_prev], "k:", alpha=0.5, label=f"Prevalence ({mean_prev:.2f})")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.02])
        ax.set_xlabel("Recall (TPR)", fontsize=9.5)
        ax.set_ylabel("Precision (Unknown Precision)", fontsize=9.5)
        ax.set_title(f"Held-out Family: {fam}", fontsize=11, fontweight="bold")
        ax.legend(loc="lower left", fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.6)

    for ax in axes_flat[n_fams:]:
        ax.set_visible(False)

    fig.suptitle("Precision-Recall (PR) Curves by Held-Out Family", fontsize=13, fontweight="bold", y=0.99)
    fig.text(0.5, 0.01, caption_text, ha="center", fontsize=8.5, style="italic")
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    pr_out = os.path.join(out_dir, "fig_pr_curves.png")
    plt.savefig(pr_out, dpi=200)
    plt.close()
    print(f"[plot] Saved PR curves: {pr_out}")


def plot_threshold_tradeoff(results_dir, out_dir):
    thr_file = os.path.join(results_dir, "e4_threshold.csv")
    if not os.path.exists(thr_file):
        print(f"[skip] {thr_file} not found for threshold plot")
        return
    try:
        df = pd.read_csv(thr_file, header=[0, 1], index_col=0)
        x = df[("known_accept_rate", "mean")].astype(float)
        x_err = df[("known_accept_rate", "std")].fillna(0.0).astype(float)
        y = df[("unknown_detection_rate", "mean")].astype(float)
        y_err = df[("unknown_detection_rate", "std")].fillna(0.0).astype(float)
        targets = [str(idx) for idx in df.index]
    except Exception:
        df = pd.read_csv(thr_file)
        x = df["known_accept_rate"].astype(float)
        x_err = [0.0] * len(df)
        y = df["unknown_detection_rate"].astype(float)
        y_err = [0.0] * len(df)
        targets = [str(t) for t in df.get("known_accept_target", range(len(df)))]

    plt.figure(figsize=(7.5, 5))
    plt.errorbar(
        x, y,
        xerr=x_err, yerr=y_err,
        fmt="o-", color="#2c3e50", ecolor="#e74c3c", elinewidth=1.5,
        capsize=4, capthick=1.5, linewidth=2, markersize=7,
        label="Calibrated Rejection (Mean ± Std)"
    )
    for t, xi, yi in zip(targets, x, y):
        plt.annotate(f"Target: {t}", (xi, yi), textcoords="offset points", xytext=(8, -4), fontsize=8.5, fontweight="semibold")
    plt.xlabel("Empirical Known Acceptance Rate", fontsize=11)
    plt.ylabel("Unknown Detection Rate", fontsize=11)
    plt.title("Threshold Trade-Off: Calibrated Rejection Only\n(Known Acceptance vs Unknown Detection)", fontsize=11, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_threshold_tradeoff.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[plot] Saved threshold trade-off: {out_path}")


def main():
    args = parse_args()
    res_dir = args.results_dir
    out_dir = args.out_dir or res_dir
    os.makedirs(out_dir, exist_ok=True)
    
    model = detect_model(res_dir, args.model)
    print(f"Generating report from directory: {res_dir}")
    if model:
        print(f"Model detected: {model}")
    print(f"Outputs will be saved to: {out_dir}")
    
    # 1. Build Markdown Tables
    build_markdown_tables(res_dir, out_dir, model=model)
    
    # 2. Per-class flag-rate heatmap
    plot_flag_rate_heatmap(res_dir, out_dir, model=model)
    
    # 3. NPZ files for ROC, PR, and Reliability diagrams
    npz_pattern = os.path.join(res_dir, "scores_*.npz")
    npz_files = glob.glob(npz_pattern)
    print(f"Found {len(npz_files)} npz files matching '{npz_pattern}'")
    
    plot_reliability_diagrams(npz_files, out_dir, results_dir=res_dir)
    plot_roc_and_pr_curves(npz_files, out_dir, results_dir=res_dir, model=model)
    
    # 4. Threshold trade-off curve
    plot_threshold_tradeoff(res_dir, out_dir)
    
    print("Report and all figures generated successfully.")


if __name__ == "__main__":
    main()
