"""E2-E5: open-set evaluation (leave-one-family-out), calibration effect, threshold study, ablation."""
import argparse, os, yaml
import pandas as pd
from src import data as D
from src.experiment import open_set_run

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); a = ap.parse_args()
cfg = yaml.safe_load(open(a.config)); os.makedirs(cfg["results_dir"], exist_ok=True)
df = D.load_clean(cfg); X = D.feature_columns(df, cfg)
R, C, S, PC = [], [], [], []
for fam in cfg["holdout_families"]:
    for seed in cfg["seeds"]:
        print(f"[run] held-out family={fam} seed={seed}", flush=True)
        r, c, sw, pc = open_set_run(df, X, cfg, seed, fam)
        R += r; C.append(c); S += sw; PC += pc
out = cfg["results_dir"]
res = pd.DataFrame(R); res.to_csv(f"{out}/e2_open_set_runs.csv", index=False)
cols = ["auroc", "aupr", "tpr_at_fpr", "known_fpr", "unknown_detection_rate", "unknown_precision", "open_set_f1"]
res.groupby("method")[cols].agg(["mean", "std"]).round(4).to_csv(f"{out}/e2_open_set_summary.csv")
res.groupby(["family", "method"])[cols].agg(["mean", "std"]).round(4).to_csv(f"{out}/e2_per_family.csv")
pd.DataFrame(C).groupby("family")[["T", "ece_raw", "ece_cal"]].agg(["mean", "std"]).round(4).to_csv(f"{out}/e3_calibration.csv")
pd.DataFrame(S).groupby("known_accept_target")[["known_accept_rate", "unknown_detection_rate"]].agg(["mean", "std"]).round(4).to_csv(f"{out}/e4_threshold.csv")
pd.DataFrame(PC).groupby(["family", "method", "cls"]).agg(n=("n", "mean"), flag_rate=("flag_rate", "mean")).round(3).to_csv(f"{out}/e2_per_class.csv")
print(pd.read_csv(f"{out}/e2_open_set_summary.csv", header=[0, 1], index_col=0).xs("mean", axis=1, level=1))
