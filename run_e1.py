"""E1: closed-set baselines (+ majority and shuffled-label sanity checks, latency, model size)."""
import argparse, os, pickle, time, yaml
import numpy as np, pandas as pd
from src import data as D, pipeline as P, metrics as M
from src.experiment import prepare

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); a = ap.parse_args()
cfg = yaml.safe_load(open(a.config)); os.makedirs(cfg["results_dir"], exist_ok=True)
df = D.load_clean(cfg); X = D.feature_columns(df, cfg)
print(df[cfg["label_col"]].value_counts())
rows = []
for seed in cfg["seeds"]:
    s = prepare(df, X, cfg, seed, None)
    kn = ~s["is_unk"]
    maj = np.bincount(s["ytr"]).argmax()
    rows.append(dict(model="majority_baseline", seed=seed, **M.closed_metrics(s["yte"][kn], np.full(kn.sum(), maj))))
    for name in cfg["models"]:
        try:
            m = P.make_model(name, seed)
        except ImportError:
            print(f"[skip] {name} not installed"); continue
        t0 = time.time(); m.fit(s["Xtr"], s["ytr"], sample_weight=P.class_weights(s["ytr"], cfg.get("class_balance"))); fit_s = time.time() - t0
        val_f1 = M.closed_metrics(s["yva"], m.predict(s["Xva"]))["macro_f1"]
        t0 = time.time(); pred = m.predict(s["Xte"]); batch_ms = (time.time() - t0) / len(pred) * 1000
        sub = s["Xte"][:200]; t0 = time.time()
        for r in sub: m.predict(r[None, :])
        single_ms = (time.time() - t0) / len(sub) * 1000
        rows.append(dict(model=name, seed=seed, val_macro_f1=val_f1, fit_seconds=fit_s,
                         latency_ms_batch=batch_ms, latency_ms_single=single_ms,
                         model_size_mb=len(pickle.dumps(m)) / 1e6, **M.closed_metrics(s["yte"][kn], pred[kn])))
    # sanity check: shuffled labels must give near-chance accuracy
    ysh = np.random.default_rng(seed).permutation(s["ytr"])
    m = P.make_model("rf", seed).fit(s["Xtr"], ysh)
    rows.append(dict(model="shuffled_label_rf", seed=seed, **M.closed_metrics(s["yte"][kn], m.predict(s["Xte"])[kn])))
res = pd.DataFrame(rows); res.to_csv(f"{cfg['results_dir']}/e1_closed_set_runs.csv", index=False)
summ = res.groupby("model").agg(["mean", "std"]).round(4)
summ.to_csv(f"{cfg['results_dir']}/e1_closed_set_summary.csv"); print(summ[["macro_f1", "accuracy"]])
