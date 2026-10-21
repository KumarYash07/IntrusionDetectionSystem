import os
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy.special import logsumexp, softmax
from . import data as D, pipeline as P, metrics as M


def prepare(df, X, cfg, seed, family):
    """Split, encode labels on training classes only, fit preprocessing on training data only."""
    tr, va, te = D.make_split(df, cfg, seed, family)
    D.check_no_overlap(df, cfg, tr, va, te)
    if len(tr) > cfg["max_train_rows"]:
        tr = np.random.default_rng(seed).choice(tr, cfg["max_train_rows"], replace=False)
    lab = df[cfg["label_col"]].values
    le = LabelEncoder().fit(lab[tr])
    known_va = np.isin(lab[va], le.classes_)
    va = va[known_va]
    prep = P.Prep(cfg.get("select_k")).fit(X.values[tr], le.transform(lab[tr]))
    out = dict(le=le, prep=prep, lab=lab)
    for k, ix in (("tr", tr), ("va", va), ("te", te)):
        out[k] = ix
        out["X" + k] = prep.transform(X.values[ix])
    out["ytr"] = le.transform(lab[tr]); out["yva"] = le.transform(lab[va])
    out["is_unk"] = ~np.isin(lab[te], le.classes_)
    out["yte"] = np.where(out["is_unk"], -1, le.transform(np.where(out["is_unk"], le.classes_[0], lab[te])))
    gc = cfg.get("group_col")
    out["gtr"] = df[gc].values[tr] if (gc and gc in df.columns) else None
    return out


def open_set_run(df, X, cfg, seed, family):
    s = prepare(df, X, cfg, seed, family)
    name, K = cfg["model"], len(s["le"].classes_)
    w = P.class_weights(s["ytr"], cfg.get("class_balance"))
    if cfg.get("tune_hyperparameters") and name in P.TUNING_GRIDS:
        model, _, _ = P.tune_model(name, s["Xtr"], s["ytr"], s.get("gtr"), seed, sample_weight=w)
    else:
        model = P.make_model(name, seed).fit(s["Xtr"], s["ytr"], sample_weight=w)
    if name == "rf":
        model.set_params(n_jobs=1)
    zva, zte = P.get_logits(model, name, s["Xva"]), P.get_logits(model, name, s["Xte"])
    T = P.fit_temperature(zva, s["yva"])
    pred = zte.argmax(1)
    kn = ~s["is_unk"]
    cal = dict(family=family, seed=seed, T=T, ece_raw=P.ece(softmax(zte[kn], 1), s["yte"][kn]),
               ece_cal=P.ece(softmax(zte[kn] / T, 1), s["yte"][kn]))

    anom = P.Anomaly(cfg["anomaly"], seed).fit(s["Xtr"])
    ava, ate = anom.score(s["Xva"]), anom.score(s["Xte"])
    mah = P.Mahalanobis().fit(s["Xtr"], s["ytr"])
    q = cfg["known_accept_rate"]

    conf_va, conf_te = softmax(zva / T, 1).max(1), softmax(zte / T, 1).max(1)
    rej_va, rej_te = 1 - conf_va, 1 - conf_te                      # calibrated rejection score
    mva, mte = mah.score(s["Xva"]), mah.score(s["Xte"])

    def fuse(v1, t1, v2, t2):
        """AND-fusion (both signals must be high): min of percentile ranks vs validation-known scores."""
        pv = np.minimum(P.rank_vs(v1, v1), P.rank_vs(v2, v2))
        pt = np.minimum(P.rank_vs(v1, t1), P.rank_vs(v2, t2))
        return pv, pt

    scores = {  # name -> (val, test); higher = more unknown
        "msp_uncalibrated": (1 - softmax(zva, 1).max(1), 1 - softmax(zte, 1).max(1)),
        "energy": (-logsumexp(zva, 1), -logsumexp(zte, 1)),
        "mahalanobis": (mva, mte),
        "anomaly_only": (ava, ate),
        "rejection_only": (rej_va, rej_te),
        "proposed": fuse(rej_va, rej_te, ava, ate),            # calibrated rejection AND anomaly stage
        "proposed_maha": fuse(rej_va, rej_te, mva, mte),       # calibrated rejection AND Mahalanobis
    }
    rows, pc = [], []
    flags, thrs = {}, {}
    for m, (v, t) in scores.items():
        thr = float(np.quantile(v, q))  # same rule for every method: known-FPR budget (1-q) on validation known data
        thrs[m] = thr
        mass_gt = float((v > thr).mean())
        mass_eq = float((v == thr).mean())
        p_tie = float(np.clip((1 - q - mass_gt) / mass_eq, 0.0, 1.0)) if mass_eq > 0 else 0.0
        val_rng = np.random.default_rng(seed)
        val_flagged = (v > thr) | ((v == thr) & (val_rng.random(len(v)) < p_tie))
        val_known_fpr = float(val_flagged.mean())

        rng = np.random.default_rng(seed)
        flagged = (t > thr) | ((t == thr) & (rng.random(len(t)) < p_tie))
        flags[m] = flagged

        row = dict(family=family, seed=seed, method=m)
        row.update(M.detection_metrics(s["is_unk"], t, cfg["fpr_target"]))
        row.update(M.decision_metrics(s["yte"], pred, s["is_unk"], flagged, K))
        row.update(val_known_fpr=val_known_fpr, tie_flag_prob=p_tie)
        rows.append(row)
        for k, cname in enumerate(s["le"].classes_):  # per-class flag rate on KNOWN test samples (diagnostic)
            mk = kn & (s["yte"] == k)
            if mk.any():
                pc.append(dict(family=family, seed=seed, method=m, cls=cname, n=int(mk.sum()), flag_rate=float(flagged[mk].mean())))
    # ablation baseline: classifier only (never flags unknown)
    row = dict(family=family, seed=seed, method="classifier_only", auroc=np.nan, aupr=np.nan, tpr_at_fpr=0.0)
    row.update(M.decision_metrics(s["yte"], pred, s["is_unk"], np.zeros(len(pred), bool), K))
    row.update(val_known_fpr=0.0, tie_flag_prob=0.0)
    rows.append(row)
    # threshold study (E4): rejection-only trade-off
    sweep = []
    for a in (0.80, 0.85, 0.90, 0.95, 0.99):
        fl = rej_te > np.quantile(rej_va, a)
        sweep.append(dict(family=family, seed=seed, known_accept_target=a,
                          known_accept_rate=1 - fl[kn].mean(), unknown_detection_rate=fl[s["is_unk"]].mean()))

    # Save per-run test scores (npz) so ROC/PR curves and diagnostic figures can be drawn later
    if cfg.get("results_dir"):
        os.makedirs(cfg["results_dir"], exist_ok=True)
        npz_path = os.path.join(cfg["results_dir"], f"scores_{family}_seed{seed}.npz")
        npz_dict = {
            "family": np.array(family),
            "seed": np.int64(seed),
            "is_unk": s["is_unk"].astype(bool),
            "yte": s["yte"].astype(np.int64),
            "yva": s["yva"].astype(np.int64),
            "pred": pred.astype(np.int64),
            "classes": np.array(s["le"].classes_),
            "T": np.float64(T),
            "conf_uncal": softmax(zte, 1).max(1).astype(np.float32),
            "conf_cal": softmax(zte / T, 1).max(1).astype(np.float32),
        }
        for m, (v, t) in scores.items():
            npz_dict[f"score_{m}"] = t.astype(np.float32)
            npz_dict[f"val_score_{m}"] = v.astype(np.float32)
            thr = float(thrs.get(m, np.quantile(v, q)))
            npz_dict[f"thr_{m}"] = np.float32(thr)
            npz_dict[f"flagged_{m}"] = flags.get(m, t > thr).astype(bool)
        np.savez_compressed(npz_path, **npz_dict)

    return rows, cal, sweep, pc

