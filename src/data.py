import warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


def load_clean(cfg):
    """Load CSV, drop identifier columns, remove duplicate rows, encode categoricals."""
    df = pd.read_csv(cfg["data_path"], low_memory=False)
    if cfg.get("label_map"):
        df[cfg["label_col"]] = df[cfg["label_col"]].replace(cfg["label_map"])
    if cfg.get("order_col") and cfg.get("blocks_per_class"):
        # Data is concatenated attack by attack and has no session id. Within each class, cut the row order
        # into contiguous blocks; whole blocks go to train/val/test (reduces leakage between neighbouring flows).
        df = df.sort_values(cfg["order_col"], kind="stable").reset_index(drop=True)
        lab = df[cfg["label_col"]].astype(str)
        pos = df.groupby(cfg["label_col"]).cumcount()
        size = df.groupby(cfg["label_col"])[cfg["label_col"]].transform("size")
        df["__block"] = lab + "_" + (pos * cfg["blocks_per_class"] // size).astype(str)
    df = df.drop(columns=[c for c in cfg.get("drop_cols", []) if c in df.columns])
    skip = {cfg.get("group_col"), cfg.get("time_col")}
    dedup_cols = [c for c in df.columns if c not in skip]
    n0 = len(df)
    df = df.drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    print(f"[data] rows {n0} -> {len(df)} after removing duplicates")
    return df


def feature_columns(df, cfg):
    excl = {cfg["label_col"], cfg.get("group_col"), cfg.get("time_col")}
    cols = [c for c in df.columns if c not in excl]
    X = df[cols].copy()
    for c in X.columns:  # categorical -> integer codes (no label information used)
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype("category").cat.codes
    X = X.replace([np.inf, -np.inf], np.nan)
    for c in [c for c in X.columns if X[c].isna().any()]:
        X[c + "__isna"] = X[c].isna().astype("float32")
    X = X.fillna(X.median(numeric_only=True))
    return X.astype("float32")


def family_classes(cfg, family):
    return set(cfg["families"][family]) if family else set()


def make_split(df, cfg, seed, holdout_family=None):
    """Session/time-wise split. Held-out family is removed from train and val, kept in test."""
    n = len(df)
    idx = np.arange(n)
    gc, tc = cfg.get("group_col"), cfg.get("time_col")
    if gc:
        g = df[gc].values
        s1 = GroupShuffleSplit(1, test_size=cfg["test_size"], random_state=seed)
        trval, te = next(s1.split(idx, groups=g))
        rel = cfg["val_size"] / (1 - cfg["test_size"])
        s2 = GroupShuffleSplit(1, test_size=rel, random_state=seed)
        a, b = next(s2.split(trval, groups=g[trval]))
        tr, va = trval[a], trval[b]
    elif tc:
        order = np.argsort(df[tc].values, kind="stable")
        n_te = int(n * cfg["test_size"]); n_va = int(n * cfg["val_size"])
        tr, va, te = order[: n - n_te - n_va], order[n - n_te - n_va: n - n_te], order[n - n_te:]
    else:
        warnings.warn("No group_col/time_col: using stratified random split. Near-duplicate flows may leak. Not recommended.")
        trval, te = train_test_split(idx, test_size=cfg["test_size"], stratify=df[cfg["label_col"]], random_state=seed)
        rel = cfg["val_size"] / (1 - cfg["test_size"])
        tr, va = train_test_split(trval, test_size=rel, stratify=df[cfg["label_col"]].values[trval], random_state=seed)
    if holdout_family:
        unk = family_classes(cfg, holdout_family)
        lab = df[cfg["label_col"]].values
        tr = tr[~np.isin(lab[tr], list(unk))]
        va = va[~np.isin(lab[va], list(unk))]
    return tr, va, te


def check_no_overlap(df, cfg, tr, va, te):
    """Correctness check: no index and (if group_col) no group shared between sets."""
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    gc = cfg.get("group_col")
    if gc:
        g = df[gc].values
        assert not (set(g[tr]) & set(g[te])) and not (set(g[va]) & set(g[te])) and not (set(g[tr]) & set(g[va]))
