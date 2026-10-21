import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.ensemble import RandomForestClassifier, IsolationForest, HistGradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class Prep:
    """Fit on training data only: drop constant columns, scale, optional top-k selection."""
    def __init__(self, k=None):
        self.k = k

    def fit(self, X, y):
        self.vt = VarianceThreshold(0.0).fit(X)
        Xv = self.vt.transform(X)
        self.sc = StandardScaler().fit(Xv)
        Xs = self.sc.transform(Xv)
        self.sel = SelectKBest(f_classif, k=min(self.k, Xs.shape[1])).fit(Xs, y) if self.k else None
        return self

    def transform(self, X):
        Z = self.sc.transform(self.vt.transform(X))
        return self.sel.transform(Z) if self.sel else Z


import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold


TUNING_GRIDS = {
    "rf": [
        {"max_depth": 15, "min_samples_split": 2, "n_estimators": 100},
        {"max_depth": 25, "min_samples_split": 5, "n_estimators": 150},
        {"max_depth": None, "min_samples_split": 2, "n_estimators": 200},
    ],
    "xgb": [
        {"max_depth": 4, "learning_rate": 0.1, "subsample": 0.8},
        {"max_depth": 6, "learning_rate": 0.1, "subsample": 1.0},
        {"max_depth": 8, "learning_rate": 0.05, "subsample": 0.8},
    ],
    "lgbm": [
        {"num_leaves": 31, "learning_rate": 0.1, "max_depth": -1},
        {"num_leaves": 63, "learning_rate": 0.05, "max_depth": 10},
        {"num_leaves": 31, "learning_rate": 0.05, "max_depth": 8},
    ],
}


def make_model(name, seed, params=None):
    params = params or {}
    if name == "rf":
        kwargs = dict(n_estimators=200, n_jobs=-1, random_state=seed)
        kwargs.update(params)
        return RandomForestClassifier(**kwargs)
    if name == "lr":
        kwargs = dict(max_iter=1000)
        kwargs.update(params)
        return LogisticRegression(**kwargs)
    if name == "hgb":  # sklearn gradient boosting, always available
        kwargs = dict(random_state=seed)
        kwargs.update(params)
        return HistGradientBoostingClassifier(**kwargs)
    if name == "xgb":
        import xgboost as xgb
        kwargs = dict(n_estimators=300, max_depth=6, learning_rate=0.1,
                      tree_method="hist", random_state=seed, n_jobs=-1)
        kwargs.update(params)
        return xgb.XGBClassifier(**kwargs)
    if name == "lgbm":
        import lightgbm as lgb
        kwargs = dict(n_estimators=300, random_state=seed, verbose=-1)
        kwargs.update(params)
        return lgb.LGBMClassifier(**kwargs)
    raise ValueError(name)


def tune_model(name, Xtr, ytr, groups_tr, seed, sample_weight=None, n_splits=3, grid=None):
    """Small grouped-CV hyperparameter tuning on training data only, selected by validation macro-F1."""
    if name not in TUNING_GRIDS and grid is None:
        m = make_model(name, seed)
        m.fit(Xtr, ytr, sample_weight=sample_weight)
        return m, {}, None

    candidates = grid or TUNING_GRIDS[name]

    # CV strictly within training data
    if groups_tr is not None and len(np.unique(groups_tr)) >= n_splits:
        cv = GroupKFold(n_splits=n_splits)
        splits = list(cv.split(Xtr, ytr, groups=groups_tr))
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splits = list(cv.split(Xtr, ytr))

    results = []
    best_score = -1.0
    best_params = candidates[0]

    for params in candidates:
        fold_scores = []
        for train_idx, val_idx in splits:
            X_fold_tr, y_fold_tr = Xtr[train_idx], ytr[train_idx]
            X_fold_va, y_fold_va = Xtr[val_idx], ytr[val_idx]
            w_fold_tr = sample_weight[train_idx] if sample_weight is not None else None

            model = make_model(name, seed, params=params)
            model.fit(X_fold_tr, y_fold_tr, sample_weight=w_fold_tr)
            if name == "rf":
                model.set_params(n_jobs=1)
            preds = model.predict(X_fold_va)
            f1 = f1_score(y_fold_va, preds, average="macro", zero_division=0)
            fold_scores.append(f1)

        mean_f1 = float(np.mean(fold_scores))
        std_f1 = float(np.std(fold_scores))
        results.append({
            "model": name,
            "params": str(params),
            "mean_cv_macro_f1": mean_f1,
            "std_cv_macro_f1": std_f1,
            "raw_params": params,
        })
        if mean_f1 > best_score:
            best_score = mean_f1
            best_params = params

    # Refit best model on the complete training set
    best_model = make_model(name, seed, params=best_params)
    best_model.fit(Xtr, ytr, sample_weight=sample_weight)
    if name == "rf":
        best_model.set_params(n_jobs=1)

    cv_df = pd.DataFrame(results).sort_values(by="mean_cv_macro_f1", ascending=False).reset_index(drop=True)
    return best_model, best_params, cv_df


def get_logits(model, name, X):
    """Raw class scores used for calibration, energy and confidence."""
    if name == "rf":
        model.set_params(n_jobs=1)
    if name == "xgb":
        return model.predict(X, output_margin=True)
    if name == "lgbm":
        return model.predict(X, raw_score=True)
    if name == "lr":
        return model.decision_function(X)
    return np.log(np.clip(model.predict_proba(X), 1e-9, 1.0))  # rf, hgb


def fit_temperature(z, y):
    """Temperature scaling: choose T minimising NLL on the validation set."""
    def nll(logt):
        lp = z / np.exp(logt)
        lp = lp - logsumexp(lp, axis=1, keepdims=True)
        return -lp[np.arange(len(y)), y].mean()
    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(out)


class Anomaly:
    """Trained on known-class training data only. Higher score = more anomalous."""
    def __init__(self, kind, seed):
        self.kind, self.seed = kind, seed

    def fit(self, X):
        if self.kind == "iforest":
            self.m = IsolationForest(n_estimators=200, random_state=self.seed, n_jobs=-1).fit(X)
        else:  # autoencoder via sklearn MLP (reconstruction error)
            h = max(2, X.shape[1] // 4)
            self.m = MLPRegressor((X.shape[1] // 2 or 2, h, X.shape[1] // 2 or 2), max_iter=60,
                                  random_state=self.seed).fit(X, X)
        return self

    def score(self, X):
        if self.kind == "iforest":
            return -self.m.score_samples(X)
        return ((self.m.predict(X) - X) ** 2).mean(1)


class Mahalanobis:
    """Class-conditional Gaussians with shared covariance. Score = min squared distance."""
    def fit(self, X, y):
        self.mu = np.stack([X[y == k].mean(0) for k in np.unique(y)])
        Xc = X - self.mu[y]
        self.P = np.linalg.pinv(np.cov(Xc, rowvar=False) + 1e-6 * np.eye(X.shape[1]))
        return self

    def score(self, X):
        d = [np.einsum("ij,jk,ik->i", X - m, self.P, X - m) for m in self.mu]
        return np.min(d, axis=0)


def rank_vs(ref, s):
    """Empirical CDF of s relative to reference (validation known) scores."""
    ref_s = np.sort(np.round(np.asarray(ref), 9))
    s_r = np.round(np.asarray(s), 9)
    return 0.5 * (np.searchsorted(ref_s, s_r, side="left") + np.searchsorted(ref_s, s_r, side="right")) / len(ref)


def class_weights(y, mode):
    """Per-sample weights. mode: none | sqrt (inverse sqrt frequency) | balanced (inverse frequency)."""
    if mode in (None, "none"):
        return None
    cnt = np.bincount(y).astype(float)
    w = cnt.sum() / (len(cnt) * np.maximum(cnt, 1))
    if mode == "sqrt":
        w = np.sqrt(w)
    return w[y]
