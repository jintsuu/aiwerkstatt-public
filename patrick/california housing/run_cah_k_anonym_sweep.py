# run_cah_k_anonym_sweep.py
import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

# =========================
# CONFIG
# =========================
DATA_HOME = os.path.join(os.getcwd(), "sklearn_data_cache")
FIG_DIR   = os.path.join(os.getcwd(), "figures")
os.makedirs(DATA_HOME, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# mehrere Durchläufe -> Vergleich / Stabilität
SEEDS = [0, 1, 2, 3, 4]

# k-Sweep (für CAH lieber nicht zu klein, sonst zu viele Gruppen)
K_LIST = [10, 20, 50, 100, 200]

# quasi-identifiers (QIs) – typisch sensible/spatial/demographic proxies
QI_COLS = ["Latitude", "Longitude", "HouseAge", "MedInc"]

# Feature-Weights für "Feature-Gewichtung" in der Mondrian-Splitauswahl
# (größer = eher wird entlang dieser Spalte gesplittet -> mehr "Schutz" dort)
QI_WEIGHTS = {"Latitude": 1.0, "Longitude": 1.0, "HouseAge": 0.6, "MedInc": 0.8}

# Generalisierung: "mean" hält alles numerisch (stabil, kein Dim-Explode).
# alternative: "interval_str" (macht strings wie [min,max] -> dann OneHot auf Intervalle)
GENERALIZE_MODE = "mean"  # "mean" | "interval_str"

# Optional: abgeleitete categorical + multi-hot Features hinzufügen (für Meeting nice)
USE_ENGINEERED = True

# =========================
# FIG SAVE
# =========================
def _safe(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")

def savefig(name: str):
    path = os.path.join(FIG_DIR, _safe(name) + ".png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print("saved:", path)

# =========================
# MULTI-HOT ENCODER (sklearn-compatible)
# =========================


class MultiHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, col: str):
        self.col = col
        self.mlb = MultiLabelBinarizer(sparse_output=True)

    def _get_series(self, X):
        # ColumnTransformer gives DataFrame (n,1) for ["GeoTags"]
        if isinstance(X, pd.DataFrame):
            s = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            s = X
        else:
            s = pd.Series(np.asarray(X).ravel())

        # ensure list-of-tokens per row
        return s.apply(lambda v: v if isinstance(v, list) else ([] if v is None else [str(v)]))

    def fit(self, X, y=None):
        s = self._get_series(X)
        self.mlb.fit(s)
        return self

    def transform(self, X):
        s = self._get_series(X)
        return self.mlb.transform(s)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{self.col}__{c}" for c in self.mlb.classes_], dtype=object)


# =========================
# FEATURE ENGINEERING (categorical + multi-hot)
# CA Housing is numeric only -> we derive small categorical bins + multi-label tags
# =========================
def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    # categorical bins (low cardinality)
    X["LatBin"]  = pd.qcut(X["Latitude"],  6, duplicates="drop").astype(str)
    X["LongBin"] = pd.qcut(X["Longitude"], 6, duplicates="drop").astype(str)
    X["AgeBin"]  = pd.cut(X["HouseAge"], bins=[0,10,20,30,40,60], include_lowest=True).astype(str)

    # multi-label tags (each row gets multiple tags)
    lat_med  = X["Latitude"].median()
    lon_med  = X["Longitude"].median()
    inc_med  = X["MedInc"].median()
    occ_med  = X["AveOccup"].median()

    def row_tags(r):
        tags = []
        tags += ["north"] if r["Latitude"] >= lat_med else ["south"]
        tags += ["west"]  if r["Longitude"] <= lon_med else ["east"]
        tags += ["hi_inc"] if r["MedInc"] >= inc_med else ["lo_inc"]
        tags += ["hi_occ"] if r["AveOccup"] >= occ_med else ["lo_occ"]
        # simple "coastal-ish" proxy (CA coastline longitudes are more negative)
        tags += ["coastal"] if r["Longitude"] < -121 else ["inland"]
        return tags

    X["GeoTags"] = X.apply(row_tags, axis=1)
    return X

# =========================
# MONDRIAN-LIKE k-ANON (numeric QIs) with optional weighting
# - partition via median splits (top-down)
# - apply generalization per equivalence class
# =========================
def mondrian_numeric(df: pd.DataFrame, qi_cols, k: int, weights=None, max_depth=50, mode="mean") -> pd.DataFrame:
    df = df.copy()
    qi_cols = [c for c in qi_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if not qi_cols:
        return df

    if weights is None:
        weights = {c: 1.0 for c in qi_cols}
    w = np.array([float(weights.get(c, 1.0)) for c in qi_cols])

    idx_all = df.index.to_numpy()
    parts = [idx_all]
    depth = 0

    while depth < max_depth:
        depth += 1
        changed = False
        new_parts = []

        for idx in parts:
            if len(idx) < 2 * k:
                new_parts.append(idx)
                continue

            sub = df.loc[idx, qi_cols]
            ranges = (sub.max() - sub.min()).to_numpy()
            score = ranges * w  # weighted choice
            order = np.argsort(-score)

            split_done = False
            for j in order:
                if score[j] <= 0:
                    continue
                col = qi_cols[j]
                med = sub[col].median()
                left  = idx[df.loc[idx, col].to_numpy() <= med]
                right = idx[df.loc[idx, col].to_numpy() > med]
                if len(left) >= k and len(right) >= k:
                    new_parts += [left, right]
                    split_done = True
                    changed = True
                    break

            if not split_done:
                new_parts.append(idx)

        parts = new_parts
        if not changed:
            break

    # generalize within each equivalence class
    if mode == "interval_str":
        for idx in parts:
            for c in qi_cols:
                mn = df.loc[idx, c].min()
                mx = df.loc[idx, c].max()
                df.loc[idx, c] = f"[{mn:.3g},{mx:.3g}]"
    else:
        # default: mean -> keeps numeric (stable dimensionality)
        for idx in parts:
            for c in qi_cols:
                m = df.loc[idx, c].mean()
                df.loc[idx, c] = float(m)

    return df

# =========================
# PIPELINE (Regression)
# - numeric: StandardScaler
# - categorical: OneHot
# - multi-hot: MultiHotEncoder
# =========================


def make_reg_pipe(X: pd.DataFrame):
    # categorical columns BUT exclude GeoTags (multi-label lists)
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if "GeoTags" in cat_cols:
        cat_cols.remove("GeoTags")

    # numeric columns
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    if "GeoTags" in X.columns:
        transformers.append(("mhot", MultiHotEncoder("GeoTags"), ["GeoTags"]))

    pre = ColumnTransformer(transformers, remainder="drop")
    model = Ridge(alpha=1.0)
    return Pipeline([("pre", pre), ("model", model)])


# =========================
# RUN: baseline + k-sweep, multiple seeds
# =========================
def run_k_sweep(X, y, k_list, seeds, qi_cols, qi_weights, mode, title_prefix):
    rows = []

    for seed in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed)

        # baseline (no anon)
        pipe0 = make_reg_pipe(Xtr)
        pipe0.fit(Xtr, ytr)
        pred0 = pipe0.predict(Xte)
        base = r2_score(yte, pred0)

        rows.append({"seed": seed, "k": 0, "r2": float(base), "setting": "baseline"})

        # anonymized train only
        for k in k_list:
            Xtr_an = mondrian_numeric(
                Xtr, qi_cols=qi_cols, k=k,
                weights=qi_weights, mode=mode
            )
            pipe = make_reg_pipe(Xtr_an)
            pipe.fit(Xtr_an, ytr)
            pred = pipe.predict(Xte)
            r2 = r2_score(yte, pred)

            rows.append({"seed": seed, "k": int(k), "r2": float(r2), "setting": f"k={k}"})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(os.getcwd(), f"results_{title_prefix}.csv"), index=False)
    return df

def agg_curve(df):
    g = df.groupby("k")["r2"]
    return pd.DataFrame({"k": g.mean().index, "mean": g.mean().values, "sd": g.std(ddof=0).values})

def plot_curve(df_agg, title):
    plt.figure()
    plt.errorbar(df_agg["k"], df_agg["mean"], yerr=df_agg["sd"], capsize=3)
    plt.xlabel("k (0 = baseline)")
    plt.ylabel("R² (test)")
    plt.title(title)
    plt.tight_layout()
    savefig(title)
    plt.show()
    plt.close()

def plot_loss(df_agg, title):
    base = df_agg.loc[df_agg["k"] == 0, "mean"].iloc[0]
    loss = 100.0 * (base - df_agg["mean"]) / (abs(base) + 1e-12)
    plt.figure()
    plt.bar(df_agg["k"].astype(str), loss)
    plt.xlabel("k (0 = baseline)")
    plt.ylabel("relative loss (%) vs baseline")
    plt.title(title)
    plt.tight_layout()
    savefig(title)
    plt.show()
    plt.close()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # load California Housing
    cah = fetch_california_housing(as_frame=True, data_home=DATA_HOME)
    X = cah.data
    y = cah.target

    # optional engineered cat + multi-hot
    if USE_ENGINEERED:
        X = add_engineered_features(X)
        title_prefix = f"CAH_engineered_{GENERALIZE_MODE}"
    else:
        title_prefix = f"CAH_base_{GENERALIZE_MODE}"

    print("X shape:", X.shape, "| y:", y.name)
    print("dtypes:\n", X.dtypes.value_counts())

    df_res = run_k_sweep(
        X, y,
        k_list=K_LIST,
        seeds=SEEDS,
        qi_cols=QI_COLS,
        qi_weights=QI_WEIGHTS,
        mode=GENERALIZE_MODE,
        title_prefix=title_prefix
    )

    df_agg = agg_curve(df_res)
    print("\nAggregated curve:\n", df_agg)

    plot_curve(df_agg, f"{title_prefix}_k_sweep_R2_mean_std")
    plot_loss(df_agg, f"{title_prefix}_k_sweep_R2_loss_percent")
def plot_seed_lines(df_res, title):
    plt.figure()
    for seed, g in df_res.groupby("seed"):
        g = g.sort_values("k")
        plt.plot(g["k"], g["r2"], marker="o", linewidth=1, label=f"seed={seed}")
    plt.xlabel("k (0 = baseline)")
    plt.ylabel("R² (test)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    savefig(title)
    plt.show()
    plt.close()
