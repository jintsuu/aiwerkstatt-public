import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)

import anonypy


CARDIO_PATH = r"C:/Users/Patrick/Desktop/UNI/AI Werkstatt/cardio_train.csv"
K_SINGLE = 10
K_VALUES = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
TEST_SIZE = 0.2
SEED = 42

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def expand_groups(df_g: pd.DataFrame) -> pd.DataFrame:
    if "count" not in df_g.columns:
        return df_g.copy()
    rec = []
    for _, r in df_g.iterrows():
        d = r.drop("count").to_dict()
        rec += [d] * int(r["count"])
    return pd.DataFrame(rec)


def prepare_split(df: pd.DataFrame, num_hint):
    y = df["cardio"].astype(int)
    X = df.drop(columns=["cardio", "id"], errors="ignore")

    auto_num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in num_hint if c in auto_num]
    cat = [c for c in X.columns if c not in num]

    Xn = X[num].astype(float) if num else None
    Xc = pd.get_dummies(X[cat].astype(str), drop_first=True) if cat else None
    Xp = pd.concat([z for z in (Xn, Xc) if z is not None], axis=1)

    Xp = Xp.replace([np.inf, -np.inf], np.nan)
    if Xp.isna().any().any():
        Xp = Xp.astype(float).fillna(Xp.mean())

    Xtr, Xte, ytr, yte = train_test_split(
        Xp, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    sc = StandardScaler()
    return sc.fit_transform(Xtr), sc.transform(Xte), ytr, yte


def fit_eval(Xtr, Xte, ytr, yte):
    m = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    m.fit(Xtr, ytr)

    yp = m.predict(Xte)
    pp = m.predict_proba(Xte)[:, 1]

    cm = confusion_matrix(yte, yp)
    acc = accuracy_score(yte, yp)
    p, r, f1, _ = precision_recall_fscore_support(
        yte, yp, average="weighted", zero_division=0
    )
    fpr, tpr, _ = roc_curve(yte, pp)
    ra = auc(fpr, tpr)

    return dict(cm=cm, accuracy=acc, precision=p, recall=r, f1=f1, fpr=fpr, tpr=tpr, roc_auc=ra)


def anonymize_anonypy(df: pd.DataFrame, k: int) -> pd.DataFrame:
    feat = [
        "age", "gender", "height", "weight", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active",
    ]
    sens = "cardio"
    df0 = df.drop(columns=["id"], errors="ignore")[feat + [sens]].copy()

    cat = ["gender", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]
    for c in cat:
        df0[c] = df0[c].astype("category")

    rows = anonypy.Preserver(df0, feature_columns=feat, sensitive_column=sens).anonymize_k_anonymity(k=k)
    df_g = pd.DataFrame(rows)
    df_a = expand_groups(df_g)
    df_a["cardio"] = df_a["cardio"].astype(int)
    return df_a


def plot_comparison(a, b, k):
    sns.set_style("darkgrid")

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(f"Classifier Comparison: Original vs Anonymized Data (k={k})",
                 fontsize=18, fontweight="bold", y=0.98)

    ax1 = plt.subplot(2, 3, 1)
    labs = ["Original", "Anonymized"]
    vals = [a["accuracy"], b["accuracy"]]
    bars = ax1.bar(labs, vals, alpha=0.9)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy", fontsize=14, fontweight="bold")
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v*100:.2f}%",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax2 = plt.subplot(2, 3, 2)
    sns.heatmap(a["cm"], annot=True, fmt="d", cbar=False, square=True, ax=ax2, cmap="Greens")
    ax2.set_title("Confusion Matrix - Original", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Predicted"); ax2.set_ylabel("True")

    ax3 = plt.subplot(2, 3, 3)
    sns.heatmap(b["cm"], annot=True, fmt="d", cbar=False, square=True, ax=ax3, cmap="Blues")
    ax3.set_title("Confusion Matrix - Anonymized", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Predicted"); ax3.set_ylabel("True")

    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(a["fpr"], a["tpr"], lw=2.5, label=f"Original (AUC={a['roc_auc']:.3f})")
    ax4.plot(b["fpr"], b["tpr"], lw=2.5, label=f"Anonymized (AUC={b['roc_auc']:.3f})")
    ax4.plot([0, 1], [0, 1], "k--", lw=1.5)
    ax4.set_xlim(0, 1); ax4.set_ylim(0, 1.05)
    ax4.set_xlabel("FPR"); ax4.set_ylabel("TPR")
    ax4.set_title("ROC", fontsize=13, fontweight="bold")
    ax4.legend(loc="lower right", fontsize=9)

    ax5 = plt.subplot(2, 3, 5)
    m = ["precision", "recall", "f1"]
    oa = [a[x] for x in m]
    ob = [b[x] for x in m]
    x = np.arange(3); w = 0.35
    p1 = ax5.bar(x - w/2, oa, w, label="Original")
    p2 = ax5.bar(x + w/2, ob, w, label="Anonymized")
    ax5.set_xticks(x); ax5.set_xticklabels(["Precision", "Recall", "F1"])
    ax5.set_ylim(0, 1)
    ax5.set_title("Weighted Avg Metrics", fontsize=13, fontweight="bold")
    ax5.legend()
    for bars in (p1, p2):
        for bar in bars:
            h = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                     ha="center", va="bottom", fontsize=9)

    ax6 = plt.subplot(2, 3, 6)
    ax6.axis("off")
    acc_o, acc_a = a["accuracy"] * 100, b["accuracy"] * 100
    auc_o, auc_a = a["roc_auc"] * 100, b["roc_auc"] * 100
    d_acc, d_auc = acc_a - acc_o, auc_a - auc_o
    s = "Minimal impact on model performance."
    if abs(d_auc) >= 5:
        s = "Noticeable performance drop after anonymization." if d_auc < 0 else "Performance slightly improved."

    txt = (
        "Performance Impact of Anonymization\n"
        "====================================\n\n"
        f"Original Accuracy:   {acc_o:6.2f}%\n"
        f"Anonymized Accuracy: {acc_a:6.2f}%\n"
        f"Δ Accuracy:          {d_acc:+6.2f}%\n\n"
        f"Original AUC:        {auc_o:6.2f}%\n"
        f"Anonymized AUC:      {auc_a:6.2f}%\n"
        f"Δ AUC:               {d_auc:+6.2f}%\n\n"
        "====================================\n\n"
        "Summary:\n"
        f"{s}"
    )
    ax6.text(0.02, 0.98, txt, va="top", ha="left", fontsize=10, family="monospace",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = os.path.join(FIG_DIR, f"cardio_anonypy_comparison_k{k}.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()


def sweep_k(df, num_cols, ks):
    acc, ra, f1 = [], [], []
    for k in ks:
        dfk = anonymize_anonypy(df, k)
        Xtr, Xte, ytr, yte = prepare_split(dfk, num_cols)
        r = fit_eval(Xtr, Xte, ytr, yte)
        acc.append(r["accuracy"]); ra.append(r["roc_auc"]); f1.append(r["f1"])
    return np.array(acc), np.array(ra), np.array(f1)


def plot_k_sweep(ks, acc, ra, f1, base_acc, base_auc):
    sns.set_style("darkgrid")

    fig, ax1 = plt.subplots(figsize=(10, 6))
    fig.suptitle("Impact of k on Model Performance (anonypy / Mondrian)",
                 fontsize=16, fontweight="bold")

    ax1.plot(ks, acc, marker="o", lw=2, label="Accuracy (anon)")
    ax1.axhline(base_acc, color="grey", ls="--", lw=1.5, label="Accuracy (orig)")
    ax1.scatter(ks, f1, marker="^", s=60, label="F1 (anon)")
    ax1.set_xlabel("k"); ax1.set_ylabel("Accuracy / F1"); ax1.set_ylim(0, 1)
    ax1.set_xticks(ks); ax1.set_xticklabels([str(k) for k in ks])

    ax2 = ax1.twinx()
    ax2.plot(ks, ra, marker="s", lw=2, label="AUC (anon)")
    ax2.axhline(base_auc, ls="--", lw=1.5, label="AUC (orig)")
    ax2.set_ylabel("AUC"); ax2.set_ylim(0, 1)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out = os.path.join(FIG_DIR, "cardio_anonypy_k_sweep.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    df = pd.read_csv(CARDIO_PATH, sep=";")
    num_cols = ["age", "height", "weight", "ap_hi", "ap_lo"]

    Xtr, Xte, ytr, yte = prepare_split(df, num_cols)
    res_o = fit_eval(Xtr, Xte, ytr, yte)

    dfk = anonymize_anonypy(df, K_SINGLE)
    Xtr, Xte, ytr, yte = prepare_split(dfk, num_cols)
    res_k = fit_eval(Xtr, Xte, ytr, yte)

    plot_comparison(res_o, res_k, K_SINGLE)

    acc, ra, f1 = sweep_k(df, num_cols, K_VALUES)
    plot_k_sweep(K_VALUES, acc, ra, f1, res_o["accuracy"], res_o["roc_auc"])


if __name__ == "__main__":
    main()
