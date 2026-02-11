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
    auc
)


def anonymize_cardio(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    age_years = df["age"] // 365
    df["age_group"] = (age_years // 10) * 10
    df["height_group"] = (df["height"] // 10) * 10
    df["weight_group"] = (df["weight"] // 5) * 5
    return df.drop(columns=["age", "height", "weight"])


def train_and_evaluate(X_train, X_test, y_train, y_test, seed=42):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train)
    Xte = sc.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    clf.fit(Xtr, y_train)

    y_pred = clf.predict(Xte)
    y_proba = clf.predict_proba(Xte)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    return dict(
        model=clf, scaler=sc, y_test=y_test, y_pred=y_pred, y_proba=y_proba,
        cm=cm, accuracy=acc, precision=p, recall=r, f1=f1, fpr=fpr, tpr=tpr, roc_auc=roc_auc
    )


def plot_comparison(a, b, out="cardio_original_vs_anonymized.png"):
    sns.set_style("darkgrid")

    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Classifier Comparison: Original vs Anonymized Data", fontsize=18, fontweight="bold", y=0.98)

    # Accuracy
    labels = ["Original Data", "Anonymized Data"]
    vals = [a["accuracy"], b["accuracy"]]
    bars = ax[0, 0].bar(labels, vals, alpha=0.9)
    ax[0, 0].set_ylim(0, 1)
    ax[0, 0].set_ylabel("Accuracy")
    ax[0, 0].set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    for bar, v in zip(bars, vals):
        ax[0, 0].text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v*100:.2f}%",
                      ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Confusion matrices
    sns.heatmap(a["cm"], annot=True, fmt="d", cbar=False, square=True, ax=ax[0, 1], cmap="Greens")
    ax[0, 1].set_title("Confusion Matrix - Original Data", fontsize=13, fontweight="bold")
    ax[0, 1].set_xlabel("Predicted Label"); ax[0, 1].set_ylabel("True Label")

    sns.heatmap(b["cm"], annot=True, fmt="d", cbar=False, square=True, ax=ax[0, 2], cmap="Blues")
    ax[0, 2].set_title("Confusion Matrix - Anonymized Data", fontsize=13, fontweight="bold")
    ax[0, 2].set_xlabel("Predicted Label"); ax[0, 2].set_ylabel("True Label")

    # ROC
    ax[1, 0].plot(a["fpr"], a["tpr"], linewidth=2.5, label=f"Original (AUC = {a['roc_auc']:.3f})")
    ax[1, 0].plot(b["fpr"], b["tpr"], linewidth=2.5, label=f"Anonymized (AUC = {b['roc_auc']:.3f})")
    ax[1, 0].plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Classifier")
    ax[1, 0].set_xlim(0, 1); ax[1, 0].set_ylim(0, 1.05)
    ax[1, 0].set_xlabel("False Positive Rate"); ax[1, 0].set_ylabel("True Positive Rate")
    ax[1, 0].set_title("ROC Curves Comparison", fontsize=13, fontweight="bold")
    ax[1, 0].legend(loc="lower right", fontsize=10)

    # Metrics
    m = ["precision", "recall", "f1"]
    oa = [a[x] for x in m]
    ob = [b[x] for x in m]
    x = np.arange(len(m)); w = 0.35
    p1 = ax[1, 1].bar(x - w/2, oa, w, label="Original")
    p2 = ax[1, 1].bar(x + w/2, ob, w, label="Anonymized")
    ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels(["Precision", "Recall", "F1-Score"])
    ax[1, 1].set_ylim(0, 1); ax[1, 1].set_ylabel("Score")
    ax[1, 1].set_title("Metrics Comparison (Weighted Avg)", fontsize=13, fontweight="bold")
    ax[1, 1].legend()
    for bars in (p1, p2):
        for bar in bars:
            h = bar.get_height()
            ax[1, 1].text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.3f}",
                          ha="center", va="bottom", fontsize=9)

    # Summary box
    ax[1, 2].axis("off")
    acc_o, acc_a = a["accuracy"] * 100, b["accuracy"] * 100
    auc_o, auc_a = a["roc_auc"] * 100, b["roc_auc"] * 100
    d_acc, d_auc = acc_a - acc_o, auc_a - auc_o
    summary = ("Minimal impact on model performance." if abs(d_auc) < 5 else
               "Noticeable performance drop after anonymization." if d_auc < 0 else
               "Performance slightly improved (possible overfitting reduction).")

    text = (
        "Performance Impact of Anonymization\n"
        "====================================\n\n"
        f"Original Data Accuracy:   {acc_o:6.2f}%\n"
        f"Anonymized Data Accuracy: {acc_a:6.2f}%\n"
        f"Accuracy Change:          {d_acc:+6.2f}%\n\n"
        f"Original AUC-ROC:         {auc_o:6.2f}%\n"
        f"Anonymized AUC-ROC:       {auc_a:6.2f}%\n"
        f"AUC Change:               {d_auc:+6.2f}%\n\n"
        "====================================\n\n"
        "Summary:\n"
        f"{summary}"
    )
    ax[1, 2].text(0.02, 0.98, text, va="top", ha="left", fontsize=10, family="monospace",
                  bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.show()


def main():
    DATA_PATH = r"C:/Users/Patrick/Desktop/UNI/AI Werkstatt/cardio_train.csv"
    ANON_PATH = r"C:/Users/Patrick/Desktop/UNI/AI Werkstatt/cardio_train_k_anon.csv"

    df = pd.read_csv(DATA_PATH, sep=";")
    df_anon = anonymize_cardio(df)
    df_anon.to_csv(ANON_PATH, sep=";", index=False)

    qis = ["age_group", "gender", "height_group", "weight_group"]
    k = int(df_anon.groupby(qis).size().min())
    print(f"Estimated k (min group size over QIs) = {k}")

    idx = np.arange(len(df))
    y = df["cardio"].to_numpy()
    idx_tr, idx_te, y_tr, y_te = train_test_split(idx, y, test_size=0.2, random_state=42, stratify=y)

    feat_o = [c for c in df.columns if c not in ["id", "cardio"]]
    feat_a = [c for c in df_anon.columns if c not in ["id", "cardio"]]

    Xtr_o, Xte_o = df.iloc[idx_tr][feat_o], df.iloc[idx_te][feat_o]
    Xtr_a, Xte_a = df_anon.iloc[idx_tr][feat_a], df_anon.iloc[idx_te][feat_a]

    res_o = train_and_evaluate(Xtr_o, Xte_o, y_tr, y_te, seed=42)
    res_a = train_and_evaluate(Xtr_a, Xte_a, y_tr, y_te, seed=42)

    plot_comparison(res_o, res_a)


if __name__ == "__main__":
    main()
