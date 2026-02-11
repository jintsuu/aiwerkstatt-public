import pandas as pd
import anonypy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns



def baseline_model(df):
    df = df.copy()
    le = LabelEncoder()
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = le.fit_transform(df[column])
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(class_weight='balanced', max_depth=10)
    model.fit(X_train, y_train)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    print(f"Baseline Model Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", report)
    return model, accuracy, report

def k_anonymize(dataframe, quasi_idents, categorical, k=5):
    df_copy = dataframe.copy()
    for col in quasi_idents:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='raise')
    for col in categorical:
        df_copy[col] = df_copy[col].astype("category")
    anonymizer = anonypy.Preserver(df_copy, categorical, 'class')
    anonymized_rows = anonymizer.anonymize_k_anonymity(k=k)
    if not anonymized_rows:
        return pd.DataFrame(columns=df_copy.columns)
    cols = list(df_copy.columns)
    has_count = len(anonymized_rows[0]) == len(cols) + 1
    if has_count:
        cols.append('count')
    anonymized_df = pd.DataFrame(anonymized_rows, columns=cols)
    if has_count:
        anonymized_df = anonymized_df.loc[
            anonymized_df.index.repeat(anonymized_df['count'])
        ].drop(columns=['count']).reset_index(drop=True)
    anonymized_df.to_csv(f"nursery_k{k}.csv", index=False)
    return anonymized_df

def evaluate_anonymized_model(df, k=5):

    df_copy = df.copy()

    df_train_original, df_test_original = train_test_split(df_copy, test_size=0.2)


    quasi_idents = ['children']
    categorical = list(df.columns[:-1])

    anonymized_train = k_anonymize(df_train_original, quasi_idents, categorical, k=k)
    if anonymized_train.empty:
        print("It's over")
        raise Exception
    for col in anonymized_train.columns:
        anonymized_train[col] = anonymized_train[col].apply(
            lambda x: ','.join(map(str, x)) if isinstance(x, list) else str(x)
        )

    X_train = pd.get_dummies(anonymized_train.drop('class', axis=1))

    le = LabelEncoder()
    le.fit(df['class'])
    y_train = le.transform(anonymized_train['class'])

    X_test = pd.get_dummies(df_test_original.drop('class', axis=1))
    y_test = le.transform(df_test_original['class'])

    X_train, X_test = X_train.align(X_test, join='outer', axis=1)

    model = RandomForestClassifier(class_weight='balanced', max_depth=10)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"Anonymized Model Accuracy (k={k}): {accuracy:4f}")
    print("Classification Report:\n", report)

    return model, accuracy, report


def compare_models(df):
    print("=" * 50)
    print("Training Baseline Model...")
    print("=" * 50)
    baseline_model_obj, baseline_acc, baseline_report = baseline_model(df)

    print("\n" + "=" * 50)
    print("Training K-Anonymized Model (k=5)...")
    print("=" * 50)
    anonymized_model_obj, anonymized_acc, anonymized_report = evaluate_anonymized_model(df, k=5)

    print("\n" + "=" * 50)
    print("VERGLEICH DER MODELLE")
    print("=" * 50)
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    print(f"K-Anonymized (k=5) Accuracy: {anonymized_acc:.4f}")
    print(
        f"Genauigkeitsverlust: {(baseline_acc - anonymized_acc):.4f} ({((baseline_acc - anonymized_acc) / baseline_acc * 100):.2f}%)")

    return {
        'baseline': {'model': baseline_model_obj, 'accuracy': baseline_acc, 'report': baseline_report},
        'anonymized': {'model': anonymized_model_obj, 'accuracy': anonymized_acc, 'report': anonymized_report}
    }

def get_aggregated_importances(model, feature_names, original_columns):
    raw_importances = model.feature_importances_

    agg_imp = {col: 0.0 for col in original_columns}

    for feat_name, importance in zip(feature_names, raw_importances):
        match_found = False
        for orig_col in original_columns:
            if feat_name == orig_col or feat_name.startswith(f"{orig_col}_"):
                agg_imp[orig_col] += importance
                match_found = True
                break
    return agg_imp


def visualize_feature_trends(importance_df):
    pivot_df = importance_df.pivot(index='Feature', columns='K_Value', values='Importance')

    cols = sorted(pivot_df.columns, key=lambda x: -1 if x == 'Baseline' else int(x))
    pivot_df = pivot_df[cols]

    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, cmap='magma', fmt='.3f')
    plt.title('Change in Feature Importance (Baseline vs. K-Anonymity)')
    plt.xlabel('Anonymization Level (K)')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('feature_importance_heatmap.png', dpi=150)
    plt.show()


def analyze_feature_importance(df, k_range=range(2, 11)):
    print("\n" + "=" * 50)
    print("ANALYZING FEATURE IMPORTANCE")
    print("=" * 50)

    original_cols = [c for c in df.columns if c != 'class']
    importance_data = []

    print("Extracting Baseline features...")
    base_model, _, _ = baseline_model(df)

    base_feat_names = df.drop('class', axis=1).columns
    base_imps = get_aggregated_importances(base_model, base_feat_names, original_cols)

    for feat, val in base_imps.items():
        importance_data.append({'K_Value': 'Baseline', 'Feature': feat, 'Importance': val})

    for k in k_range:
        print(f"Extracting features for k={k}...")
        try:
            model, acc, _ = evaluate_anonymized_model(df, k=k)

            if hasattr(model, 'feature_names_in_'):
                feat_names = model.feature_names_in_
            else:
                print("Warning: Model does not store feature names. Skipping...")
                continue

            k_imps = get_aggregated_importances(model, feat_names, original_cols)

            for feat, val in k_imps.items():
                importance_data.append({'K_Value': str(k), 'Feature': feat, 'Importance': val})

        except Exception as e:
            print(f"Skipping k={k} in feature analysis due to error: {e}")

    imp_df = pd.DataFrame(importance_data)
    visualize_feature_trends(imp_df)
    return imp_df

def compare_over_k(df, k_range=range(2, 11), runs_per_k=10):
    results = {'k': [], 'run': [], 'accuracy': [], 'type': []}

    print("Berechne Baseline-Accuracies...")
    for run in range(runs_per_k):
        _, acc, _ = baseline_model(df)
        results['k'].append(0)
        results['run'].append(run + 1)
        results['accuracy'].append(acc)
        results['type'].append('Baseline')

    for k in k_range:
        print(f"\nTesting k={k} ({runs_per_k} Durchläufe)")
        for run in range(runs_per_k):
            try:
                _, acc, _ = evaluate_anonymized_model(df, k=k)
                results['k'].append(k)
                results['run'].append(run + 1)
                results['accuracy'].append(acc)
                results['type'].append(f'k={k}')
            except Exception as e:
                print(f"Fehler bei k={k}, run={run+1}: {e}")

    results_df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results_df, x='type', y='accuracy', palette='viridis')
    plt.title('Accuracy-Verteilung pro k-Wert')
    plt.xlabel('Modell')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('k_comparison_plot.png', dpi=150)
    plt.show()

    return results_df


if __name__ == "__main__":
    df = pd.read_csv("nursery.csv")

    children_map = {"1": 0, "2": 1, "3": 2, "more": 7}
    df['children'] = df['children'].map(children_map)

    print(">>> PART 1: Accuracy Comparison (Boxplots)")
    results_df = compare_over_k(df, k_range=range(2, 11), runs_per_k=10)

    print("\n>>> PART 2: Feature Importance Analysis (Heatmap)")
    feature_df = analyze_feature_importance(df, k_range=range(2, 11))




