import pandas as pd
import numpy as np
import anonypy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import visualisation
from joblib import Parallel, delayed


def k_anonymize(dataframe, quasi_idents, categorical, k=5):
    df_copy = dataframe.copy()
    for col in quasi_idents:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='raise')
    for col in categorical:
        df_copy[col] = df_copy[col].astype("category")
    anonymizer = anonypy.Preserver(df_copy, quasi_idents, 'Default')
    anonymized_rows = anonymizer.anonymize_k_anonymity(k=k)
    if not anonymized_rows:
        return pd.DataFrame(columns=df_copy.columns)
    cols = list(df_copy.columns)
    if len(anonymized_rows[0]) == len(cols) + 1:
        cols.append('count')
    anonymized_df = pd.DataFrame(anonymized_rows, columns=cols)
    return anonymized_df


def preprocess_data(data, target_column):
    data = data.copy()
    for col in data.columns:
        if col == target_column or col == 'count':
            continue
        if data[col].dtype == 'object':
            numeric_col = pd.to_numeric(data[col], errors='coerce')
            is_range = numeric_col.isna() & data[col].astype(str).str.contains(' - ', regex=False)

            if is_range.any():
                ranges = data.loc[is_range, col].astype(str).str.extract(r'(\d+\.?\d*) - (\d+\.?\d*)').astype(float)
                means = (ranges[0] + ranges[1]) / 2
                numeric_col.loc[is_range] = means

            if numeric_col.notna().any():
                data[col] = numeric_col

            if data[col].dtype == 'object' or data[col].isna().any():
                data[col] = data[col].fillna("Unknown")
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))
    return data


def single_run(dataframe, quasi_idents, categorical, target_column, k, test_size):
    X = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size
    )

    train_data = pd.concat([X_train, y_train], axis=1)
    train_data_anon = k_anonymize(train_data, quasi_idents, categorical, k=k)

    test_data = pd.concat([X_test, y_test], axis=1)
    test_data_anon = k_anonymize(test_data, quasi_idents, categorical, k=1)

    train_data_anon = preprocess_data(train_data_anon, target_column)

    if 'count' in train_data_anon.columns:
        weights = train_data_anon['count']
        X_train_anon = train_data_anon.drop(columns=[target_column, 'count'])
    else:
        weights = None
        X_train_anon = train_data_anon.drop(columns=[target_column])

    y_train_anon = train_data_anon[target_column]

    test_data_anon = preprocess_data(test_data_anon, target_column)

    if 'count' in test_data_anon.columns:
        X_test_anon = test_data_anon.drop(columns=[target_column, 'count'])
    else:
        X_test_anon = test_data_anon.drop(columns=[target_column])

    y_test_anon = test_data_anon[target_column]

    X_test_anon = X_test_anon.reindex(columns=X_train_anon.columns, fill_value=0)

    clf = RandomForestClassifier(n_jobs=1, class_weight={0: 1, 1: 4})

    clf.fit(X_train_anon, y_train_anon, sample_weight=weights)

    y_proba = clf.predict_proba(X_test_anon)[:, 1]
    threshold = 0.3

    y_pred = (y_proba >= threshold).astype(int)
    report = classification_report(y_test_anon, y_pred, output_dict=True)
    acc = accuracy_score(y_test_anon, y_pred)

    return {
        'accuracy': acc,
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score']
    }


def train_classifier(dataframe, target_column, test_size=0.2):
    data = preprocess_data(dataframe, target_column)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size
    )

    clf = RandomForestClassifier(n_jobs=-1, class_weight={0: 1, 1: 5})
    clf.fit(X_train, y_train)

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    threshold = 0.3
    y_pred = (y_pred_proba >= threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("\n" + "=" * 30)
    print(f"Baseline Model Accuracy: {acc:.2%}")
    print("=" * 30)

    return clf, {
        'accuracy': acc,
        'report': report,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }


def compare_over_k(dataframe, quasi_idents, categorical, target_column, k_values, test_size=0.2, n_runs=10):
    results = {}
    for k in k_values:
        print(f"\nStarting {n_runs} parallel runs for k={k}...")
        parallel_results = Parallel(n_jobs=-1)(
            delayed(single_run)(
                dataframe, quasi_idents, categorical, target_column, k, test_size
            )
            for _ in range(n_runs)
        )
        accuracies = [res['accuracy'] for res in parallel_results]
        precisions = [res['precision'] for res in parallel_results]
        recalls = [res['recall'] for res in parallel_results]
        f1_scores = [res['f1_score'] for res in parallel_results]
        results[k] = {
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'precision_mean': np.mean(precisions),
            'precision_std': np.std(precisions),
            'recall_mean': np.mean(recalls),
            'recall_std': np.std(recalls),
            'f1_score_mean': np.mean(f1_scores),
            'f1_score_std': np.std(f1_scores)
        }
        print(f"  k={k} Complete - Mean Acc: {results[k]['accuracy_mean']:.2%} (±{results[k]['accuracy_std']:.4f})")
    return results


if __name__ == "__main__":
    try:
        loan_data = pd.read_csv('datasets/Loan_default.csv', low_memory=False)
    except FileNotFoundError:
        print("Error: 'datasets/Loan_default.csv' not found.")
        exit()

    loan_data.drop(columns=['LoanID'], inplace=True, errors='ignore')
    education_map = {
        'High School': 0, "Bachelor's": 1, "Master's": 2, 'PhD': 3
    }
    loan_data['Education'] = loan_data['Education'].map(education_map).fillna(0)
    binary_map = {'Yes': 1, 'No': 0}
    for col in ['HasMortgage', 'HasDependents', 'HasCoSigner']:
        if col in loan_data.columns:
            loan_data[col] = loan_data[col].map(binary_map).fillna(0)
    nominal_cols = ["EmploymentType", "MaritalStatus", "LoanPurpose"]
    loan_data = pd.get_dummies(loan_data, columns=nominal_cols, prefix=nominal_cols)

    loan_data = loan_data.astype(float)
    loan_data['Default'] = loan_data['Default'].astype(int)


    quasi_identifiers = ['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines',
                         'InterestRate', 'LoanTerm', 'DTIRatio']

    quasi_identifiers.extend(['Education', 'HasMortgage', 'HasDependents', 'HasCoSigner'])

    new_one_hot_cols = [c for c in loan_data.columns if any(c.startswith(p) for p in nominal_cols)]
    quasi_identifiers.extend(new_one_hot_cols)

    categorical = set()

    print("\nTraining on Original Data (Baseline)...")
    result_loan_orig = train_classifier(loan_data, target_column='Default')

    print("\n" + "=" * 50)
    print("Analyzing quality over different k values (Parallelized + Vectorized)...")
    print("=" * 50)

    k_results = compare_over_k(loan_data, quasi_identifiers, categorical, 'Default', k_values=range(2, 11), n_runs=10)

    visualisation.create_k_comparison(result_loan_orig, k_results, 'outputs/k_comparison.png')
    print("\n" + "=" * 50)
    print("K-value comparison saved to: outputs/k_comparison.png")
    print("=" * 50)