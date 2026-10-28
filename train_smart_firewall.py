#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LABEL_CANDIDATES = ["label", "attack_cat", "class", "target"]
NORMAL_LABELS = {"normal", "benign", "0", "false", "legitimate", "non-attack"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a smart-firewall classifier for normal vs attack traffic"
    )
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument(
        "--label-column",
        default=None,
        help="Label column name (auto-detected if omitted)",
    )
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.95,
        help="Variance to preserve with PCA (0-1)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="F-beta weight for threshold tuning (>1 favors recall, <1 favors precision)",
    )
    parser.add_argument(
        "--model-out",
        default="smart_firewall_model.joblib",
        help="Output path for trained model bundle",
    )
    return parser.parse_args()


def detect_label_column(dataframe: pd.DataFrame, explicit_column: str | None) -> str:
    if explicit_column:
        if explicit_column not in dataframe.columns:
            raise ValueError(f"Label column '{explicit_column}' not found in dataset")
        return explicit_column

    for col in LABEL_CANDIDATES:
        if col in dataframe.columns:
            return col

    raise ValueError(
        "Could not detect label column. Pass --label-column explicitly "
        f"or include one of: {', '.join(LABEL_CANDIDATES)}"
    )


def binarize_labels(y: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(y):
        unique_vals = sorted(pd.unique(y.dropna()))
        if len(unique_vals) != 2:
            raise ValueError("Numeric labels must have exactly 2 unique values for binary classification")
        return (y == unique_vals[-1]).astype(int)

    normalized = y.astype(str).str.strip().str.lower()
    return (~normalized.isin(NORMAL_LABELS)).astype(int)


def pick_threshold(y_true: np.ndarray, scores: np.ndarray, beta: float) -> tuple[float, dict]:
    best_threshold = 0.5
    best_metrics = {"precision": 0.0, "recall": 0.0, "f_beta": -1.0}

    for threshold in np.linspace(0.05, 0.95, 19):
        preds = (scores >= threshold).astype(int)
        precision, recall, _, _ = precision_recall_fscore_support(
            y_true, preds, average="binary", zero_division=0
        )
        beta_sq = beta**2
        denom = (beta_sq * precision) + recall
        f_beta = 0.0 if denom == 0 else ((1 + beta_sq) * precision * recall) / denom

        if f_beta > best_metrics["f_beta"]:
            best_metrics = {
                "precision": float(precision),
                "recall": float(recall),
                "f_beta": float(f_beta),
            }
            best_threshold = float(threshold)

    return best_threshold, best_metrics


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    label_column = detect_label_column(df, args.label_column)

    y = binarize_labels(df[label_column])
    x = df.drop(columns=[label_column])

    x = x.select_dtypes(include=[np.number])
    if x.empty:
        raise ValueError("No numerical features found. Expected numeric network traffic features.")

    x = x.fillna(x.median(numeric_only=True))

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.30, random_state=42, stratify=y
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    pipeline = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=args.pca_variance, random_state=42)),
            ("clf", LogisticRegression(max_iter=500, solver="lbfgs")),
        ]
    )
    pipeline.fit(x_train, y_train)

    val_scores = pipeline.predict_proba(x_val)[:, 1]
    threshold, val_metrics = pick_threshold(y_val.to_numpy(), val_scores, beta=args.beta)

    test_scores = pipeline.predict_proba(x_test)[:, 1]
    test_preds = (test_scores >= threshold).astype(int)
    report = classification_report(y_test, test_preds, digits=4)
    auc = roc_auc_score(y_test, test_scores)

    model_bundle = {
        "pipeline": pipeline,
        "threshold": threshold,
        "label_column": label_column,
        "feature_columns": list(x.columns),
    }
    joblib.dump(model_bundle, args.model_out)

    summary = {
        "label_column": label_column,
        "rows": int(df.shape[0]),
        "numerical_features": int(x.shape[1]),
        "chosen_threshold": threshold,
        "validation_metrics": val_metrics,
        "test_roc_auc": float(auc),
        "model_path": str(Path(args.model_out).resolve()),
    }

    print(json.dumps(summary, indent=2))
    print("\nTest Classification Report:\n")
    print(report)


if __name__ == "__main__":
    main()
