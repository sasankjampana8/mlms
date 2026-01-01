# trainer.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans


def basic_profile(df: pd.DataFrame) -> Dict[str, Any]:
    missing = df.isna().sum().to_dict()
    dtypes = {c: str(t) for c, t in df.dtypes.items()}
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "dtypes": dtypes,
        "missing_top": dict(sorted(missing.items(), key=lambda x: x[1], reverse=True)[:10]),
    }


def _split_cols(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols


def _build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
    )


def _pick_estimator(problem_type: str, algorithm: str) -> Any:
    algo = algorithm.lower()

    if problem_type == "classification":
        if algo == "logistic_regression":
            return LogisticRegression(max_iter=200)
        if algo == "random_forest":
            return RandomForestClassifier(n_estimators=300, random_state=42)
        if algo == "gradient_boosting":
            return GradientBoostingClassifier(random_state=42)
        raise ValueError(f"Unsupported classification algorithm: {algorithm}")

    if problem_type == "regression":
        if algo == "linear_regression":
            return LinearRegression()
        if algo == "random_forest":
            return RandomForestRegressor(n_estimators=300, random_state=42)
        if algo == "gradient_boosting":
            return GradientBoostingRegressor(random_state=42)
        raise ValueError(f"Unsupported regression algorithm: {algorithm}")

    if problem_type == "clustering":
        if algo == "kmeans":
            return KMeans(n_clusters=3, random_state=42, n_init="auto")
        raise ValueError(f"Unsupported clustering algorithm: {algorithm}")

    raise ValueError(f"Unsupported problem_type: {problem_type}")


def train_and_evaluate(
    df: pd.DataFrame,
    problem_type: str,
    target_col: str,
    metric: str,
    algorithm: str,
    model_out_path: str,
    params: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Returns (metrics, logs_text) and saves model pipeline to model_out_path.
    """
    params = params or {}
    logs: List[str] = []

    logs.append(f"Problem type: {problem_type}")
    logs.append(f"Algorithm: {algorithm}")
    logs.append(f"Metric: {metric}")

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    if problem_type in ("classification", "regression"):
        if not target_col or target_col not in df.columns:
            raise ValueError("Target column is missing/invalid for supervised problems.")

        y = df[target_col]
        X = df.drop(columns=[target_col])

        num_cols, cat_cols = _split_cols(X)
        logs.append(f"Features: {len(X.columns)} | numeric={len(num_cols)} categorical={len(cat_cols)}")

        pre = _build_preprocessor(num_cols, cat_cols)
        est = _pick_estimator(problem_type, algorithm)

        # Apply optional params safely (only if attribute exists)
        try:
            est.set_params(**params)
            logs.append(f"Applied params: {json.dumps(params)}")
        except Exception:
            logs.append("Warning: params could not be applied; continuing with defaults.")

        pipe = Pipeline(steps=[("preprocess", pre), ("model", est)])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if problem_type == "classification" else None
        )
        logs.append(f"Split: train={len(X_train)} test={len(X_test)}")

        pipe.fit(X_train, y_train)
        logs.append("Training completed.")

        metrics: Dict[str, Any] = {}

        if problem_type == "classification":
            # AUC only reliable for binary with predict_proba
            y_pred = pipe.predict(X_test)
            metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
            metrics["f1_macro"] = float(f1_score(y_test, y_pred, average="macro"))

            if hasattr(pipe.named_steps["model"], "predict_proba"):
                try:
                    probs = pipe.predict_proba(X_test)
                    if probs.shape[1] == 2:
                        metrics["roc_auc"] = float(roc_auc_score(y_test, probs[:, 1]))
                except Exception:
                    pass

        else:  # regression
            y_pred = pipe.predict(X_test)
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            metrics["rmse"] = float(rmse)
            metrics["mae"] = float(mean_absolute_error(y_test, y_pred))
            metrics["r2"] = float(r2_score(y_test, y_pred))

        os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
        joblib.dump(pipe, model_out_path)
        logs.append(f"Saved model pipeline to: {model_out_path}")

        return metrics, "\n".join(logs)

    # clustering
    X = df
    num_cols, cat_cols = _split_cols(X)
    pre = _build_preprocessor(num_cols, cat_cols)
    est = _pick_estimator(problem_type, algorithm)

    # optional params
    try:
        est.set_params(**params)
        logs.append(f"Applied params: {json.dumps(params)}")
    except Exception:
        logs.append("Warning: params could not be applied; continuing with defaults.")

    X_trans = pre.fit_transform(X)
    logs.append(f"Transformed shape: {getattr(X_trans, 'shape', None)}")

    labels = est.fit_predict(X_trans)
    logs.append("Clustering completed.")

    metrics = {"n_clusters": int(len(set(labels)))}

    # silhouette only if > 1 cluster
    if len(set(labels)) > 1:
        try:
            metrics["silhouette"] = float(silhouette_score(X_trans, labels))
        except Exception:
            pass

    pipe = Pipeline(steps=[("preprocess", pre), ("model", est)])
    os.makedirs(os.path.dirname(model_out_path), exist_ok=True)
    joblib.dump(pipe, model_out_path)
    logs.append(f"Saved clustering pipeline to: {model_out_path}")

    return metrics, "\n".join(logs)


def predict_from_model(model_path: str, input_row: Dict[str, Any]) -> Dict[str, Any]:
    pipe = joblib.load(model_path)
    X = pd.DataFrame([input_row])
    pred = pipe.predict(X)
    out: Dict[str, Any] = {"prediction": pred.tolist()}

    # If classifier supports probabilities
    model = pipe.named_steps.get("model")
    if hasattr(model, "predict_proba"):
        try:
            proba = pipe.predict_proba(X)
            out["probabilities"] = proba.tolist()
        except Exception:
            pass

    return out
