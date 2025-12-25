import argparse
import json
import os
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_dataframe(data_uri: str) -> pd.DataFrame:
    """
    Loads CSV from:
      - GCS: gs://bucket/path/file.csv
      - Local: /path/to/file.csv or relative path
    Note: pandas can read gs:// if gcsfs is installed (it is in your requirements).
    """
    return pd.read_csv(data_uri)


def build_pipeline(feature_cols):
    # Numeric-only pipeline (your columns look numeric)
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_cols),
        ],
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=2000,
        n_jobs=1,  # safer on managed envs
        solver="lbfgs",
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-uri", required=True, help="CSV path (gs://... or local)")
    p.add_argument("--model-dir", required=True, help="Output directory for model artifacts")
    p.add_argument("--target-col", default="sales", help="Target column name")
    args = p.parse_args()

    print(f"[{_utc_now_iso()}] Loading data from: {args.data_uri}")
    df = load_dataframe(args.data_uri)

    if args.target_col not in df.columns:
        raise ValueError(
            f"Target column '{args.target_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Basic cleanup: drop fully empty rows
    df = df.dropna(how="all")

    # Ensure model_dir exists
    os.makedirs(args.model_dir, exist_ok=True)

    # Define features
    feature_cols = [c for c in df.columns if c != args.target_col]

    # Try to coerce everything to numeric for safety (your features are numeric)
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Target: if sales is 0/1 or numeric, we treat as classification.
    # If it’s not 0/1, we still do a simple binary split around median as fallback.
    y_raw = df[args.target_col]
    if pd.api.types.is_numeric_dtype(y_raw):
        # If already looks binary, keep it
        unique_vals = sorted(pd.Series(y_raw.dropna().unique()).tolist())
        if set(unique_vals).issubset({0, 1}):
            y = y_raw.astype(int)
        else:
            # Fallback: convert to binary around median
            med = float(y_raw.median())
            print(f"[{_utc_now_iso()}] Target not binary. Converting to binary: sales > median({med})")
            y = (y_raw > med).astype(int)
    else:
        # Non-numeric target: convert to category codes (binary if possible)
        y = y_raw.astype("category").cat.codes
        # If multi-class, keep as is (LogReg can handle multinomial with lbfgs)
        y = y.astype(int)

    X = df[feature_cols]

    # Simple train/valid split (no sklearn train_test_split to keep deps minimal)
    # 80/20 split by row order (good enough for this demo)
    n = len(df)
    if n < 10:
        raise ValueError(f"Not enough rows to train. Need at least ~10, got {n}.")

    split = int(n * 0.8)
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    pipe = build_pipeline(feature_cols)

    print(f"[{_utc_now_iso()}] Training rows: {len(X_train)}, Validation rows: {len(X_val)}")
    pipe.fit(X_train, y_train)

    # Evaluate
    y_pred = pipe.predict(X_val)
    acc = float(accuracy_score(y_val, y_pred))
    metrics = {"accuracy": acc}

    # AUC if binary and we have predict_proba
    try:
        if len(pd.unique(y_train)) == 2:
            y_prob = pipe.predict_proba(X_val)[:, 1]
            auc = float(roc_auc_score(y_val, y_prob))
            metrics["roc_auc"] = auc
    except Exception as e:
        print(f"[{_utc_now_iso()}] AUC calc skipped: {e}")

    print(f"[{_utc_now_iso()}] Metrics: {json.dumps(metrics, indent=2)}")
    try:
        print(classification_report(y_val, y_pred))
    except Exception:
        pass

    # Save model + metadata into model-dir (THIS is what Vertex expects)
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(pipe, model_path)

    meta = {
        "created_utc": _utc_now_iso(),
        "data_uri": args.data_uri,
        "target_col": args.target_col,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "model_file": "model.joblib",
    }
    with open(os.path.join(args.model_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[{_utc_now_iso()}] Saved model to: {model_path}")
    print(f"[{_utc_now_iso()}] Saved metadata to: {os.path.join(args.model_dir, 'metadata.json')}")
    print(f"[{_utc_now_iso()}] Training complete.")


if __name__ == "__main__":
    main()
