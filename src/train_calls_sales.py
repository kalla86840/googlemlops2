import argparse
import os
import tempfile
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def _is_gcs(uri: str) -> bool:
    return uri.startswith("gs://")

def _write_bytes_to_gcs(gcs_uri: str, data: bytes) -> None:
    import gcsfs  # comes from requirements.txt
    fs = gcsfs.GCSFileSystem()
    with fs.open(gcs_uri, "wb") as f:
        f.write(data)

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-uri", required=True, help="GCS path to CSV, e.g. gs://.../calls_sales_dataset.csv")
    p.add_argument("--model-dir", required=True, help="Output directory (local or gs://) where model artifacts go")
    p.add_argument("--target-col", default="sales", help="Target column name in the CSV")
    args = p.parse_args()

    df = pd.read_csv(args.data_uri)

    if args.target_col not in df.columns:
        raise ValueError(f"Target column '{args.target_col}' not found. Columns: {list(df.columns)}")

    X = df.drop(columns=[args.target_col])
    y = df[args.target_col]

    # Simple baseline model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    metrics_text = f"MAE={mae:.6f}\nR2={r2:.6f}\nrows={len(df)}\nfeatures={list(X.columns)}\n"

    # Save artifacts
    if _is_gcs(args.model_dir):
        # Write model.joblib + metrics.txt to GCS
        with tempfile.TemporaryDirectory() as td:
            local_model = os.path.join(td, "model.joblib")
            local_metrics = os.path.join(td, "metrics.txt")
            joblib.dump(model, local_model)
            with open(local_metrics, "w", encoding="utf-8") as f:
                f.write(metrics_text)

            with open(local_model, "rb") as f:
                _write_bytes_to_gcs(f"{args.model_dir.rstrip('/')}/model.joblib", f.read())
            with open(local_metrics, "rb") as f:
                _write_bytes_to_gcs(f"{args.model_dir.rstrip('/')}/metrics.txt", f.read())
    else:
        _ensure_dir(args.model_dir)
        joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
        with open(os.path.join(args.model_dir, "metrics.txt"), "w", encoding="utf-8") as f:
            f.write(metrics_text)

    print("Training complete.")
    print("Saved artifacts to:", args.model_dir)
    print(metrics_text)

if __name__ == "__main__":
    main()
