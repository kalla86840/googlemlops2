import argparse
import os
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

FEATURE_COLUMNS = [
    "call_duration_sec",
    "customer_tenure_days",
    "prior_calls_count",
    "agent_experience_level",
    "call_queue_position",
]

TARGET_COLUMN = "sales"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-uri", required=True)
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data_uri)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    model = LinearRegression()
    model.fit(X, y)

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)

    print("Training complete")
    print("Saved model to:", model_path)

if __name__ == "__main__":
    main()
