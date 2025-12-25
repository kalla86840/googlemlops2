import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-uri", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--target-col", default="sales")
    args = p.parse_args()

    df = pd.read_csv(args.data_uri)
    y = df[args.target_col]
    X = df.drop(columns=[args.target_col])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)

    print(f"Saved model to: {model_path}")

if __name__ == "__main__":
    main()
