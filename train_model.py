"""Train the crop recommendation Random Forest and save it to models/RandomForest.pkl.

Usage:  python train_model.py
"""
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent
DATA = ROOT / "Data" / "crop_recommendation.csv"
MODEL = ROOT / "models" / "RandomForest.pkl"


def make_features(N, P, K, ph, rainfall):
    """Feature vector used by both training and the web app (keep them identical)."""
    return [N, P, K, N + P, N + K, ph, rainfall]


def main():
    df = pd.read_csv(DATA)
    X = pd.DataFrame(
        [make_features(r.N, r.P, r.K, r.ph, r.rainfall) for r in df.itertuples()],
        columns=["N", "P", "K", "N_plus_P", "N_plus_K", "ph", "rainfall"],
    )
    X = X.to_numpy()   # the web app predicts from a plain array, so train on one too
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"Test accuracy: {acc:.3f}  ({len(X_test)} held-out rows, {y.nunique()} crops)")

    # Retrain on all data for the saved model
    model.fit(X, y)
    MODEL.parent.mkdir(exist_ok=True)
    with open(MODEL, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {MODEL}")


if __name__ == "__main__":
    main()
