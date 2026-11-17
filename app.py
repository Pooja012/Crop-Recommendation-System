"""Flask web app: recommend a crop from soil nutrients, pH and rainfall."""
import pickle
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request

from train_model import make_features

MODEL_PATH = Path(__file__).parent / "models" / "RandomForest.pkl"
if not MODEL_PATH.exists():
    raise SystemExit("Model not found. Run `python train_model.py` first.")
with open(MODEL_PATH, "rb") as f:
    crop_recommendation_model = pickle.load(f)

app = Flask(__name__)
TITLE = "Crop Recommendation"


@app.route("/")
@app.route("/index")
def home():
    return render_template("index.html", title=TITLE)


@app.route("/crop-recommend")
def crop_recommend():
    return render_template("crop.html", title=TITLE)


@app.route("/crop-predict", methods=["POST"])
def crop_prediction():
    try:
        N = int(request.form["nitrogen"])
        P = int(request.form["phosphorous"])
        K = int(request.form["pottasium"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])
    except (KeyError, ValueError):
        return render_template("try_again.html", title=TITLE), 400

    data = np.array([make_features(N, P, K, ph, rainfall)])
    prediction = crop_recommendation_model.predict(data)[0]
    return render_template("crop-result.html", prediction=prediction, title=TITLE)


if __name__ == "__main__":
    app.run(debug=False)
