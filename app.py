"""
app.py  –  Anemia Classification Flask App
Features: Gender, Hemoglobin, MCH, MCHC, MCV  ->  Result (0=No Anemia, 1=Anemia)
"""

from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
import os

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

def load_asset(filename):
    path = os.path.join(MODEL_DIR, filename)
    return joblib.load(path) if os.path.exists(path) else None

MODELS = {
    "svm":               load_asset("svm.pkl"),
    "random_forest":     load_asset("random_forest.pkl"),
    "gradient_boosting": load_asset("gradient_boosting.pkl"),
}
SCALER = load_asset("scaler.pkl")

class MockModel:
    def predict(self, X):
        hb = X[0][1]
        return [1] if hb < 12.0 else [0]
    def predict_proba(self, X):
        hb = X[0][1]
        p = max(0.05, min(0.95, (12.0 - hb) / 6.0 + 0.5))
        return [[round(1 - p, 3), round(p, 3)]]

for k in MODELS:
    if MODELS[k] is None:
        MODELS[k] = MockModel()

LABELS = {0: "No Anemia", 1: "Anemia Detected"}
LABEL_COLORS = {0: "green", 1: "red"}

@app.route("/")
def index():
    models_trained = all(
        os.path.exists(os.path.join(MODEL_DIR, f"{m}.pkl"))
        for m in ("svm", "random_forest", "gradient_boosting")
    )
    return render_template("index.html", models_trained=models_trained)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        features = np.array([[
            float(data["gender"]),
            float(data["hemoglobin"]),
            float(data["mch"]),
            float(data["mchc"]),
            float(data["mcv"]),
        ]])
        features_in = SCALER.transform(features) if SCALER else features
        model_key = data.get("model", "random_forest")
        model = MODELS.get(model_key, MODELS["random_forest"])
        pred  = int(model.predict(features_in)[0])
        proba = model.predict_proba(features_in)[0]
        return jsonify({
            "prediction": pred,
            "label":      LABELS[pred],
            "color":      LABEL_COLORS[pred],
            "probabilities": {
                "No Anemia": round(float(proba[0]) * 100, 1),
                "Anemia":    round(float(proba[1]) * 100, 1),
            },
            "model_used": model_key.replace("_", " ").title(),
            "confidence": round(float(max(proba)) * 100, 1),
        })
    except (KeyError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
