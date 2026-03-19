from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load best model (change if needed)
model = joblib.load("../models/logistic.pkl")


@app.route("/")
def home():
    return "Churn Prediction API Running"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        df = pd.DataFrame([data])

        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return jsonify({
            "churn_prediction": int(prediction),
            "churn_probability": float(probability)
        })

    except Exception as e:
        return jsonify({"error": str(e)})
    

if __name__ == "__main__":
    app.run(debug=True)