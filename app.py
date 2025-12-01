from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Load trained ensemble model and scaler

MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.joblib")
SCALER_PATH = os.path.join(BASE_DIR, "models", "preprocessor.joblib")
HISTORY_PATH = os.path.join(BASE_DIR, "data", "aqi_history.csv")


model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ---- helper: get lag features from history file ----
def get_lag_features():
    """
    Returns aqi_lag1, aqi_lag2, aqi_roll3 based on previous predictions.
    If not enough history, uses reasonable defaults.
    """
    if not os.path.exists(HISTORY_PATH):
        # no history yet → assume moderate AQI
        return 3.0, 3.0, 3.0

    hist = pd.read_csv(HISTORY_PATH)

    if len(hist) == 0:
        return 3.0, 3.0, 3.0

    # ensure sorted by date
    hist = hist.sort_values("date")

    last_values = hist["aqi"].values

    if len(last_values) == 1:
        a1 = last_values[-1]
        return float(a1), float(a1), float(a1)

    if len(last_values) == 2:
        a1 = last_values[-1]
        a2 = last_values[-2]
        roll3 = (a1 + a2) / 2.0
        return float(a1), float(a2), float(roll3)

    # 3 or more rows
    a1 = last_values[-1]
    a2 = last_values[-2]
    a3 = last_values[-3]
    roll3 = (a1 + a2 + a3) / 3.0
    return float(a1), float(a2), float(roll3)


# ---- helper: append new prediction to history ----
def save_prediction_to_history(aqi_value):
    """
    Store the predicted AQI (in original 1–5 scale) with today's date.
    """
    today = datetime.today().date().isoformat()
    row = {"date": today, "aqi": aqi_value}

    if os.path.exists(HISTORY_PATH):
        df = pd.read_csv(HISTORY_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(HISTORY_PATH, index=False)


@app.route("/")
def home():
    return render_template("index.html", prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # -------- 1. Read city and pollutants from form --------
        city_code = float(request.form["city"])
        co = float(request.form["co"])
        no = float(request.form["no"])
        no2 = float(request.form["no2"])
        o3 = float(request.form["o3"])
        so2 = float(request.form["so2"])
        pm2_5 = float(request.form["pm2_5"])
        pm10 = float(request.form["pm10"])
        nh3 = float(request.form["nh3"])

        # -------- 2. Date-based features (today) --------
        today = datetime.today()
        day = today.day
        month = today.month
        year = today.year
        dayofweek = today.weekday()   # Monday=0 ... Sunday=6 (same as pandas)

        # -------- 3. Lag features from history --------
        aqi_lag1, aqi_lag2, aqi_roll3 = get_lag_features()

        # -------- 4. City feature (now from user selection) --------
        # city_code is already set from form input above

        # -------- 5. Build feature vector in SAME ORDER as training --------
        feature_order = [
            "city", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
            "day", "month", "year", "dayofweek",
            "aqi_lag1", "aqi_lag2", "aqi_roll3"
        ]

        values = [
            city_code, co, no, no2, o3, so2, pm2_5, pm10, nh3,
            day, month, year, dayofweek,
            aqi_lag1, aqi_lag2, aqi_roll3
        ]

        X = pd.DataFrame([values], columns=feature_order)

        # -------- 6. Scale + Predict --------
        X_scaled = scaler.transform(X)
        pred_class = int(model.predict(X_scaled)[0])  # 0–4

        # training time you did y = y - y.min() (likely min=1)
        # so original AQI category = pred_class + 1
        original_aqi_value = pred_class + 1

        # save to history for future lag calculation
        save_prediction_to_history(original_aqi_value)

        # -------- 7. Map class to label --------
        labels = {
            0: "Good",
            1: "Satisfactory",
            2: "Moderate",
            3: "Poor",
            4: "Very Poor"
        }
        result = labels.get(pred_class, "Unknown")

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)
