# src/model_training.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score

def train_models():
    print("Loading dataset...")
    df = pd.read_csv("../data/features.csv")

    # target
    target = "aqi"
    y = df[target].copy()
    X = df.drop(columns=[target, 'date'], errors='ignore')

    # encode categorical columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # numeric conversion + missing fill
    X = X.apply(pd.to_numeric, errors='coerce').ffill().bfill()

    print("Task: CLASSIFICATION")

    # fix AQI labels from 1–5 → 0–4
    y = y - y.min()

    # train-test split
    split = int(0.8 * len(df))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print("\nTraining ensemble models...")

    rf  = RandomForestClassifier(n_estimators=200, random_state=42)
    xgb = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
    lgb = LGBMClassifier()

    # ------- ENSEMBLE LEARNING (Soft Voting) -------
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('xgb', xgb),
            ('lgbm', lgb)
        ],
        voting='soft' 
    )

    ensemble.fit(X_train_scaled, y_train)
    preds = ensemble.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    f1  = f1_score(y_test, preds, average='weighted')
    print(f"\nEnsemble Model -> Accuracy: {acc:.4f} | F1 Score: {f1:.4f}")

    # save model + scaler
    joblib.dump(ensemble, "../models/final_model.joblib")
    joblib.dump(scaler, "../models/preprocessor.joblib")

    print("\nEnsemble model and scaler saved successfully.")

if __name__ == "__main__":
    train_models()
