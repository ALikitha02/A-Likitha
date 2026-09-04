import joblib
import pandas as pd

MODEL_PATH = "models/razorguard_model.joblib"

model_package = joblib.load(MODEL_PATH)

model = model_package["model"]
features = model_package["features"]
threshold = model_package["threshold"]


def analyze_transaction(transaction):

    transaction_df = pd.DataFrame([transaction])

    X = transaction_df[features]

    fraud_probability = model.predict_proba(X)[0][1]

    risk_score = round(fraud_probability * 100, 2)

    if fraud_probability >= 0.70:
        risk_level = "HIGH"
        recommended_action = "HOLD FOR REVIEW"

    elif fraud_probability >= threshold:
        risk_level = "MEDIUM"
        recommended_action = "ADDITIONAL VERIFICATION"

    else:
        risk_level = "LOW"
        recommended_action = "ALLOW"

    reasons = []

    if transaction["amount"] >= 1000:
        reasons.append("High transaction amount")

    if transaction["transactions_last_24h"] >= 8:
        reasons.append("High transaction velocity")

    if transaction["account_age_days"] <= 30:
        reasons.append("New account")

    if transaction["device_changes_7d"] >= 3:
        reasons.append("Multiple device changes")

    if transaction["location_changes_7d"] >= 3:
        reasons.append("Multiple location changes")

    if (
        transaction["transaction_hour"] <= 4
        or transaction["transaction_hour"] >= 23
    ):
        reasons.append("Unusual transaction time")

    if not reasons:
        if risk_level == "HIGH":         
            reasons.append("Model detected elevated fraud risk from transaction behavior")
        elif risk_level == "MEDIUM":
            reasons.append("Model detected moderate fraud risk from transaction behavior")
        else:
            reasons.append("No major behavioral risk indicators detected")
    return {
        "transaction_id": transaction.get("transaction_id", "UNKNOWN"),
        "fraud_probability": round(fraud_probability, 4),
        "risk_score": risk_score,
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "reasons": reasons
    }