import pandas as pd


def detect_fraud_spike(history_df, window_minutes=60, threshold=0.20):

    if history_df.empty:
        return {
            "status": "NO DATA",
            "risk_rate": 0.0,
            "transaction_count": 0,
            "message": "No transaction data available."
        }

    df = history_df.copy()

    df["time"] = pd.to_datetime(df["time"])

    latest_time = df["time"].max()

    window_start = latest_time - pd.Timedelta(
        minutes=window_minutes
    )

    recent_df = df[df["time"] >= window_start]

    transaction_count = len(recent_df)

    if transaction_count == 0:
        return {
            "status": "NO DATA",
            "risk_rate": 0.0,
            "transaction_count": 0,
            "message": "No recent transactions available."
        }

    risky_transactions = len(
        recent_df[
            recent_df["risk_level"].isin(
                ["HIGH", "MEDIUM"]
            )
        ]
    )

    risk_rate = risky_transactions / transaction_count

    if risk_rate >= threshold:

        return {
            "status": "SPIKE DETECTED",
            "risk_rate": round(risk_rate, 4),
            "transaction_count": transaction_count,
            "message": (
                "Unusual concentration of risky transactions "
                "detected in the recent time window."
            )
        }

    return {
        "status": "NORMAL",
        "risk_rate": round(risk_rate, 4),
        "transaction_count": transaction_count,
        "message": (
            "No significant fraud-risk spike detected."
        )
    }