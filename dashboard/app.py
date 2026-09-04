import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
import joblib
import requests
def analyze_via_api(transaction):
    response = requests.post(
        "http://127.0.0.1:8000/analyze",
        json=transaction,
        timeout=10
    )

    response.raise_for_status()

    return response.json()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
DATA_PATH = PROJECT_ROOT / "data"
AUDIT_FILE = DATA_PATH / "audit_log.csv"

sys.path.insert(0, str(SRC_PATH))

from risk_engine import analyze_transaction
from fraud_spike import detect_fraud_spike


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="RazorGuard AI",
    page_icon="🛡️",
    layout="wide"
)


# -----------------------------
# Create data folder
# -----------------------------
DATA_PATH.mkdir(exist_ok=True)


# -----------------------------
# Load Audit History
# -----------------------------
def load_history():
    if AUDIT_FILE.exists():
        return pd.read_csv(AUDIT_FILE)
    
    return pd.DataFrame(
        columns=[
            "time",
            "transaction_id",
            "amount",
            "risk_score",
            "risk_level",
            "fraud_probability"
        ]
    )


history_df = load_history()


# -----------------------------
# Header
# -----------------------------
st.title("🛡️ RazorGuard AI")
st.subheader("AI Payment Risk Manager")

st.write(
    "Analyze payment transactions and estimate fraud risk "
    "using a machine-learning model."
)

st.divider()


# -----------------------------
# Risk Summary
# -----------------------------
st.header("📊 Risk Summary")

total_transactions = len(history_df)

high_risk = len(
    history_df[history_df["risk_level"] == "HIGH"]
)

medium_risk = len(
    history_df[history_df["risk_level"] == "MEDIUM"]
)

low_risk = len(
    history_df[history_df["risk_level"] == "LOW"]
)

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

with summary_col1:
    st.metric("Total Transactions", total_transactions)

with summary_col2:
    st.metric("🔴 High Risk", high_risk)

with summary_col3:
    st.metric("🟡 Medium Risk", medium_risk)

with summary_col4:
    st.metric("🟢 Low Risk", low_risk)


st.divider()


# -----------------------------
# Transaction Analysis
# -----------------------------
st.header("🔎 Transaction Analysis")

col1, col2 = st.columns(2)

with col1:

    transaction_id = st.text_input(
        "Transaction ID",
        value="TXN-DEMO-001"
    )

    amount = st.number_input(
        "Transaction Amount (₹)",
        min_value=1.0,
        max_value=50000.0,
        value=2500.0,
        step=100.0
    )

    transaction_hour = st.slider(
        "Transaction Hour",
        min_value=0,
        max_value=23,
        value=14
    )

    transactions_last_24h = st.number_input(
        "Transactions in Last 24 Hours",
        min_value=0,
        max_value=30,
        value=3,
        step=1
    )


with col2:

    account_age_days = st.number_input(
        "Account Age (days)",
        min_value=1,
        max_value=2000,
        value=365,
        step=1
    )

    device_changes_7d = st.number_input(
        "Device Changes in Last 7 Days",
        min_value=0,
        max_value=8,
        value=0,
        step=1
    )

    location_changes_7d = st.number_input(
        "Location Changes in Last 7 Days",
        min_value=0,
        max_value=8,
        value=0,
        step=1
    )


st.divider()


# -----------------------------
# Analyze Transaction
# -----------------------------
if st.button(
    "🚀 Analyze Transaction",
    type="primary",
    width="stretch"
):

    transaction = {
        "transaction_id": transaction_id,
        "amount": amount,
        "transaction_hour": transaction_hour,
        "transactions_last_24h": transactions_last_24h,
        "account_age_days": account_age_days,
        "device_changes_7d": device_changes_7d,
        "location_changes_7d": location_changes_7d
    }

    try:

        result = analyze_via_api(transaction)

        # -----------------------------
        # Save Audit Record
        # -----------------------------
        new_record = pd.DataFrame([{
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "transaction_id": result["transaction_id"],
            "amount": amount,
            "risk_score": result["risk_score"],
            "risk_level": result["risk_level"],
            "fraud_probability": result["fraud_probability"]
        }])

        updated_history = pd.concat(
            [history_df, new_record],
            ignore_index=True
        )

        updated_history.to_csv(
            AUDIT_FILE,
            index=False
        )

        # Refresh history so the dashboard summary updates
        history_df = load_history()


        # -----------------------------
        # Risk Assessment
        # -----------------------------
        st.header("📊 Risk Assessment")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.metric(
                "Fraud Probability",
                f"{result['fraud_probability'] * 100:.2f}%"
            )

        with result_col2:
            st.metric(
                "Risk Score",
                f"{result['risk_score']}/100"
            )

        with result_col3:
            st.metric(
                "Risk Level",
                result["risk_level"]
            )
        st.subheader("⚡ Recommended Action")

        if result["recommended_action"] == "ALLOW":
           st.success("✅ ALLOW — Transaction can proceed.")

        elif result["recommended_action"] == "ADDITIONAL VERIFICATION":
            st.warning("⚠️ ADDITIONAL VERIFICATION — Verify the transaction before proceeding.")

        else:
            st.error("🚨 HOLD FOR REVIEW — Transaction requires manual review.")


        # -----------------------------
        # Risk Message
        # -----------------------------
        if result["risk_level"] == "HIGH":

            st.error(
                "🚨 HIGH RISK — Transaction requires review."
            )

        elif result["risk_level"] == "MEDIUM":

            st.warning(
                "⚠️ MEDIUM RISK — Additional verification recommended."
            )

        else:

            st.success(
                "✅ LOW RISK — No major risk detected."
            )


        # -----------------------------
        # Risk Reasons
        # -----------------------------
        st.subheader("🧠 Why was this transaction flagged?")

        for reason in result["reasons"]:
            st.write(f"• {reason}")


        # -----------------------------
        # Transaction Details
        # -----------------------------
        st.subheader("📋 Transaction Details")

        details_col1, details_col2 = st.columns(2)

        with details_col1:

            st.write(
                f"**Transaction ID:** {result['transaction_id']}"
            )

            st.write(
                f"**Amount:** ₹{amount:,.2f}"
            )

            st.write(
                f"**Transaction hour:** {transaction_hour}:00"
            )

        with details_col2:

            st.write(
                f"**Transactions / 24h:** "
                f"{transactions_last_24h}"
            )

            st.write(
                f"**Account age:** {account_age_days} days"
            )

            st.write(
                f"**Device changes:** {device_changes_7d}"
            )

            st.write(
                f"**Location changes:** {location_changes_7d}"
            )


        st.success(
            "✅ Transaction saved to the audit log."
        )


    except Exception as error:

        st.error(
            f"Unable to analyze transaction: {error}"
        )


# -----------------------------
# Risk Trend
# -----------------------------
st.divider()

st.header("📈 Risk Trend")

if not history_df.empty:
    chart_df = history_df.copy()

    chart_df["Time"] = pd.to_datetime(chart_df["time"])
    chart_df = chart_df.set_index("Time")

    st.line_chart(
        chart_df["risk_score"],
        width="stretch"
    )
else:
    st.info("Analyze some transactions to see the risk trend.")
# -----------------------------
# Fraud Spike Detection
# -----------------------------
st.divider()

st.header("🚨 Fraud Spike Detection")

st.write(
    "RazorGuard monitors recent transactions and detects "
    "unusual concentrations of risky transactions."
)

spike_result = detect_fraud_spike(
    history_df,
    window_minutes=60,
    threshold=0.20
)

spike_col1, spike_col2, spike_col3 = st.columns(3)

with spike_col1:
    st.metric(
        "Recent Transactions",
        spike_result["transaction_count"]
    )

with spike_col2:
    st.metric(
        "Risk Rate",
        f"{spike_result['risk_rate'] * 100:.2f}%"
    )

with spike_col3:
    st.metric(
        "Spike Status",
        spike_result["status"]
    )

if spike_result["status"] == "SPIKE DETECTED":

    st.error(
        "🚨 FRAUD-RISK SPIKE DETECTED"
    )

    st.write(
        spike_result["message"]
    )

elif spike_result["status"] == "NORMAL":

    st.success(
        "✅ No significant fraud-risk spike detected."
    )

else:

    st.info(
        "ℹ️ Not enough recent transaction data for spike detection."
    )
st.divider()

st.header("📜 Transaction History")

history_df = load_history()

if not history_df.empty:

    display_df = history_df.copy()

    display_df["risk_score"] = display_df["risk_score"].round(2)

    display_df["fraud_probability"] = (
        display_df["fraud_probability"] * 100
    ).round(2)

    display_df = display_df.rename(
        columns={
            "time": "Time",
            "transaction_id": "Transaction ID",
            "amount": "Amount (₹)",
            "risk_score": "Risk Score",
            "risk_level": "Risk Level",
            "fraud_probability": "Fraud Probability (%)"
        }
    )

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True
    )

else:

    st.info(
        "No transactions analyzed yet."
    )


# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "RazorGuard AI • Defensive fraud-risk detection prototype"
)
# -----------------------------
# Model Performance
# -----------------------------
st.divider()

st.header("🤖 Model Performance")

st.write(
    "RazorGuard AI was evaluated on a held-out test set "
    "that was not used for model training."
)
# Load saved model evaluation results
model_package = joblib.load(
    PROJECT_ROOT / "models" / "razorguard_model.joblib"
)

precision = model_package["precision"]
recall = model_package["recall"]
f1 = model_package["f1"]
test_transactions = model_package["test_transactions"]
threshold = model_package["threshold"]
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

with metric_col1:
    st.metric(
        "Precision", 
        f"{precision * 100:.2f}%"
    )

with metric_col2:
    st.metric(
        "Recall", 
        f"{recall * 100:.2f}%"
    )

with metric_col3:
    st.metric(
        "F1 Score",
        f"{f1 * 100:.2f}%"
    )

with metric_col4:
    st.metric(
        "Test Transactions",
        f"{test_transactions:,}"
    )
st.metric(
    "Decision Threshold",
    f"{threshold:.2f}"
)

st.info(
    "These metrics are loaded directly from the latest "
    "held-out test evaluation."
)
# -----------------------------
# Confusion Matrix
# -----------------------------
st.subheader("📌 Confusion Matrix")

confusion = model_package["confusion_matrix"]

cm_col1, cm_col2 = st.columns(2)

with cm_col1:
    st.metric("True Negatives", confusion[0][0])
    st.metric("False Positives", confusion[0][1])

with cm_col2:
    st.metric("False Negatives", confusion[1][0])
    st.metric("True Positives", confusion[1][1])    





# -----------------------------
# Risk Cost Analysis
# -----------------------------
st.divider()

st.header("💰 Risk Cost Analysis")

st.write(
    "Estimated operational cost of incorrect fraud decisions "
    "using the held-out test results."
)

false_positive_cost = 50
false_negative_cost = 1000

# Current held-out confusion matrix:
# [[1750, 71],
#  [65, 114]]

false_positives = confusion[0][1]
false_negatives = confusion[1][0]

estimated_fp_cost = false_positives * false_positive_cost
estimated_fn_cost = false_negatives * false_negative_cost
total_estimated_cost = estimated_fp_cost + estimated_fn_cost

cost_col1, cost_col2, cost_col3 = st.columns(3)

with cost_col1:
    st.metric(
        "False Positive Cost",
        f"₹{estimated_fp_cost:,}"
    )

with cost_col2:
    st.metric(
        "False Negative Cost",
        f"₹{estimated_fn_cost:,}"
    )

with cost_col3:
    st.metric(
        "Estimated Total Cost",
        f"₹{total_estimated_cost:,}"
    )

st.caption(
    "Illustrative cost assumptions: ₹50 per false positive "
    "and ₹1,000 per false negative."
)