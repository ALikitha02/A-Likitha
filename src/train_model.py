import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# -----------------------------
# 1. Load dataset
# -----------------------------

data = pd.read_csv("data/transactions.csv")

features = [
    "amount",
    "transaction_hour",
    "transactions_last_24h",
    "account_age_days",
    "device_changes_7d",
    "location_changes_7d"
]

X = data[features]
y = data["is_fraud"]


# -----------------------------
# 2. Train / Validation / Test
# -----------------------------

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

print(f"Training transactions: {len(X_train)}")
print(f"Validation transactions: {len(X_val)}")
print(f"Held-out test transactions: {len(X_test)}")


# -----------------------------
# 3. Train Random Forest
# -----------------------------

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    min_samples_leaf=3,
    n_jobs=-1
)

model.fit(X_train, y_train)


# -----------------------------
# 4. Get validation probabilities
# -----------------------------

validation_probabilities = model.predict_proba(X_val)[:, 1]


# -----------------------------
# 5. Find best threshold using validation data
# -----------------------------

best_threshold = 0.50
best_f1 = 0

for threshold in [i / 100 for i in range(10, 91)]:

    validation_predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    score = f1_score(
        y_val,
        validation_predictions,
        zero_division=0
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold


print("\n===== THRESHOLD SELECTION =====")
print(f"Selected threshold: {best_threshold:.2f}")
print(f"Validation F1: {best_f1:.4f}")


# -----------------------------
# 6. Evaluate different thresholds
# -----------------------------

test_probabilities = model.predict_proba(X_test)[:, 1]

print("\n===== THRESHOLD COMPARISON =====")

thresholds_to_test = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80
]

for threshold in thresholds_to_test:

    predictions = (
        test_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    print(
        f"Threshold {threshold:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"F1: {f1:.3f}"
    )


# -----------------------------
# 7. Final held-out test result
# -----------------------------

y_pred = (
    test_probabilities >= best_threshold
).astype(int)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n===== RAZORGUARD AI TEST RESULTS =====")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


# -----------------------------
# 8. Classification report
# -----------------------------

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Fraud"],
        zero_division=0
    )
)


# -----------------------------
# 9. Confusion matrix
# -----------------------------

print("===== CONFUSION MATRIX =====")

print(confusion_matrix(y_test, y_pred))


# -----------------------------
# 10. Save model
# -----------------------------

model_package = {
    "model": model,
    "features": features,
    "threshold": best_threshold,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "test_transactions": len(X_test)
}

joblib.dump(
    model_package,
    "models/razorguard_model.joblib"
)

print("\nModel saved successfully!")
print("Location: models/razorguard_model.joblib")