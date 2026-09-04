import pandas as pd
import numpy as np

# ============================================================
# RazorGuard AI - Realistic Synthetic Transaction Generator
# ============================================================

np.random.seed(42)

N = 10000

# ------------------------------------------------------------
# 1. Basic transaction information
# ------------------------------------------------------------

transaction_id = [
    f"TXN{i:06d}"
    for i in range(1, N + 1)
]

# Transaction amounts have a realistic right-skewed distribution
amount = np.random.lognormal(
    mean=4.0,
    sigma=0.85,
    size=N
)

amount = np.clip(amount, 5, 50000)

# Most transactions happen during normal hours,
# but transactions can occur at any time.
hour_probabilities = np.array([
    0.015, 0.012, 0.010, 0.008,
    0.008, 0.010, 0.020, 0.035,
    0.055, 0.070, 0.075, 0.080,
    0.080, 0.080, 0.075, 0.070,
    0.065, 0.065, 0.065, 0.060,
    0.050, 0.040, 0.030, 0.020
])

hour_probabilities = (
    hour_probabilities / hour_probabilities.sum()
)

transaction_hour = np.random.choice(
    np.arange(24),
    size=N,
    p=hour_probabilities
)

# ------------------------------------------------------------
# 2. Customer behavior
# ------------------------------------------------------------

transactions_last_24h = np.random.poisson(
    lam=3.2,
    size=N
)

transactions_last_24h = np.clip(
    transactions_last_24h,
    0,
    30
)

# Account age: many newer accounts, but also many old accounts
account_age_days = np.random.gamma(
    shape=2.2,
    scale=180,
    size=N
)

account_age_days = np.clip(
    account_age_days,
    1,
    2000
).astype(int)

# Device changes are usually uncommon
device_changes_7d = np.random.poisson(
    lam=0.35,
    size=N
)

device_changes_7d = np.clip(
    device_changes_7d,
    0,
    8
)

# Location changes are also usually uncommon
location_changes_7d = np.random.poisson(
    lam=0.45,
    size=N
)

location_changes_7d = np.clip(
    location_changes_7d,
    0,
    8
)

# ------------------------------------------------------------
# 3. Create several realistic fraud patterns
# ------------------------------------------------------------

# Start with a low baseline fraud probability.
fraud_probability = np.full(N, 0.025)


# Pattern A:
# High transaction velocity
high_velocity = transactions_last_24h >= 8
fraud_probability[high_velocity] += 0.10


# Pattern B:
# Very new account + relatively high activity
new_account = account_age_days <= 30
active_new_account = (
    new_account &
    (transactions_last_24h >= 5)
)

fraud_probability[active_new_account] += 0.12


# Pattern C:
# Multiple device changes
many_device_changes = device_changes_7d >= 3
fraud_probability[many_device_changes] += 0.09


# Pattern D:
# Multiple location changes
many_location_changes = location_changes_7d >= 3
fraud_probability[many_location_changes] += 0.08


# Pattern E:
# Unusual transaction time
night_transaction = (
    (transaction_hour <= 4) |
    (transaction_hour >= 23)
)

fraud_probability[night_transaction] += 0.05


# Pattern F:
# High-value transaction
high_amount = amount >= 1000
fraud_probability[high_amount] += 0.05


# Pattern G:
# Combination of suspicious signals
multiple_signals = (
    (transactions_last_24h >= 7) &
    (
        (device_changes_7d >= 2) |
        (location_changes_7d >= 2)
    )
)

fraud_probability[multiple_signals] += 0.14


# Pattern H:
# Some fraud can look relatively normal.
# Randomly select a small group and increase risk slightly.
hidden_risk_indices = np.random.choice(
    N,
    size=int(N * 0.025),
    replace=False
)

fraud_probability[hidden_risk_indices] += 0.045


# ------------------------------------------------------------
# 4. Add randomness
# ------------------------------------------------------------

# Legitimate users can sometimes behave unusually.
# Fraudulent behavior is therefore not perfectly separated.
random_noise = np.random.normal(
    loc=0,
    scale=0.012,
    size=N
)

fraud_probability += random_noise

fraud_probability = np.clip(
    fraud_probability,
    0.001,
    0.60
)


# ------------------------------------------------------------
# 5. Generate fraud labels
# ------------------------------------------------------------

is_fraud = np.random.binomial(
    n=1,
    p=fraud_probability,
    size=N
)


# ------------------------------------------------------------
# 6. Keep fraud rate in a useful range
# ------------------------------------------------------------

# We want approximately 5-8% fraud for this prototype.
current_fraud_rate = is_fraud.mean()

if current_fraud_rate < 0.05:

    required = int(N * 0.06)

    ranked_indices = np.argsort(
        fraud_probability
    )[::-1]

    is_fraud[ranked_indices[:required]] = 1

elif current_fraud_rate > 0.08:

    target = int(N * 0.06)

    fraud_indices = np.where(
        is_fraud == 1
    )[0]

    # Keep the highest-risk fraud examples.
    fraud_indices_sorted = fraud_indices[
        np.argsort(
            fraud_probability[fraud_indices]
        )[::-1]
    ]

    is_fraud[:] = 0

    selected = fraud_indices_sorted[:target]

    is_fraud[selected] = 1


# ------------------------------------------------------------
# 7. Build the dataframe
# ------------------------------------------------------------

data = pd.DataFrame({
    "transaction_id": transaction_id,
    "amount": np.round(amount, 2),
    "transaction_hour": transaction_hour,
    "transactions_last_24h": transactions_last_24h,
    "account_age_days": account_age_days,
    "device_changes_7d": device_changes_7d,
    "location_changes_7d": location_changes_7d,
    "is_fraud": is_fraud
})


# ------------------------------------------------------------
# 8. Save dataset
# ------------------------------------------------------------

data.to_csv(
    "data/transactions.csv",
    index=False
)


# ------------------------------------------------------------
# 9. Display useful information
# ------------------------------------------------------------

fraud_count = int(
    data["is_fraud"].sum()
)

normal_count = int(
    (data["is_fraud"] == 0).sum()
)

fraud_rate = (
    fraud_count / N
) * 100

print("==========================================")
print("RazorGuard AI Dataset Generated")
print("==========================================")

print(f"Total transactions : {N}")
print(f"Fraud transactions : {fraud_count}")
print(f"Normal transactions: {normal_count}")
print(f"Fraud rate         : {fraud_rate:.2f}%")

print("\nDataset columns:")
print(list(data.columns))

print("\nFirst 5 transactions:")
print(data.head())

print("\nFraud distribution:")
print(data["is_fraud"].value_counts())

print("\nDataset saved to:")
print("data/transactions.csv")