from fastapi import FastAPI
from pydantic import BaseModel

from src.risk_engine import analyze_transaction


app = FastAPI(
    title="RazorGuard AI",
    description="AI-powered defensive payment risk detection API",
    version="1.0.0"
)


class Transaction(BaseModel):
    transaction_id: str
    amount: float
    transaction_hour: int
    transactions_last_24h: int
    account_age_days: int
    device_changes_7d: int
    location_changes_7d: int


@app.get("/")
def home():
    return {
        "system": "RazorGuard AI",
        "status": "running",
        "message": "AI Risk Manager API is active"
    }


@app.post("/analyze")
def analyze(transaction: Transaction):

    transaction_data = transaction.model_dump()

    result = analyze_transaction(transaction_data)

    return result