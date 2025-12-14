from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('smart_spend_model.pkl')

# Define input schema
class TransactionInput(BaseModel):
    user_id: int
    year: int
    month: int
    transaction_count: int
    avg_transaction: float
    # add more features if needed

app = FastAPI()

@app.post("/predict")
def predict_spend(data: TransactionInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return {"predicted_monthly_spend": prediction}
