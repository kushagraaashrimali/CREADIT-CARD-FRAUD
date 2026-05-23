from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("fraud_model.pkl")

@app.post("/predict")
def predict(transaction: dict):
    data = np.array(list(transaction.values())).reshape(1, -1)
    prob = model.predict_proba(data)[0][1]
    return {"fraud_probability": prob}
