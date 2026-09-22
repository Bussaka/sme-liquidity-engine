from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn

# Initialize the API
app = FastAPI(title="SME Liquidity Risk Engine")

print("Booting up the intelligence engine...")

# Load the serialized deployment package
try:
    model_package = joblib.load('liquidity_risk_engine.pkl')
    hgb_model = model_package['model']
    optimal_threshold = model_package['threshold']
    expected_features = model_package['features']
    print("Model loaded successfully. Ready for incoming vectors.")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load the .pkl file. Details: {e}")

# Define the expected structure of incoming web requests
class BusinessLedgerPayload(BaseModel):
    features: dict

@app.post("/evaluate_risk")
def evaluate_risk(data: BusinessLedgerPayload):
    # Convert the incoming JSON into a pandas DataFrame
    df = pd.DataFrame([data.features])
    
    # Validation: Ensure the payload isn't missing any critical columns
    missing_cols = [col for col in expected_features if col not in df.columns]
    if missing_cols:
        raise HTTPException(
            status_code=400, 
            detail=f"Payload rejected. Missing required features: {missing_cols}"
        )
        
    # Reorder columns to match the exact mathematical structure of the training data
    df = df[expected_features]
    
    # The predictor is applied to the new vectors
    probability = hgb_model.predict_proba(df)[:, 1][0]
    
    # Apply the aggressive Day 16 operational threshold
    is_stressed = int(probability >= optimal_threshold)
    
    # Return the actionable intelligence
    return {
        "status": "success",
        "liquidity_risk_probability": round(float(probability), 4),
        "stress_flag": is_stressed,
        "action_required": "FLAG_FOR_REVIEW" if is_stressed == 1 else "SAFE"
    }

if __name__ == "__main__":
    # Run the server locally on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)