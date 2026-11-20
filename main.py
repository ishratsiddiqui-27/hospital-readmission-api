
from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import os

# ---------------------------------------
# 1. INITIALISE FASTAPI
# ---------------------------------------

app = FastAPI(
    title="Hospital Readmission Risk API",
    description="Predicts the probability of readmission within 30 days.",
    version="2.0"
)

# ---------------------------------------
# 2. LOAD THE TRAINED MODEL PIPELINE
# ---------------------------------------

bundle = joblib.load("readmission_model.pkl")

model = bundle["model"]
feature_columns = bundle["feature_columns"]
threshold = bundle["threshold"]


# ---------------------------------------
# 3. HOME ENDPOINT
# ---------------------------------------

@app.get("/")
def home():
    return {
        "message": "Hospital Readmission Risk API is running",
        "model": "Random Forest Classifier",
        "target": "Readmission within 30 days",
        "documentation": "/docs"
    }


# ---------------------------------------
# 4. HEALTH CHECK
# ---------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy"}


# ---------------------------------------
# 5. MODEL INFORMATION
# ---------------------------------------

@app.get("/model-info")
def model_info():
    return {
        "model": "Random Forest Classifier",
        "target": bundle["target_description"],
        "threshold": threshold,
        "required_features": len(feature_columns),
        "features": feature_columns
    }


# ---------------------------------------
# 6. PREDICTION ENDPOINT
# ---------------------------------------

@app.post("/predict")
def predict(data: dict):

    # Check for missing features
    missing = [
        col for col in feature_columns
        if col not in data
    ]

    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Missing required features",
                "missing_features": missing
            }
        )

    try:
        # Arrange input in training column order
        input_data = pd.DataFrame(
            [[data[col] for col in feature_columns]],
            columns=feature_columns
        )

        # Run the complete pipeline
        probability = model.predict_proba(input_data)[0][1]

        # Apply threshold
        prediction = int(probability >= threshold)

        result = "High Risk" if prediction == 1 else "Low Risk"

        return {
            "readmission_prediction": prediction,
            "result": result,
            "readmission_probability": round(float(probability), 4),
            "risk_percentage": round(float(probability * 100), 2),
            "threshold_used": threshold,
            "target": "Readmission within 30 days",
            "message": (
                "Flag for clinical review"
                if prediction == 1
                else "No high-risk flag from this model"
            )
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
