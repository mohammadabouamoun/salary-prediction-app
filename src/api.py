import os
import joblib
import pandas as pd
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from enum import Enum

# Load model once at startup
MODEL_PATH = "models/salary_predictor.pkl"
if not os.path.exists(MODEL_PATH):
    raise RuntimeError("Model not found. Run train_model.py first.")
model = joblib.load(MODEL_PATH)

# Validation enums
class ExperienceLevel(str, Enum):
    EN = "EN"
    MI = "MI"
    SE = "SE"
    EX = "EX"

class EmploymentType(str, Enum):
    FT = "FT"
    PT = "PT"
    CT = "CT"
    FL = "FL"

class CompanySize(str, Enum):
    S = "S"
    M = "M"
    L = "L"

# Known job titles 
KNOWN_JOB_TITLES = [
    "Data Scientist", "Data Engineer", "Data Analyst", "Machine Learning Engineer",
    "Research Scientist", "Data Architect", "Business Analyst", "Analytics Engineer",
    "Cloud Engineer", "BI Analyst", "Data Science Manager", "Data Specialist"
]

app = FastAPI(title="Salary Prediction API", description="Decision Tree with engineered features")

class PredictionResponse(BaseModel):
    predicted_salary_usd: float


@app.get("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_salary(
    experience_level: ExperienceLevel = Query(...),
    employment_type: EmploymentType = Query(...),
    job_title: str = Query(...),
    remote_ratio: int = Query(..., ge=0, le=100, description="0, 50, or 100"),
    company_size: CompanySize = Query(...),
    employee_residence: str = Query(..., min_length=2, max_length=2),
    company_location: str = Query(..., min_length=2, max_length=2)
):
    # Validate job title
    if job_title not in KNOWN_JOB_TITLES:
        raise HTTPException(status_code=400, detail=f"Invalid job_title. Must be one of {KNOWN_JOB_TITLES}")
    
    # Derive engineered features (same as training)
    remote_type_map = {0: "Onsite", 50: "Hybrid", 100: "Remote"}
    remote_type = remote_type_map.get(remote_ratio, "Unknown")
    same_country = 1 if employee_residence == company_location else 0
    
    # Create input DataFrame with the exact columns the model expects
    input_data = pd.DataFrame([{
        'experience_level': experience_level.value,
        'employment_type': employment_type.value,
        'job_title': job_title,
        'company_size': company_size.value,
        'employee_residence': employee_residence,
        'company_location': company_location,
        'remote_type': remote_type,
        'same_country': same_country
    }])
    
    try:
        pred = model.predict(input_data)[0]
        return PredictionResponse(predicted_salary_usd=round(pred, 2))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)