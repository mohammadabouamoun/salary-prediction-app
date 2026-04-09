import os
import sys
import json
import base64
import io
import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from supabase import create_client, Client
from typing import Dict, Any

# Load environment variables
load_dotenv()

# ==============================
# Configuration
# ==============================
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434") + "/api/generate"
PREDICTION_API_URL = "http://localhost:8000/predict"   # local FastAPI

# Validate required environment variables
if not SUPABASE_URL or not SUPABASE_KEY:
    sys.exit("Error: SUPABASE_URL and SUPABASE_KEY must be set in .env file")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ==============================
# Helper Functions
# ==============================
def call_prediction_api(input_params: Dict[str, Any]) -> float:
    """Call the local FastAPI prediction endpoint and return predicted salary."""
    try:
        response = requests.get(PREDICTION_API_URL, params=input_params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return float(data["predicted_salary_usd"])
    except requests.exceptions.RequestException as e:
        print(f"API call failed: {e}")
        sys.exit(1)

def get_llm_narrative(input_params: Dict[str, Any], predicted_salary: float) -> str:
    """Send job details and salary to Ollama, return narrative text."""
    prompt = f"""
You are a data analyst. Given the following job details and predicted salary, provide a short, insightful narrative (2-4 sentences) explaining the salary landscape for this role. Focus on how experience level, remote ratio, company size, and location affect compensation.

Job details:
- Experience level: {input_params['experience_level']}
- Employment type: {input_params['employment_type']}
- Job title: {input_params['job_title']}
- Remote ratio: {input_params['remote_ratio']}%
- Company size: {input_params['company_size']}
- Employee residence: {input_params['employee_residence']}
- Company location: {input_params['company_location']}

Predicted salary: ${predicted_salary:,.0f} USD

Narrative:
"""
    payload = {
        "model": "llama3.2",   # ensure you have pulled this model
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7}
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result["response"].strip()
    except Exception as e:
        print(f"Ollama request failed: {e}")
        return "LLM analysis unavailable."

def generate_salary_chart(input_params: Dict[str, Any], predicted_salary: float) -> str:
    """
    Create a bar chart comparing average salaries by experience level
    (from cleaned dataset) and highlight the predicted salary.
    Returns chart as base64 string.
    """
    try:
        df = pd.read_csv("data/cleaned_salaries.csv")
        avg_by_exp = df.groupby("experience_level")["salary_in_usd"].mean().to_dict()
    except FileNotFoundError:
        # Fallback if cleaned data missing
        avg_by_exp = {"EN": 60000, "MI": 90000, "SE": 130000, "EX": 180000}

    levels = ["EN", "MI", "SE", "EX"]
    averages = [avg_by_exp.get(lvl, 0) for lvl in levels]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(levels, averages, color='steelblue', alpha=0.7, label='Average Salary by Experience Level')
    
    # Highlight the bar corresponding to the input experience level
    exp_level = input_params['experience_level']
    if exp_level in levels:
        idx = levels.index(exp_level)
        bars[idx].set_color('darkorange')
        bars[idx].set_alpha(1.0)
    
    # Add horizontal line for predicted salary
    plt.axhline(y=predicted_salary, color='red', linestyle='--', linewidth=2,
                label=f'Predicted Salary: ${predicted_salary:,.0f}')
    
    plt.ylabel('Salary (USD)')
    plt.title('Salary Comparison by Experience Level')
    plt.legend()
    plt.tight_layout()
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_base64

def store_in_supabase(input_params: Dict[str, Any], predicted_salary: float,
                      narrative: str, chart_base64: str) -> None:
    """Insert the analysis record into Supabase."""
    record = {
        "input_data": input_params,
        "predicted_salary": int(predicted_salary),
        "llm_narrative": narrative,
        "chart_base64": chart_base64
    }
    try:
        result = supabase.table("salary_analyses").insert(record).execute()
        print(f"Stored successfully. ID: {result.data[0]['id'] if result.data else 'unknown'}")
    except Exception as e:
        print(f"Failed to store in Supabase: {e}")
        sys.exit(1)

# ==============================
# Main Execution
# ==============================
if __name__ == "__main__":
    # You can modify this sample input or read from command line / config file
    sample_input = {
         "experience_level": "MI",
         "employment_type": "FT",
         "job_title": "Data Engineer",
         "remote_ratio": 50,
         "company_size": "M",
         "employee_residence": "DE",
         "company_location": "DE"
    }
    
    print("=== Salary Prediction Pipeline ===")
    print(f"Input: {json.dumps(sample_input, indent=2)}")
    
    # 1. Get prediction from API
    print("\n📡 Calling prediction API...")
    salary = call_prediction_api(sample_input)
    print(f"💰 Predicted salary: ${salary:,.0f}")
    
    # 2. Generate LLM narrative
    print("\n🤖 Asking Ollama for narrative...")
    narrative = get_llm_narrative(sample_input, salary)
    print(f"📝 Narrative: {narrative}")
    
    # 3. Generate chart
    print("\n📊 Generating chart...")
    chart_b64 = generate_salary_chart(sample_input, salary)
    print("Chart created.")
    
    # 4. Store in Supabase
    print("\n💾 Saving to Supabase...")
    store_in_supabase(sample_input, salary, narrative, chart_b64)
    
    print("\n✅ Pipeline completed successfully.")