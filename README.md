
# Salary Prediction Application

A pre‑generation architecture application that predicts data science job salaries using a trained Decision Tree model, generates human‑readable insights via a local LLM (Ollama), stores results in Supabase, and presents an interactive dashboard built with Streamlit.

## 🚀 Live Deliverables

- **Streamlit Dashboard**: [https://salary-prediction-app-9detb9yjvs2jgzemnmf25f.streamlit.app/ ](https://salary-prediction-app-9detb9yjvs2jgzemnmf25f.streamlit.app/)

- **FastAPI Endpoint**: [https://salary-prediction-app-zxkx.onrender.com](https://salary-prediction-app-zxkx.onrender.com)

## 📐 Architecture Overview
Local pipeline Cloud storage Deployed dashboard
───────────────── ───────────── ─────────────────
Data cleaning → Model training → Supabase (Postgres) → Streamlit
↓ ↑
Prediction API (FastAPI) ────────────────────────────────────────┘
↓
LLM (Ollama) → Narrative + Chart

text

- The **local pipeline** (`src/data_cleaning.py`, `src/train_model.py`) cleans the dataset and trains a Decision Tree regressor.
- A **local FastAPI** (later deployed) serves the model via a GET endpoint.
- The **Phase 4 script** (`src/predict_and_analyze.py`) calls the API, enriches the prediction with an LLM narrative and chart, and stores everything in Supabase.
- The **Streamlit dashboard** reads from Supabase and displays the analyses.

## 🛠️ Tech Stack

- **Python 3.11+** – core language
- **FastAPI** – prediction API
- **scikit‑learn** – Decision Tree Regressor + preprocessing
- **Ollama** (local LLM, model `llama3.2`) – narrative generation
- **Supabase** (PostgreSQL) – data storage
- **Streamlit** – interactive dashboard
- **Render** – FastAPI deployment
- **Streamlit Cloud** – dashboard deployment

## 📦 Setup Instructions (Local Development)

### 1. Clone the repository

```bash
git clone https://github.com/mohammadabouamoun/salary-prediction-app.git
cd salary-prediction-app
2. Create and activate a virtual environment
bash
python3.11 -m venv .venv
source .venv/bin/activate   # Linux/macOS/WSL
# .venv\Scripts\activate   on Windows
3. Install dependencies
bash
pip install --upgrade pip
pip install -r requirements.txt
4. Set up environment variables
Create a .env file (never commit it) with:

text
SUPABASE_URL=https://isqbkrexuvygkwefrgqs.supabase.co
SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImlzcWJrcmV4dXZ5Z2t3ZWZyZ3FzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzU1ODUzMDEsImV4cCI6MjA5MTE2MTMwMX0.LL8u55yfkrDXKoa-zwp53V17D7h8ieLri6J2Tg07Acs
OLLAMA_HOST=http://localhost:11434
(Also provide a .env.example file for reference.)

5. Prepare the dataset
Download the dataset from Kaggle – Data Science Job Salaries and place ds_salaries.csv inside the data/ folder.

6. Run the data cleaning script
bash
python src/data_cleaning.py
This creates data/cleaned_salaries.csv.

7. Train the LightGBM model
bash
python src/train_model.py
Expected output:

text
📊 Model Performance (LightGBM with engineered features):
MAE: $29,489
R² Score: 0.476
✅ Model saved to models/salary_predictor.pkl

8. Start the FastAPI prediction server (local)
bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
Test with curl or open http://localhost:8000/predict?experience_level=SE&...

9. Install and run Ollama (local LLM)
bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2
ollama serve   # keep this terminal open
10. Run the end‑to‑end pipeline
In a separate terminal (with venv activated):

bash
python src/predict_and_analyze.py
This calls the API, generates a narrative and chart, and stores the result in Supabase.

11. Launch the Streamlit dashboard (local)
bash
streamlit run src/dashboard.py
Visit http://localhost:8501 to see the dashboard.

📄 API Documentation
The deployed FastAPI endpoint accepts GET requests at /predict with the following query parameters (all required):

Parameter	Type	Description	Example
experience_level	string	EN, MI, SE, EX	SE
employment_type	string	FT, PT, CT, FL	FT
job_title	string	Known titles (see source for list)	Data Scientist
remote_ratio	int	0, 50, 100	100
company_size	string	S, M, L	L
employee_residence	string	2‑letter country code	US
company_location	string	2‑letter country code	US
Example request:

bash
curl "https://salary-prediction-app-zxkx.onrender.com/predict?experience_level=MI&employment_type=FT&job_title=Data%20Engineer&remote_ratio=50&company_size=M&employee_residence=DE&company_location=DE"
Example response: 

json
  {"predicted_salary_usd":50134.00}


📁 Project Structure
text
salary-prediction-app/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── ds_salaries.csv           # raw dataset (ignored by git)
│   └── cleaned_salaries.csv      # cleaned dataset
├── models/
│   ├── salary_predictor.pkl      # trained pipeline
│   └── feature_columns.txt
├── src/
│   ├── data_cleaning.py
│   ├── train_model.py
│   ├── api.py
│   ├── predict_and_analyze.py
│   └── dashboard.py
└── tests/                        # (optional)
🔒 Security & Best Practices
Secrets are stored in .env (ignored by Git) or as Streamlit Cloud secrets.

Input validation is enforced on the API (enums, length checks).

The LLM prompt is designed to avoid injection.

The Decision Tree model is saved and loaded with joblib.

All dependencies are pinned for reproducibility.

🧪 Testing
Run tests (if any) with:

bash
pytest tests/
Coverage (minimum 80% recommended):

bash
pytest --cov=src --cov-report=term-missing