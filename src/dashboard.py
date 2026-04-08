import streamlit as st
import pandas as pd
import base64
import io
from PIL import Image
from supabase import create_client
from dotenv import load_dotenv
import os

load_dotenv()

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

st.set_page_config(page_title="Salary Prediction Dashboard", layout="wide")
st.title("📊 Data Science Salary Prediction Dashboard")
st.markdown("Displays predictions, LLM narratives, and charts from the salary analysis pipeline.")

@st.cache_data(ttl=60)
def fetch_analyses(limit=50):
    response = supabase.table("salary_analyses").select("*").order("created_at", desc=True).limit(limit).execute()
    return response.data

data = fetch_analyses()

if not data:
    st.warning("No data found in Supabase. Run `src/predict_and_analyze.py` first.")
else:
    for record in data:
        with st.expander(f"Analysis #{record['id']} - ${record['predicted_salary']:,} - {record['created_at']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Input Details")
                st.json(record['input_data'])
                st.subheader("LLM Narrative")
                st.write(record['llm_narrative'])
            with col2:
                st.subheader("Salary Chart")
                if record.get('chart_base64'):
                    img_data = base64.b64decode(record['chart_base64'])
                    img = Image.open(io.BytesIO(img_data))
                    st.image(img, use_container_width=True)
                else:
                    st.info("No chart available")
    
    st.subheader("Aggregate Insights")
    df_inputs = pd.DataFrame([r['input_data'] for r in data])
    if not df_inputs.empty:
        avg_salary = pd.DataFrame([r['predicted_salary'] for r in data]).mean()[0]
        st.metric("Average Predicted Salary", f"${avg_salary:,.0f}")
        job_counts = df_inputs['job_title'].value_counts().head(5)
        st.bar_chart(job_counts)