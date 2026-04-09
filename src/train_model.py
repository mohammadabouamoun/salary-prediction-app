import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor



def train_salary_model(
    data_path: str = "data/cleaned_salaries.csv",
    model_path: str = "models/salary_predictor.pkl"
):
    # ==============================
    # 1. Load Data
    # ==============================
    df = pd.read_csv(data_path)

    # ==============================
    # 2. Basic Cleaning
    # ==============================
    # Remove extreme outliers (salaries > $300k)
    df = df[df["salary_in_usd"] < 300000]

    # Drop missing values (if any)
    df = df.dropna()

    # ==============================
    # 3. Feature Engineering
    # ==============================
    # Remote as categorical
    df["remote_type"] = df["remote_ratio"].map({
        0: "Onsite",
        50: "Hybrid",
        100: "Remote"
    })

    # Same country feature
    df["same_country"] = (
        df["employee_residence"] == df["company_location"]
    ).astype(int)

    # ==============================
    # 4. Features & Target
    # ==============================
    feature_cols = [
        'experience_level',
        'employment_type',
        'job_title',
        'company_size',
        'employee_residence',
        'company_location',
        'remote_type',
        'same_country'
    ]

    target_col = 'salary_in_usd'

    X = df[feature_cols]
    y = df[target_col]

    # ==============================
    # 5. Column Types
    # ==============================
    categorical_cols = [
        'experience_level',
        'employment_type',
        'job_title',
        'company_size',
        'employee_residence',
        'company_location',
        'remote_type'
    ]

    numerical_cols = ['same_country']

    # ==============================
    # 6. Preprocessing (OneHotEncoder)
    # ==============================
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', 'passthrough', numerical_cols)
        ]
    )

    # ==============================
    # 7. Model: LGBMRegressor
    # ==============================
    model = LGBMRegressor(
        n_estimators=300,          
         max_depth=15,              
         learning_rate=0.1,         
         random_state=42,
         verbose=-1                 
    )

    # ==============================
    # 8. Pipeline
    # ==============================
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    # ==============================
    # 9. Train-Test Split
    # ==============================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ==============================
    # 10. Train Model
    # ==============================
    pipeline.fit(X_train, y_train)

    # ==============================
    # 11. Evaluate
    # ==============================
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Model Performance (LightGBM with engineered features):")
    print(f"MAE: ${mae:,.0f}")
    print(f"R² Score: {r2:.3f}")

    # ==============================
    # 12. Save Model
    # ==============================
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)

    print(f"\n✅ Model saved to {model_path}")

    # ==============================
    # 13. Save Features
    # ==============================
    with open("models/feature_columns.txt", "w") as f:
        f.write("\n".join(feature_cols))

    return pipeline


if __name__ == "__main__":
    train_salary_model()