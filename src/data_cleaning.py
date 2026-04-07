import pandas as pd

def load_and_clean_data(file_path: str = "data/ds_salaries.csv") -> pd.DataFrame:
    """Load raw dataset, clean it, and return a DataFrame."""
    df = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    cols_to_drop = ['Unnamed: 0', 'salary', 'salary_currency']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    # Drop missing values if any
    df.dropna(inplace=True)
    
    # Convert categorical columns to category type (saves memory)
    categorical_cols = ['experience_level', 'employment_type', 'job_title',
                        'employee_residence', 'company_location', 'company_size']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    return df

if __name__ == "__main__":
    df = load_and_clean_data()
    print(df.head())
    print(df.info())
    df.to_csv("data/cleaned_salaries.csv", index=False)
    print("Cleaned data saved to data/cleaned_salaries.csv")
