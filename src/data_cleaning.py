import pandas as pd

def load_and_clean_data(file_path: str = "data/ds_salaries.csv") -> pd.DataFrame:
    """Load the raw dataset and perform cleaning."""
    df = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    cols_to_drop = ['Unnamed: 0', 'salary', 'salary_currency']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    # Rename columns for clarity
    df.rename(columns={
        'work_year': 'work_year',
        'experience_level': 'experience_level',
        'employment_type': 'employment_type',
        'job_title': 'job_title',
        'salary_in_usd': 'salary_in_usd',
        'employee_residence': 'employee_residence',
        'remote_ratio': 'remote_ratio',
        'company_location': 'company_location',
        'company_size': 'company_size'
    }, inplace=True)
    
    # Check for missing values
    if df.isnull().sum().any():
        df.dropna(inplace=True)
    
    # Convert categorical columns to appropriate types
    categorical_cols = ['experience_level', 'employment_type', 'job_title', 
                        'employee_residence', 'company_location', 'company_size']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
    return df

if __name__ == "__main__":
    df = load_and_clean_data()
    print(df.head())
    print(df.info())
    # Save cleaned data
    df.to_csv("data/cleaned_salaries.csv", index=False)
    print("Cleaned data saved to data/cleaned_salaries.csv")
