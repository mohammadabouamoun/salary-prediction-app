import pandas as pd

 def load_and_clean_data(file_path: str = "data/ds_salaries.csv") -> pd.DataFrame:
    df = pd.read_csv(file_path)
    
    # Drop unnecessary columns
    cols_to_drop = ['Unnamed: 0', 'salary', 'salary_currency']
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    
    # No renaming needed – columns are  clean 
    
    # Drop missing values if any
    df.dropna(inplace=True)
    
    # Convert categorical columns to category type (saves memory)
    categorical_cols = ['experience_level', 'employment_type', 'job_title', 
                    'employee_residence', 'company_location', 'company_size']
    for col in categorical_cols:
        df[col] = df[col].astype('category')
    
 return df