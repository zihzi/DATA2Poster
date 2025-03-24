import pandas as pd
import numpy as np

def clean_csv_data(df):
    """
    Cleans a CSV file before exploratory data analysis (EDA).

    Steps:
    1. Load the dataset.
    2. Remove duplicate rows.
    3. Handle missing values.
    4. Convert data types.
    5. Standardize categorical values (remove whitespace, lowercase).
    6. Save the cleaned dataset.

    Args:
        input (dataframe): dataframe of the input CSV file.
        output(dataframe): dataframe of the cleaned CSV file.
    """
   
    # Step 1: Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # Step 2: Handle missing values
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna("Unknown", inplace=True)  # Fill missing categorical values
        else:
            df[col].fillna(df[col].median(), inplace=True)  # Fill missing numeric values with median

    # Step 3: Convert data types
    for col in df.columns:
        if df[col].dtype == "object":
            try:
                df[col] = pd.to_datetime(df[col])  # Convert date-like strings to datetime
            except:
                pass
        elif df[col].dtype == "int64" or df[col].dtype == "float64":
            df[col] = pd.to_numeric(df[col], errors="coerce")  # Ensure numeric format
     # Step 4: Handle outliers using IQR
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), df[col].median(), df[col])


    # Step 5: Standardize categorical/text columns
    categorical_cols = df.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        df[col] = df[col].str.strip().str.lower()  # Remove whitespace & lowercase

    print("Data cleaning done.")
    return df
