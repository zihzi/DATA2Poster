import json
from typing import Union
import pandas as pd
import warnings


def check_type(dtype: str, value):
        """Cast value to right type to ensure it is JSON serializable"""
        if "float" in str(dtype):
            return float(value)
        elif "int" in str(dtype):
            return int(value)
        else:
            return value

def get_column_properties(df: pd.DataFrame, n_samples: int = 3) -> list[dict]:
        """Get properties of each column in a pandas DataFrame"""
        properties_list = []
        temporal_keywords = ["year", "month", "date", "time", "week"]
        for column in df.columns:
            dtype = df[column].dtype
            properties = {}
            # Detect columns that are likely to represent temporal data.
            if any(keyword in column.lower() for keyword in temporal_keywords):
                properties["data_type"] = dtype
                properties["dtype"] = "T"
            elif dtype in [int, float, complex]:
                properties["data_type"] = dtype
                properties["dtype"] = "N"
                properties["std"] = check_type(dtype, df[column].std())
                properties["mean"] = check_type(dtype, df[column].mean())
                properties["median"] = check_type(dtype, df[column].median())
                properties["min"] = check_type(dtype, df[column].min())
                properties["max"] = check_type(dtype, df[column].max())
            elif dtype == bool:
                properties["dtype"] = "boolean"
            elif dtype == object:
                # Check if the string column can be cast to a valid datetime
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pd.to_datetime(df[column], errors='raise')
                        properties["dtype"] = "T"
                except ValueError:
                    # Check if the string column has a limited number of values
                    if len(df[column])>=8 and df[column].nunique() / len(df[column]) < 0.5:
                        properties["dtype"] = dtype
                        properties["dtype"] = "C"
                    elif len(df[column]) < 8 and df[column].nunique() / len(df[column]) > 0.5:
                        properties["dtype"] = dtype
                        properties["dtype"] = "C"
                    else:
                        properties["dtype"] = "string"
            elif pd.api.types.is_categorical_dtype(df[column]):
                properties["dtype"] = "C"
            elif pd.api.types.is_datetime64_any_dtype(df[column]):
                properties["dtype"] = "T"
            
            else:
                properties["dtype"] = str(dtype)

            # add min max if dtype is date
            if properties["dtype"] == "T":
                try:
                    properties["min"] = df[column].min()
                    properties["max"] = df[column].max()
                except TypeError:
                    cast_date_col = pd.to_datetime(df[column], errors='coerce')
                    properties["min"] = cast_date_col.min()
                    properties["max"] = cast_date_col.max()
            # Add additional properties to the output dictionary
            nunique = df[column].nunique()
            if "samples" not in properties:
                non_null_values = df[column][df[column].notnull()].unique()
                n_samples = min(n_samples, len(non_null_values))
                samples = pd.Series(non_null_values).sample(
                    n_samples, random_state=42).tolist()
                properties["samples"] = samples
            properties["num_unique_values"] = nunique
            # its value is generated by llm in lida
            # properties["semantic_type"] = ""
            # properties["description"] = ""
            properties_list.append(
                {"column": column, "properties": properties})
        
        return properties_list