import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    """Loads a dataset, removes duplicates and outliers, scales features, and saves the cleaned dataset."""
    df = pd.read_csv(input_path)

    # Remove duplicates
    df = df.drop_duplicates()

    # Remove outliers using IQR
    def remove_outliers_iqr(dataframe, columns):
        for col in columns:
            Q1 = dataframe[col].quantile(0.25)
            Q3 = dataframe[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            dataframe = dataframe[(dataframe[col] >= lower) & (dataframe[col] <= upper)]
        return dataframe

    features = df.columns.drop('strength')
    df = remove_outliers_iqr(df, features)

    # Standard scaling
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    # Save cleaned dataset
    df.to_csv(output_path, index=False)
    return df

import mlflow

# Define file paths
input_file = "preprocessing/dataset.csv"
output_file = "preprocessing/dataset_clean.csv"

with mlflow.start_run(run_name="Preprocessing_Run"):
    # Preprocess and save dataset
    df_clean = preprocess_dataset(input_file, output_file)

    # Log parameters and artifacts
    mlflow.log_param("input_file", input_file)
    mlflow.log_param("output_file", output_file)
    mlflow.log_metric("rows_after_cleaning", df_clean.shape[0])
    mlflow.log_artifact(output_file)