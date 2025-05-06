import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import mlflow

def preprocess_dataset(input_path: str, output_path: str) -> pd.DataFrame:
    """Loads a dataset, removes duplicates and outliers, scales features, and saves the cleaned dataset."""
    df = pd.read_csv(input_path)

    df = df.drop_duplicates()

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

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    return df

if __name__ == "__main__":
    input_file = "preprocessing/dataset.csv"
    output_file = "preprocessing/outputs/dataset_clean.csv"

    with mlflow.start_run(run_name="Preprocessing_Run"):
        df_clean = preprocess_dataset(input_file, output_file)

        mlflow.log_param("input_file", input_file)
        mlflow.log_param("output_file", output_file)
        mlflow.log_metric("rows_after_cleaning", df_clean.shape[0])
        mlflow.log_artifact(output_file)