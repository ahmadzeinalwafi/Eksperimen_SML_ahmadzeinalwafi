import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import mlflow

def preprocess_dataset(input_path: str, output_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads a dataset, removes duplicates and outliers, scales features,
    performs train-test split, and saves the cleaned datasets as CSVs.
    """
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

    os.makedirs(output_dir, exist_ok=True)

    # Save cleaned dataset
    cleaned_path = os.path.join(output_dir, "dataset_clean.csv")
    df.to_csv(cleaned_path, index=False)

    # Split and save train/test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_path = os.path.join(output_dir, "train_data.csv")
    test_path = os.path.join(output_dir, "test_data.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    return train_df, test_df, cleaned_path, train_path, test_path

if __name__ == "__main__":
    input_file = "preprocessing/dataset.csv"
    output_dir = "preprocessing/outputs"

    mlflow.set_tracking_uri("file:./mlruns")

    with mlflow.start_run(run_name="Preprocessing_Run"):
        train_df, test_df, cleaned_path, train_path, test_path = preprocess_dataset(input_file, output_dir)

        mlflow.log_param("input_file", input_file)
        mlflow.log_param("output_dir", output_dir)
        mlflow.log_metric("rows_after_cleaning_train", train_df.shape[0])
        mlflow.log_metric("rows_after_cleaning_test", test_df.shape[0])

        mlflow.log_artifact(cleaned_path)
        mlflow.log_artifact(train_path)
        mlflow.log_artifact(test_path)
