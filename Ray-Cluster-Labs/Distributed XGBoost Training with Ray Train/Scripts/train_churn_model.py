import ray
from ray.train import ScalingConfig, RunConfig
from ray.train.xgboost import XGBoostTrainer
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import boto3
from botocore.client import Config
from io import BytesIO
import pickle
import urllib.request
import os
from pyarrow import fs

# MinIO Configuration from environment variables
MINIO_ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://10.0.1.199:9000")
MINIO_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
MINIO_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
BUCKET_NAME = "ray-data"
MODEL_OUTPUT_PATH = "models/churn_xgboost_model.pkl"

# Processed data URL
DATA_URL = "https://github.com/poridhioss/MLOps/raw/refs/heads/main/Ray-Cluster-Labs/Distributed%20XGBoost%20Training%20with%20Ray%20Train/processed-data/churned_customers.parquet"

# Initialize Ray
ray.init()

# Configure S3 client for boto3
s3_client = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

# Configure PyArrow S3 filesystem for Ray Train
endpoint_host = MINIO_ENDPOINT.replace("http://", "").replace("https://", "")
s3_fs = fs.S3FileSystem(
    endpoint_override=endpoint_host,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    scheme="http",
    region="us-east-1"
)

def load_data():
    """Download and prepare data"""
    urllib.request.urlretrieve(DATA_URL, "/tmp/churned_customers.parquet")
    df = pd.read_parquet("/tmp/churned_customers.parquet")
    return df

def prepare_features(df):
    """Prepare features for training"""
    feature_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
                      'AvgMonthlyCharges', 'IsHighValue']
    
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService',
                          'MultipleLines', 'InternetService', 'OnlineSecurity',
                          'OnlineBackup', 'DeviceProtection', 'TechSupport',
                          'StreamingTV', 'StreamingMovies', 'Contract',
                          'PaperlessBilling', 'PaymentMethod']
    
    df_encoded = df.copy()
    label_encoders = {}
    
    for col in categorical_columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
            feature_columns.append(col)
    
    df_encoded['target'] = (df_encoded['TotalCharges'] > df_encoded['TotalCharges'].median()).astype(int)
    
    return df_encoded[feature_columns + ['target']], label_encoders

def main():
    # Load and prepare data
    df = load_data()
    data_df, label_encoders = prepare_features(df)
    
    # Split data
    train_df, valid_df = train_test_split(data_df, test_size=0.2, random_state=42)
    
    # Create Ray datasets for distributed training
    train_dataset = ray.data.from_pandas(train_df)
    valid_dataset = ray.data.from_pandas(valid_df)
    
    # Configure distributed training with 2 workers
    trainer = XGBoostTrainer(
        scaling_config=ScalingConfig(
            num_workers=2,
            use_gpu=False,
            resources_per_worker={"CPU": 1}
        ),
        label_column="target",
        params={
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "error"],
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        },
        datasets={"train": train_dataset, "valid": valid_dataset},
        num_boost_round=100,
        run_config=RunConfig(
            storage_path="ray-data/checkpoints",
            storage_filesystem=s3_fs
        ),
    )
    
    # Execute distributed training
    result = trainer.fit()
    
    # Get trained model using Ray's built-in method
    booster = XGBoostTrainer.get_model(result.checkpoint)
    
    # Save to MinIO
    model_data = {'model': booster, 'label_encoders': label_encoders}
    model_buffer = BytesIO()
    pickle.dump(model_data, model_buffer)
    model_buffer.seek(0)
    
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=MODEL_OUTPUT_PATH,
        Body=model_buffer.getvalue()
    )
    
    print(f"Training completed. Metrics: {result.metrics}")
    print(f"Model saved to s3://{BUCKET_NAME}/{MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()