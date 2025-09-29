import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.xgboost import XGBoostTrainer
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import boto3
from botocore.client import Config
from io import BytesIO
import pickle
import urllib.request

# MinIO Configuration
MINIO_ENDPOINT = "http://<minio_private_ip>:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin123"
BUCKET_NAME = "ray-data"
MODEL_OUTPUT_PATH = "models/churn_xgboost_model.pkl"

# Processed data URL from Distributed Data Processing with Ray lab
DATA_URL = "https://github.com/poridhioss/MLOps/raw/refs/heads/main/Ray-Cluster-Labs/Distributed%20XGBoost%20Training%20with%20Ray%20Train/processed-data/churned_customers.parquet"

# Initialize Ray
ray.init()

# Configure S3 client for MinIO
s3_client = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

def load_data():
    """Download processed data from GitHub"""
    urllib.request.urlretrieve(DATA_URL, "/tmp/churned_customers.parquet")
    df = pd.read_parquet("/tmp/churned_customers.parquet")
    return df

def prepare_features(df):
    """Prepare features for XGBoost training"""
    # Select numerical features
    feature_columns = [
        'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
        'AvgMonthlyCharges', 'IsHighValue'
    ]
    
    # Encode categorical variables
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
    
    # Create binary target variable
    df_encoded['target'] = (df_encoded['TotalCharges'] > df_encoded['TotalCharges'].median()).astype(int)
    
    X = df_encoded[feature_columns]
    y = df_encoded['target']
    
    return X, y, label_encoders

def train_func(config):
    """
    Training function executed on each Ray worker
    
    This function runs in parallel across multiple Ray workers.
    Ray Train automatically handles:
    - Data distribution: Each worker receives a unique subset of data
    - Gradient synchronization: Workers share updates to maintain consistent model
    - Checkpoint coordination: Model states are saved periodically
    """
    
    # Get data partition assigned to this worker
    train_dataset = train.get_dataset_shard("train")
    valid_dataset = train.get_dataset_shard("valid")
    
    # Convert Ray datasets to pandas for XGBoost
    train_df = train_dataset.to_pandas()
    valid_df = valid_dataset.to_pandas()
    
    # Prepare XGBoost datasets
    train_X = train_df.drop('target', axis=1)
    train_y = train_df['target']
    valid_X = valid_df.drop('target', axis=1)
    valid_y = valid_df['target']
    
    dtrain = xgb.DMatrix(train_X, label=train_y)
    dvalid = xgb.DMatrix(valid_X, label=valid_y)
    
    # XGBoost training parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': config.get('max_depth', 6),
        'learning_rate': config.get('learning_rate', 0.1),
        'subsample': config.get('subsample', 0.8),
        'colsample_bytree': config.get('colsample_bytree', 0.8),
        'random_state': 42
    }
    
    # Train XGBoost model
    # Ray Train coordinates parallel training across all workers
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dvalid, 'valid')],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    # Evaluate model performance
    valid_predictions = model.predict(dvalid)
    valid_pred_binary = (valid_predictions > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(valid_y, valid_pred_binary),
        'precision': precision_score(valid_y, valid_pred_binary, average='weighted'),
        'recall': recall_score(valid_y, valid_pred_binary, average='weighted'),
        'f1': f1_score(valid_y, valid_pred_binary, average='weighted')
    }
    
    # Report metrics back to Ray Train coordinator
    train.report(metrics)
    
    return model

def main():
    # Load and prepare data
    df = load_data()
    X, y, label_encoders = prepare_features(df)
    
    # Combine features and target
    data_df = X.copy()
    data_df['target'] = y
    
    # Split into train and validation sets
    train_df, valid_df = train_test_split(data_df, test_size=0.2, random_state=42, stratify=y)
    
    # Create Ray datasets for distributed training
    train_dataset = ray.data.from_pandas(train_df)
    valid_dataset = ray.data.from_pandas(valid_df)
    
    # Configure distributed training
    scaling_config = ScalingConfig(
        num_workers=2,
        use_gpu=False,
        resources_per_worker={"CPU": 1}
    )
    
    # Create XGBoost trainer for distributed training
    trainer = XGBoostTrainer(
        train_loop_per_worker=train_func,
        scaling_config=scaling_config,
        datasets={"train": train_dataset, "valid": valid_dataset},
        train_loop_config={
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
    )
    
    # Execute distributed training
    result = trainer.fit()
    
    # Serialize and upload model to MinIO
    model_data = {'model': result.checkpoint, 'label_encoders': label_encoders}
    model_buffer = BytesIO()
    pickle.dump(model_data, model_buffer)
    model_buffer.seek(0)
    
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=MODEL_OUTPUT_PATH,
        Body=model_buffer.getvalue()
    )
    
    print(f"Training completed successfully")
    print(f"Final metrics: {result.metrics}")
    print(f"Model saved to: s3://{BUCKET_NAME}/{MODEL_OUTPUT_PATH}")

if __name__ == "__main__":
    main()