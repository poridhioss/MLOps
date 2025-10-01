import ray
from ray.train import ScalingConfig, RunConfig
from ray.train.xgboost import XGBoostTrainer
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import boto3
from botocore.client import Config
from io import BytesIO
import pickle
import urllib.request
import os
from pyarrow import fs
import mlflow
import mlflow.xgboost

# MinIO Configuration from environment variables
MINIO_ENDPOINT = os.getenv("AWS_ENDPOINT_URL", "http://10.0.1.199:9000")
MINIO_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
MINIO_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
BUCKET_NAME = "ray-data"
MODEL_OUTPUT_PATH = "models/churn_xgboost_model_mlflow.pkl"

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://10.0.1.200:5000")

# CRITICAL: Set these environment variables for MLflow's internal S3 client
os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Processed data URL
DATA_URL = "https://github.com/poridhioss/MLOps/raw/refs/heads/main/Ray-Cluster-Labs/Distributed%20XGBoost%20Training%20with%20Ray%20Train/processed-data/churned-customers.parquet"

def load_data():
    """Download and prepare data"""
    print("Downloading processed customer data...")
    urllib.request.urlretrieve(DATA_URL, "/tmp/churned_customers.parquet")
    df = pd.read_parquet("/tmp/churned_customers.parquet")
    print(f"Loaded {len(df)} customer records")
    return df

def prepare_features(df):
    """Prepare features for churn prediction"""
    print("Preparing features for churn prediction...")
    
    # Use behavioral and demographic features only
    # EXCLUDE TotalCharges and derived features to prevent data leakage
    feature_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges']
    
    categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService',
                          'MultipleLines', 'InternetService', 'OnlineSecurity',
                          'OnlineBackup', 'DeviceProtection', 'TechSupport',
                          'StreamingTV', 'StreamingMovies', 'Contract',
                          'PaperlessBilling', 'PaymentMethod']
    
    df_encoded = df.copy()
    label_encoders = {}
    
    # Encode categorical variables
    for col in categorical_columns:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            label_encoders[col] = le
            feature_columns.append(col)
    
    # Use actual Churn column as target
    if 'Churn' in df.columns:
        if df['Churn'].dtype == 'object':
            df_encoded['target'] = df['Churn'].map({'Yes': 1, 'No': 0})
        else:
            df_encoded['target'] = df['Churn'].astype(int)
        print("✓ Using actual Churn column as target")
    else:
        raise ValueError("Churn column not found in dataset. Please use the updated preprocessing script.")
    
    print(f"✓ Prepared {len(feature_columns)} features")
    print(f"✓ Target distribution: {df_encoded['target'].value_counts().to_dict()}")
    
    churn_rate = df_encoded['target'].mean()
    print(f"✓ Churn rate: {churn_rate:.2%}")
    
    return df_encoded[feature_columns + ['target']], label_encoders

def evaluate_model(booster, X_test, y_test):
    """Evaluate model and return comprehensive metrics"""
    print("Evaluating model performance...")
    
    # Create DMatrix for prediction
    dtest = xgb.DMatrix(X_test)
    
    # Get predictions
    y_pred_proba = booster.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    # Calculate confusion matrix components
    cm = confusion_matrix(y_test, y_pred)
    metrics['true_negatives'] = int(cm[0, 0])
    metrics['false_positives'] = int(cm[0, 1])
    metrics['false_negatives'] = int(cm[1, 0])
    metrics['true_positives'] = int(cm[1, 1])
    
    return metrics

def main():
    print("\n" + "="*60)
    print("Starting Distributed XGBoost Training with MLflow Tracking")
    print("Churn Prediction Model")
    print("="*60 + "\n")
    
    # Initialize Ray
    ray.init()
    print("✓ Ray cluster initialized")
    
    # Configure S3 client for MinIO (boto3)
    s3_client = boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )
    print("✓ MinIO S3 client configured")
    
    # Configure PyArrow S3 filesystem for Ray Train
    endpoint_host = MINIO_ENDPOINT.replace("http://", "").replace("https://", "")
    s3_fs = fs.S3FileSystem(
        endpoint_override=endpoint_host,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        scheme="http",
        region="us-east-1"
    )
    print("✓ PyArrow S3 filesystem configured")
    
    # Load and prepare data
    df = load_data()
    data_df, label_encoders = prepare_features(df)
    
    # Split data into train, validation, and test sets
    train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42)
    train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42)
    
    print(f"✓ Data split: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")
    
    # Start MLflow experiment
    mlflow.set_experiment("xgboost-churn-prediction")
    print(f"✓ MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    
    with mlflow.start_run(run_name="distributed_xgboost_training"):
        print("\n" + "-"*60)
        print("MLflow Run Started - Logging Parameters and Training...")
        print("-"*60 + "\n")
        
        # Define and log parameters
        params = {
            "objective": "binary:logistic",
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "num_workers": 2,
            "num_boost_round": 100,
            "test_size": 0.2,
            "validation_size": 0.2,
            "random_state": 42
        }
        mlflow.log_params(params)
        print("✓ Parameters logged to MLflow")
        
        # Create Ray datasets for distributed training
        train_dataset = ray.data.from_pandas(train_df)
        valid_dataset = ray.data.from_pandas(valid_df)
        print("✓ Ray datasets created")
        
        # Configure distributed training with 2 workers
        print("\nStarting distributed XGBoost training...")
        trainer = XGBoostTrainer(
            scaling_config=ScalingConfig(
                num_workers=2,
                use_gpu=False,
                resources_per_worker={"CPU": 1}
            ),
            label_column="target",
            params={
                "objective": params["objective"],
                "eval_metric": ["logloss", "error"],
                "max_depth": params["max_depth"],
                "learning_rate": params["learning_rate"],
                "subsample": params["subsample"],
                "colsample_bytree": params["colsample_bytree"],
            },
            datasets={"train": train_dataset, "valid": valid_dataset},
            num_boost_round=params["num_boost_round"],
            run_config=RunConfig(
                storage_path="ray-data/checkpoints",
                storage_filesystem=s3_fs
            ),
        )
        
        # Execute distributed training
        result = trainer.fit()
        print("✓ Distributed training completed")
        
        # Get trained model using Ray's built-in method
        booster = XGBoostTrainer.get_model(result.checkpoint)
        print("✓ Model extracted from checkpoint")
        
        # Evaluate model on test set
        X_test = test_df.drop('target', axis=1)
        y_test = test_df['target']
        
        metrics = evaluate_model(booster, X_test, y_test)
        
        # Log metrics to MLflow
        mlflow.log_metrics(metrics)
        print("✓ Metrics logged to MLflow")
        
        # Log model to MLflow with registered model name
        print("Logging model to MLflow...")
        mlflow.xgboost.log_model(
            booster, 
            "model",
            registered_model_name="xgboost-churn-model"
        )
        print("✓ Model logged to MLflow artifact store")
        
        # Save model to MinIO for backup
        model_data = {'model': booster, 'label_encoders': label_encoders}
        model_buffer = BytesIO()
        pickle.dump(model_data, model_buffer)
        model_buffer.seek(0)
        
        s3_client.put_object(
            Bucket=BUCKET_NAME,
            Key=MODEL_OUTPUT_PATH,
            Body=model_buffer.getvalue()
        )
        print("✓ Model saved to MinIO")
        
        # Log MinIO artifact location to MLflow
        mlflow.log_param("minio_model_path", f"s3://{BUCKET_NAME}/{MODEL_OUTPUT_PATH}")
        
        # Log dataset information
        mlflow.log_param("total_samples", len(data_df))
        mlflow.log_param("train_samples", len(train_df))
        mlflow.log_param("valid_samples", len(valid_df))
        mlflow.log_param("test_samples", len(test_df))
        
        # Print comprehensive results
        print("\n" + "="*60)
        print("Training Completed Successfully!")
        print("="*60)
        print(f"\nModel Performance Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {metrics['true_negatives']:4d}  FP: {metrics['false_positives']:4d}")
        print(f"  FN: {metrics['false_negatives']:4d}  TP: {metrics['true_positives']:4d}")
        print(f"\nStorage Locations:")
        print(f"  MinIO:  s3://{BUCKET_NAME}/{MODEL_OUTPUT_PATH}")
        print(f"  MLflow: {MLFLOW_TRACKING_URI}")
        print(f"\nExperiment: xgboost-churn-prediction")
        print(f"Run Name:   distributed_xgboost_training")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()