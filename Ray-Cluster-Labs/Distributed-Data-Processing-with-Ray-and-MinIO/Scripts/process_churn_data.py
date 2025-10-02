import ray
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from io import BytesIO
import urllib.request
import boto3
from botocore.client import Config

# MinIO Configuration
MINIO_ENDPOINT = "http://<minio_private_ip>:9000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "minioadmin123"
BUCKET_NAME = "ray-data"
OUTPUT_PATH = "processed-churn-data"

# Initialize Ray
ray.init()
print("Ray cluster initialized")

# Configure S3 client for MinIO
s3_client = boto3.client(
    's3',
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

# Download Telco Customer Churn dataset
print("Downloading dataset...")
churn_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
urllib.request.urlretrieve(churn_url, "/tmp/churn.csv")

# Load data into Ray Dataset
ds = ray.data.read_csv("/tmp/churn.csv")
print(f"Loaded {ds.count()} rows")

# Data Processing Function
def clean_data(batch: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for churn prediction"""
    # Clean TotalCharges column
    batch['TotalCharges'] = pd.to_numeric(batch['TotalCharges'], errors='coerce')
    batch = batch.fillna(0)
    
    # Keep Churn column as-is for target variable
    # Don't create derived features from TotalCharges (prevents data leakage)
    
    return batch

# Process data - keep ALL customers (churned and not churned)
print("Processing data...")
cleaned_ds = ds.map_batches(clean_data, batch_format="pandas")

# Convert to Pandas DataFrame
processed_data = cleaned_ds.to_pandas()
print(f"Processed {len(processed_data)} total customers")

# Calculate churn statistics
churn_count = (processed_data['Churn'] == 'Yes').sum()
churn_rate = (churn_count / len(processed_data)) * 100
print(f"Churned customers: {churn_count}")
print(f"Retained customers: {len(processed_data) - churn_count}")
print(f"Churn rate: {churn_rate:.2f}%")

# Convert to Parquet format
table = pa.Table.from_pandas(processed_data)
parquet_buffer = BytesIO()
pq.write_table(table, parquet_buffer)
parquet_buffer.seek(0)

# Upload to MinIO
print("Uploading to MinIO...")
try:
    s3_client.put_object(
        Bucket=BUCKET_NAME,
        Key=f"{OUTPUT_PATH}/churned_customers.parquet",
        Body=parquet_buffer.getvalue()
    )
    print(f"✓ Data uploaded to s3://{BUCKET_NAME}/{OUTPUT_PATH}/churned_customers.parquet")
    print(f"✓ Dataset ready for churn prediction training")
except Exception as e:
    print(f"✗ Upload failed: {e}")