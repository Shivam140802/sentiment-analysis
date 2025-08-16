import os
import io
import boto3
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
S3_REGION = "eu-north-1"
CLEANED_DATA_FOLDER = "structured_data/"

s3 = boto3.client("s3", region_name=S3_REGION)

def load_csv_from_s3(file_name: str) -> pd.DataFrame:
    key = f"{CLEANED_DATA_FOLDER}{file_name}.csv"
    obj = s3.get_object(Bucket=AWS_S3_BUCKET_NAME, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))