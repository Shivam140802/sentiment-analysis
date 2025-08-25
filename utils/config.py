import os
import io
import boto3
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor
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

load_dotenv()

class Database:
    def __init__(self):
        # Load configuration from environment
        self.host = os.getenv("LOCAL_PG_HOST", "localhost")
        self.port = int(os.getenv("LOCAL_PG_PORT", 5432))
        self.db = os.getenv("LOCAL_PG_DB", "reviews_db")
        self.user = os.getenv("LOCAL_PG_USER")
        self.password = os.getenv("LOCAL_PG_PASS")
        self.table = os.getenv("LOCAL_PG_TABLE", "reviews")

    def get_connection(self):
        """Return a new PostgreSQL connection with dict-like cursor."""
        return psycopg2.connect(
            host=self.host,
            port=self.port,
            dbname=self.db,
            user=self.user,
            password=self.password,
            cursor_factory=RealDictCursor
        )
