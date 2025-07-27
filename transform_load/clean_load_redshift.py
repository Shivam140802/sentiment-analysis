import psycopg2
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get values from environment variables or hardcoded values
host = "my-dev-workgroup.227274823571.eu-north-1.redshift-serverless.amazonaws.com"
port = 5439
dbname = "dev"  
user = "admin"  
password = os.getenv("REDSHIFT_PASSWORD")
iam_role = os.getenv("REDSHIFT_IAM_ROLE")  

# Connect to Redshift
conn = psycopg2.connect(
    host=host,
    port=port,
    dbname=dbname,
    user=user,
    password=password
)
cur = conn.cursor()

# 1. Create the table
cur.execute("""
CREATE TABLE IF NOT EXISTS reviews (
    content VARCHAR(65535),
    rating_label VARCHAR(50)
);
""")

conn.commit()
file_names = ["train.csv", "test.csv", "val.csv"]

for file in file_names:
    s3_path = f"s3://data-engineering-project-shivam/structured_data/{file}"
    copy_query = f"""
    COPY reviews
    FROM '{s3_path}'
    IAM_ROLE '{iam_role}'
    FORMAT AS CSV
    IGNOREHEADER 1;
    """
    cur.execute(copy_query)
    print(f"Loaded {file}")

conn.commit()
cur.close()
conn.close()