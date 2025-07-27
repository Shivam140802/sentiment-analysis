import pandas as pd
import boto3
import io
import os
import re
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
S3_REGION = "eu-north-1"
RAW_DATA_FOLDER = "unstructured_data/"
CLEANED_DATA_FOLDER = "structured_data/"

s3 = boto3.client("s3", region_name=S3_REGION)

def is_english(text):
    return bool(re.match("^[\x00-\x7F]+$", text))  

def fetch_all_csv_from_s3():
    response = s3.list_objects_v2(Bucket=AWS_S3_BUCKET_NAME, Prefix=RAW_DATA_FOLDER)
    csv_files = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith(".csv")]
    return csv_files

def read_and_combine_csvs():
    csv_files = fetch_all_csv_from_s3()
    dfs = []
    for file in csv_files:
        obj = s3.get_object(Bucket=AWS_S3_BUCKET_NAME, Key=file)
        df = pd.read_csv(io.BytesIO(obj['Body'].read()))
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    return combined

def clean_and_split_data(df):
    df = df[["content", "score"]]
    df.dropna(subset=["content", "score"], inplace=True)
    df = df[df["content"].apply(is_english)]

    df["content"] = df["content"].str.lower()

    df.drop_duplicates(subset=["content"], inplace=True)

    rating_map = {
        1: "negative",
        2: "negative",
        3: "neutral",
        4: "positive",
        5: "positive"
    }
    df["rating_label"] = df["score"].map(rating_map)
    df = df[["content", "rating_label"]]

    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    return train, val, test



def upload_to_s3(df, name):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    key = f"{CLEANED_DATA_FOLDER}{name}.csv"
    s3.put_object(Bucket=AWS_S3_BUCKET_NAME, Key=key, Body=csv_buffer.getvalue())
    print(f"Uploaded: {key}")

def clean_all():
    print("Fetching and cleaning data...")
    df = read_and_combine_csvs()
    train, val, test = clean_and_split_data(df)
    upload_to_s3(train, "train")
    upload_to_s3(val, "val")
    upload_to_s3(test, "test")
    print("Cleaning complete.")

if __name__ == "__main__":
    clean_all()
