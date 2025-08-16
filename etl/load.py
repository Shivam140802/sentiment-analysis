# Modified load.py
import pandas as pd
import boto3
import io
import os
import sys
from dotenv import load_dotenv
from datetime import datetime
from utils.exception import CustomException
from utils.logger import logger

class DataLoader:
    def __init__(self):
        load_dotenv()
        self.aws_s3_bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
        self.s3_region = "eu-north-1"
        self.structured_data_folder = "structured_data/"
        
        self.s3_client = boto3.client("s3", region_name=self.s3_region)

    def upload_to_s3(self, df, dataset_name):
        try:
            # Keep only content and rating_label columns
            clean_df = df[["content", "rating_label"]].copy()
            
            # Consistent filename for pipeline
            key = f"{self.structured_data_folder}{dataset_name}.csv"
            
            # Convert DataFrame to CSV buffer
            csv_buffer = io.StringIO()
            clean_df.to_csv(csv_buffer, index=False)
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.aws_s3_bucket_name, 
                Key=key, 
                Body=csv_buffer.getvalue()
            )
            
            logger.info(f"Successfully uploaded: {key} with {len(clean_df)} records")
            return key
            
        except Exception as e:
            logger.error(f"Failed to upload {dataset_name} to S3: {e}")
            raise CustomException(e, sys)

    def load_processed_data(self, train_df, test_df):
        
        try:
            # Validate input data
            if len(train_df) == 0 or len(test_df) == 0:
                raise ValueError("Cannot load empty datasets")
            
            # Upload clean datasets with only essential columns
            train_file = self.upload_to_s3(train_df, "train")
            test_file = self.upload_to_s3(test_df, "test")
            
            # Simple summary logging
            logger.info(f"Data loading completed:")
            logger.info(f"- Train dataset: {len(train_df)} samples -> {train_file}")
            logger.info(f"- Test dataset: {len(test_df)} samples -> {test_file}")
            
            return {
                "train_file": train_file,
                "test_file": test_file
            }
            
        except Exception as e:
            logger.error("Data loading process failed.")
            raise CustomException(e, sys)

    def transform_and_load(self):
        try:
            from etl.transform import run_transformation  
            logger.info("Starting transform and load pipeline...")
            train_data, test_data = run_transformation()
            result = self.load_processed_data(train_data, test_data)
            return result
            
        except Exception as e:
            logger.error("Transform and load pipeline failed.")
            raise CustomException(e, sys)

    def list_structured_datasets(self):
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.aws_s3_bucket_name, 
                Prefix=self.structured_data_folder
            )
            
            datasets = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('.csv'):
                        datasets.append({
                            'file': obj['Key'],
                            'size': obj['Size'],
                            'modified': obj['LastModified']
                        })
            
            logger.info(f"Found {len(datasets)} structured datasets in S3")
            return datasets
            
        except Exception as e:
            logger.error(f"Failed to list structured datasets: {e}")
            raise CustomException(e, sys)

def run_loading(train_data, test_data):
    logger.info("Loading started")
    loader = DataLoader()
    result = loader.load_processed_data(train_data, test_data)
    return result

def run_complete_pipeline():
    loader = DataLoader()
    result = loader.transform_and_load()
    return result

if __name__ == "__main__":
    # Now load.py can run the complete pipeline
    result = run_complete_pipeline()
    logger.info("Pipeline completed successfully!")
