import pandas as pd
import boto3
import io
import os
import re
import sys
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from utils.exception import CustomException
from utils.logger import logger


class DataTransformer:
    def __init__(self):
        load_dotenv()
        self.aws_s3_bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
        self.s3_region = "eu-north-1"
        self.raw_data_folder = "unstructured_data/"
        
        self.s3_client = boto3.client("s3", region_name=self.s3_region)
        self.emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # Emoticons
            u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # Transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # Flags
            "]+", flags=re.UNICODE
        )
        
        # English text pattern - includes common punctuation
        self.english_text_pattern = re.compile(r'^(?:[a-zA-Z\s\.,?!\'"():;-]|<NUM>)+$')

    def run_extraction_if_needed(self, force_extract=False):
        try:
            # Check if unstructured data already exists
            existing_files = self.fetch_all_csv_from_s3()
            
            if existing_files and not force_extract:
                logger.info(f"Found {len(existing_files)} existing files. Skipping extraction.")
                return existing_files
            
            logger.info("Running data extraction...")
            
            # Import and run extraction
            try:
                from etl.extract import run_extraction
            except ImportError:
                from extract import run_extraction
            
            # Run extraction to get fresh data
            extraction_results = run_extraction()
            
            logger.info("Data extraction completed successfully")
            return self.fetch_all_csv_from_s3()  # Get updated file list
            
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            raise CustomException(e, sys)

    def fetch_all_csv_from_s3(self):
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.aws_s3_bucket_name, 
                Prefix=self.raw_data_folder
            )
            csv_files = [
                item['Key'] for item in response.get('Contents', []) 
                if item['Key'].endswith(".csv")
            ]
            logger.info(f"Found {len(csv_files)} CSV files in S3 for transformation.")
            return csv_files
        except Exception as e:
            logger.error("Failed to list CSV files from S3.")
            raise CustomException(e, sys)

    def read_and_combine_csvs(self):
        try:
            csv_files = self.fetch_all_csv_from_s3()
            
            if not csv_files:
                logger.warning("No CSV files found. Running extraction...")
                self.run_extraction_if_needed(force_extract=True)
                csv_files = self.fetch_all_csv_from_s3()
            
            dfs = []
            
            for file in csv_files:
                obj = self.s3_client.get_object(Bucket=self.aws_s3_bucket_name, Key=file)
                df = pd.read_csv(io.BytesIO(obj['Body'].read()))
                dfs.append(df)
                logger.info(f"Loaded file: {file} with {len(df)} records")
            
            combined = pd.concat(dfs, ignore_index=True)
            logger.info(f"Combined dataset: {len(combined)} total records")
            return combined
            
        except Exception as e:
            logger.error("Failed to read or combine CSV files.")
            raise CustomException(e, sys)

    def is_english_no_other_lang(self, text):
        return bool(self.english_text_pattern.match(text))

    def clean_text_content(self, df):
        try:
            # Create explicit copy to avoid SettingWithCopyWarning
            df = df.copy()
            original_count = len(df)
            
            # Lowercase transformation
            df.loc[:, "content"] = df["content"].str.lower()
            logger.info("Applied lowercase transformation")
            
            # Remove emojis
            df.loc[:, "content"] = df["content"].apply(lambda x: self.emoji_pattern.sub("", x))
            logger.info("Removed emojis from content")
            
            # Replace numbers with <NUM> token
            df.loc[:, "content"] = df["content"].apply(lambda x: re.sub(r'\d+', '<NUM>', x))
            logger.info("Replaced numbers with <NUM> token")
            
            # Keep only English text + punctuation + <NUM>
            english_mask = df["content"].apply(self.is_english_no_other_lang)
            df = df[english_mask].copy()  # Create new copy after filtering
            english_count = len(df)
            logger.info(f"Filtered to English-only content: {english_count}/{original_count} retained")
            
            # Normalize whitespace
            df.loc[:, "content"] = df["content"].apply(lambda x: re.sub(r'\s+', ' ', x).strip())
            
            # Remove empty strings after cleaning
            non_empty_mask = df["content"].str.strip() != ''
            df = df[non_empty_mask].copy()  # Create new copy after filtering
            non_empty_count = len(df)
            logger.info(f"Removed empty content: {non_empty_count}/{english_count} retained")
            
            return df
            
        except Exception as e:
            logger.error("Text cleaning failed.")
            raise CustomException(e, sys)

    def filter_and_label_data(self, df):
        try:
            # Create explicit copy to avoid SettingWithCopyWarning
            df = df.copy()
            
            # Remove duplicates based on content
            original_count = len(df)
            df = df.drop_duplicates(subset=["content"]).copy()
            dedup_count = len(df)
            logger.info(f"Removed duplicates: {dedup_count}/{original_count} retained")
            
            # Keep only clear positive (4,5) and negative (1,2) scores
            score_mask = df["score"].isin([1, 2, 4, 5])
            df = df[score_mask].copy()  # Create new copy after filtering
            filtered_count = len(df)
            logger.info(f"Filtered to clear sentiment scores: {filtered_count}/{dedup_count} retained")
            
            # Create binary sentiment labels using .loc for safe assignment
            df.loc[:, "rating_label"] = df["score"].map({
                1: "negative", 
                2: "negative", 
                4: "positive", 
                5: "positive"
            })
            
            # Log class distribution
            class_counts = df["rating_label"].value_counts()
            logger.info(f"Class distribution: {class_counts.to_dict()}")
            
            positive_count = class_counts.get('positive', 0)
            if len(df) > 0:
                positive_ratio = positive_count / len(df)
                logger.info(f"Positive ratio: {positive_ratio:.2%}")
            
            return df
            
        except Exception as e:
            logger.error("Score filtering and labeling failed.")
            raise CustomException(e, sys)

    def create_train_test_split(self, df):
        try:
            # Select final columns for model training
            df_final = df[["content", "rating_label"]].copy()
            
            # Validate we have both classes for stratification
            unique_labels = df_final["rating_label"].nunique()
            if unique_labels < 2:
                logger.warning("Only one class found. Using regular split instead of stratified.")
                train, test = train_test_split(
                    df_final, 
                    test_size=0.2, 
                    random_state=42
                )
            else:
                # Stratified split to maintain class balance
                train, test = train_test_split(
                    df_final, 
                    test_size=0.2, 
                    random_state=42, 
                    stratify=df_final["rating_label"]
                )
            
            logger.info(f"Train set: {len(train)} samples")
            logger.info(f"Test set: {len(test)} samples")
            
            # Log class distribution in splits
            if len(train) > 0:
                train_dist = train["rating_label"].value_counts(normalize=True)
                logger.info(f"Train distribution: {train_dist.to_dict()}")
            
            if len(test) > 0:
                test_dist = test["rating_label"].value_counts(normalize=True)
                logger.info(f"Test distribution: {test_dist.to_dict()}")
            
            return train, test
            
        except Exception as e:
            logger.error("Train/test split failed.")
            raise CustomException(e, sys)

    def transform_data(self, force_extract=False):
        try:

            # Step 0: Run extraction if needed
            self.run_extraction_if_needed(force_extract=force_extract)
            
            # Step 1: Load and combine raw data
            df = self.read_and_combine_csvs()
            
            # Step 2: Select relevant columns and remove nulls
            df = df[["content", "score"]].copy()
            df = df.dropna(subset=["content", "score"]).copy()
            logger.info(f"Selected columns and removed nulls: {len(df)} records remaining")
            
            # Validate we have data to work with
            if len(df) == 0:
                logger.warning("No data remaining after null removal")
                raise ValueError("Dataset became empty after removing nulls")
            
            # Step 3: Clean text content
            df = self.clean_text_content(df)
            
            # Step 4: Filter scores and create labels
            df = self.filter_and_label_data(df)
            
            # Validate we have data after filtering
            if len(df) == 0:
                logger.warning("No data remaining after filtering")
                raise ValueError("Dataset became empty after filtering")
            
            # Step 5: Create train/test split
            train, test = self.create_train_test_split(df)
            
            logger.info("Complete data transformation pipeline completed successfully")
            return train, test
            
        except Exception as e:
            logger.error("Data transformation pipeline failed.")
            raise CustomException(e, sys)


def run_transformation(force_extract=False):
    logger.info("Starting data transformation pipeline...")
    transformer = DataTransformer()
    train_data, test_data = transformer.transform_data(force_extract=force_extract)
    logger.info("Data transformation pipeline completed successfully")
    return train_data, test_data

if __name__ == "__main__":
    run_transformation(force_extract=True)