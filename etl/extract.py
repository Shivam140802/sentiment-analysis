import os
import time
import boto3
import pandas as pd
import io
import re
from dotenv import load_dotenv
from google_play_scraper import reviews, Sort
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import sys

from utils.exception import CustomException
from utils.logger import logger

class ReviewExtractor:
    def __init__(self):
        load_dotenv()
        self.s3_client = boto3.client('s3', region_name='eu-north-1')
        self.bucket_name = os.getenv("AWS_S3_BUCKET_NAME")
        self.folder = 'unstructured_data/'
        
        self.apps = [
            'com.snapchat.android', 'com.linkedin.android','com.whatsapp', 
            'com.facebook.katana', 'com.instagram.android','com.twitter.android',
            'org.telegram.messenger', 'com.reddit.frontpage'
        ]


    def has_spaced_pattern(self, text):
        spaced_pattern = r'\b[a-zA-Z]\s+[a-zA-Z]\s'
        matches = re.findall(spaced_pattern, text)
        
        # Flag if found spaced patterns
        return len(matches) > 0

    def is_quality_content(self, text):
        if not text or len(text.strip()) < 10:
            return False
        
        if self.has_spaced_pattern(text):
            return False
        
        alpha_chars = sum(1 for c in text if c.isalpha())
        total_chars = len(text.replace(' ', ''))
        if total_chars > 0 and alpha_chars / total_chars < 0.5:
            return False
        
        if re.search(r'(.)\1{4,}', text):
            return False
        
        return True

    def upload_to_s3(self, app_id, reviews):
        try:
            file_name = f"{self.folder}{app_id.replace('.', '_')}.csv"
            df = pd.DataFrame(reviews)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            
            self.s3_client.put_object(
                Bucket=self.bucket_name, 
                Key=file_name, 
                Body=csv_buffer.getvalue()
            )
            
            logger.info(f"[{app_id}] Uploaded complete dataset: {file_name} (total records: {len(reviews)})")
            return file_name
            
        except Exception as e:
            logger.error(f"[{app_id}] Failed to upload to S3: {e}")
            raise CustomException(e, sys)

    def scrape_and_extract_reviews(self, app_id):
        try:
            count = 0
            token = None
            seen_reviews = set()
            all_reviews = []
            
            while count<10000:
                result, token = reviews(
                    app_id, lang='en', country='us', sort=Sort.NEWEST,
                    count=200, continuation_token=token
                )

                if not result:
                    logger.warning(f"[{app_id}] No more results returned from Play Store.")
                    break

                for r in result:
                    user_name = r.get('userName') or 'Unknown'
                    raw_content = r.get('content')
                    score = r.get('score', 0)
                    if not raw_content:
                        continue
                        
                    content = raw_content.strip()
                    
                    if not content:
                        continue

                    # Add word count filter - greater than 5 words
                    word_count = len(content.split())
                    if word_count <= 5:
                        continue

                    # Skip numeric content
                    if content.isnumeric():
                        continue

                    # Quality content filter
                    if not self.is_quality_content(content):
                        continue

                    unique_id = (user_name, content)
                    if unique_id in seen_reviews:
                        continue

                    seen_reviews.add(unique_id)

                    # Create review record
                    review = {
                        "appId": app_id,
                        "userName": user_name,
                        "content": content,
                        "score": score,
                        "extracted_at": datetime.now().isoformat()
                    }

                    all_reviews.append(review)
                    count += 1

                    if count >= 10000:
                        break

                if count % 2500 == 0 and count > 0:
                    logger.info(f"[{app_id}] Progress: {count}/10000 quality reviews extracted")

                time.sleep(1)  # Rate limiting

            # Upload all reviews to S3
            if all_reviews:
                file_name = self.upload_to_s3(app_id, all_reviews)
                logger.info(f"[{app_id}] Extraction completed: {count} quality reviews saved to {file_name}")
                return {"app_id": app_id, "count": count, "file": file_name}
            else:
                logger.warning(f"[{app_id}] No quality reviews extracted for this app")
                return {"app_id": app_id, "count": 0, "file": None}

        except Exception as e:
            logger.error(f"[{app_id}] Error during extraction: {e}")
            raise CustomException(e, sys)

    def extract_all_reviews(self):
        try:
            logger.info("Starting quality-filtered review extraction...")
            results = []

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(self.scrape_and_extract_reviews, app) for app in self.apps]
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(f"Completed extraction for {result['app_id']}: {result['count']} reviews")
                    except Exception as e:
                        logger.error(f"Thread execution failed: {e}")
                        continue

            # Summary logging
            total_reviews = sum(r['count'] for r in results)
            successful_apps = len([r for r in results if r['count'] > 0])
            
            logger.info(f"Extraction summary:")
            logger.info(f"- Total apps processed: {len(results)}")
            logger.info(f"- Successful extractions: {successful_apps}")
            logger.info(f"- Total quality reviews extracted: {total_reviews}")
            
            return results

        except Exception as e:
            logger.error("Review extraction pipeline failed.")
            raise CustomException(e, sys)

    def get_extraction_summary(self):
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, 
                Prefix=self.folder
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('.csv'):
                        files.append({
                            'file': obj['Key'],
                            'size': obj['Size'],
                            'modified': obj['LastModified']
                        })
            
            logger.info(f"Found {len(files)} extracted files in S3")
            return files
            
        except Exception as e:
            logger.error(f"Failed to get extraction summary: {e}")
            raise CustomException(e, sys)

def run_extraction():
    extractor = ReviewExtractor()
    results = extractor.extract_all_reviews()
    return results

if __name__ == "__main__":
    run_extraction()
