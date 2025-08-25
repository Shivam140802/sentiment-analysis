import os
import sys
import time
import joblib
import psycopg2
import pandas as pd
import numpy as np
import sqlalchemy
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from utils.logger import logger
from utils.exception import CustomException

load_dotenv()

class ReviewPredictorPipeline:
    """Fetch reviews from `reviews`, predict, and save to `reviews_prediction`."""

    def __init__(self):
        try:
            # Database configuration
            self.host = os.getenv("LOCAL_PG_HOST")
            self.port = int(os.getenv("LOCAL_PG_PORT", 5432))
            self.db = os.getenv("LOCAL_PG_DB")
            self.user = os.getenv("LOCAL_PG_USER")
            self.password = os.getenv("LOCAL_PG_PASS")
            self.pred_table = os.getenv("LOCAL_PG_PRED_TABLE", "reviews_prediction")
            self.reviews_table = os.getenv("LOCAL_PG_REVIEWS_TABLE", "reviews")
            
            # Load model artifacts
            self.model_path = "artifacts/model/svm_pipeline.pkl"
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
                
            saved = joblib.load(self.model_path)
            self.model = saved['model']
            self.scaler = saved['scaler']
            self.pca = saved['pca']
            
            # Initialize embedding model
            self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("ReviewPredictorPipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize predictor: {e}")
            raise CustomException(e, sys)

    def get_connection(self):
        """Get database connection with retry logic"""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                conn = psycopg2.connect(
                    host=self.host, 
                    port=self.port, 
                    dbname=self.db,
                    user=self.user, 
                    password=self.password
                )
                return conn
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"DB connection failed after {max_retries} attempts: {e}")
                    raise CustomException(e, sys)
                logger.warning(f"DB connection attempt {attempt + 1} failed, retrying...")
                time.sleep(retry_delay)

    def create_table_if_not_exists(self):
        """Create prediction table if it doesn't exist"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            # Create prediction table
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.pred_table} (
                    id SERIAL PRIMARY KEY,
                    review_id VARCHAR(128) UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    prediction INTEGER NOT NULL,
                    confidence FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create index on review_id for faster lookups
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.pred_table}_review_id 
                ON {self.pred_table}(review_id);
            """)
            
            # Create index on predicted_at for time-based queries
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.pred_table}_predicted_at 
                ON {self.pred_table}(predicted_at);
            """)
            
            conn.commit()
            logger.info(f"Ensured prediction table '{self.pred_table}' exists with indexes")
            
        except Exception as e:
            logger.error(f"Failed to create prediction table: {e}")
            raise CustomException(e, sys)
        finally:
            if conn:
                conn.close()

    def fetch_new_reviews(self, last_id=None, limit=500):
        """Fetch new reviews from the raw `reviews` table - FIXED"""
        try:
            conn = self.get_connection()
            
            if last_id is None:
                last_id = 0
            
            # Convert numpy int64 to Python int
            if isinstance(last_id, np.integer):
                last_id = int(last_id)
            if isinstance(limit, np.integer):
                limit = int(limit)
                
            query = f"""
                SELECT r.id, r.review_id, r.content, r.created_at
                FROM {self.reviews_table} r
                LEFT JOIN {self.pred_table} p ON r.review_id = p.review_id
                WHERE r.id > %s AND p.review_id IS NULL
                ORDER BY r.id ASC
                LIMIT %s;
            """
            
            # Use SQLAlchemy-style connection string to avoid pandas warning
            engine = sqlalchemy.create_engine(
                f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"
            )
            
            df = pd.read_sql(query, engine, params=(last_id, limit))
            logger.info(f"Fetched {len(df)} new reviews for prediction")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch new reviews: {e}")
            raise CustomException(e, sys)
        finally:
            if 'engine' in locals():
                engine.dispose()

    def embed_reviews(self, df):
        """Generate embeddings for review content"""
        try:
            if df.empty:
                return []
                
            logger.info(f"Generating embeddings for {len(df)} reviews")
            embeddings = self.embed_model.encode(
                df['content'].tolist(), 
                show_progress_bar=False,
                batch_size=32
            )
            logger.info("Embedding generation completed")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise CustomException(e, sys)

    def predict_reviews(self, df, embeddings):
        """Predict sentiments for reviews"""
        try:
            if df.empty or len(embeddings) == 0:
                return df
                
            # Transform embeddings
            scaled = self.scaler.transform(embeddings)
            reduced = self.pca.transform(scaled)
            
            # Get predictions and probabilities
            predictions = self.model.predict(reduced)
            
            # Get confidence scores (probability of predicted class)
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(reduced)
                confidence_scores = probabilities.max(axis=1)
            else:
                # For SVM, use decision function
                decision_scores = self.model.decision_function(reduced)
                confidence_scores = abs(decision_scores) / abs(decision_scores).max()
            
            df = df.copy()
            df['prediction'] = predictions
            df['confidence'] = confidence_scores
            
            logger.info(f"Predicted sentiments for {len(df)} reviews")
            return df
            
        except Exception as e:
            logger.error(f"Failed to predict reviews: {e}")
            raise CustomException(e, sys)

    def save_predictions(self, df):
        """Insert predictions into reviews_prediction table"""
        if df.empty:
            logger.info("No predictions to save")
            return
            
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            # Prepare batch insert
            insert_query = f"""
                INSERT INTO {self.pred_table} 
                (review_id, content, prediction, confidence, predicted_at)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (review_id) DO UPDATE SET
                    prediction = EXCLUDED.prediction,
                    confidence = EXCLUDED.confidence,
                    predicted_at = EXCLUDED.predicted_at;
            """
            
            # Batch insert
            current_time = datetime.now()
            batch_data = [
                (
                    row['review_id'], 
                    row['content'], 
                    int(row['prediction']), 
                    float(row['confidence']),
                    current_time
                )
                for _, row in df.iterrows()
            ]
            
            cur.executemany(insert_query, batch_data)
            conn.commit()
            
            logger.info(f"Saved {len(df)} predictions to database")
            
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
            raise CustomException(e, sys)
        finally:
            if conn:
                conn.close()

    def get_prediction_stats(self):
        """Get statistics about predictions"""
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            
            cur.execute(f"""
                SELECT 
                    COUNT(*) as total_predictions,
                    COUNT(CASE WHEN prediction = 1 THEN 1 END) as positive_count,
                    COUNT(CASE WHEN prediction = 0 THEN 1 END) as negative_count,
                    AVG(confidence) as avg_confidence,
                    MAX(predicted_at) as last_prediction
                FROM {self.pred_table};
            """)
            
            stats = cur.fetchone()
            return {
                'total_predictions': stats[0],
                'positive_count': stats[1],
                'negative_count': stats[2],
                'avg_confidence': float(stats[3]) if stats[3] else 0.0,
                'last_prediction': stats[4]
            }
            
        except Exception as e:
            logger.error(f"Failed to get prediction stats: {e}")
            return {}
        finally:
            if conn:
                conn.close()
