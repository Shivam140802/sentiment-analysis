import os
import sys
import time
import json
import threading
import re
import signal
import psycopg2
from datetime import datetime, timezone
from kafka import KafkaProducer, KafkaConsumer, OffsetAndMetadata
from kafka.errors import KafkaError, KafkaTimeoutError
from google_play_scraper import reviews, Sort
from dotenv import load_dotenv
from utils.exception import CustomException
from utils.logger import logger

load_dotenv()

class ReviewPipeline:
    """Kafka + PostgreSQL review scraping and storing pipeline for continuous operation."""

    def __init__(self):
        # Kafka config
        self.bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self.topic = os.getenv("KAFKA_TOPIC", "app_reviews")
        self.app_id = os.getenv("APP_ID")
        self.lang = os.getenv("SCRAPE_LANG", "en")
        self.country = os.getenv("SCRAPE_COUNTRY", "us")
        self.page_size = int(os.getenv("SCRAPE_PAGE_SIZE", "100"))
        self.idle_sleep = int(os.getenv("SCRAPE_IDLE_SLEEP_SEC", "300"))  # 5 minutes when no new reviews
        self.page_sleep = float(os.getenv("SCRAPE_PAGE_SLEEP_SEC", "2.0"))
        self.cycle_sleep = int(os.getenv("SCRAPE_CYCLE_SLEEP_SEC", "3600"))  # 1 hour between full cycles

        # Postgres config
        self.host = os.getenv("LOCAL_PG_HOST")
        self.port = int(os.getenv("LOCAL_PG_PORT", "5432"))
        self.db = os.getenv("LOCAL_PG_DB")
        self.user = os.getenv("LOCAL_PG_USER")
        self.password = os.getenv("LOCAL_PG_PASS")

        # Validate required configs
        if not all([self.bootstrap_servers, self.topic, self.app_id, self.host, self.db, self.user, self.password]):
            missing = [k for k, v in {
                'KAFKA_BOOTSTRAP_SERVERS': self.bootstrap_servers,
                'KAFKA_TOPIC': self.topic,
                'APP_ID': self.app_id,
                'LOCAL_PG_HOST': self.host,
                'LOCAL_PG_DB': self.db,
                'LOCAL_PG_USER': self.user,
                'LOCAL_PG_PASS': self.password
            }.items() if not v]
            raise ValueError(f"Missing required environment variables: {missing}")

        # Threading - REMOVED MAX_REVIEWS LIMIT
        self.stop_event = threading.Event()
        self.total_reviews_sent = 0
        self.review_count_lock = threading.Lock()
        self.consecutive_empty_cycles = 0
        
        # Connection pools/caching
        self._db_connection_cache = threading.local()

        # Regex for cleaning
        self.emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002700-\U000027BF"  # dingbats
            u"\U0001f926-\U0001f937"  # additional emojis
            "]+", flags=re.UNICODE
        )
        self.allowed_chars = re.compile(r"^[a-zA-Z0-9\s.,!?;:'\"()@&%+\-\/<>]+$")

        logger.info("ReviewPipeline initialized for continuous operation")

    def get_connection(self):
        """Get database connection with connection pooling and retry logic"""
        if hasattr(self._db_connection_cache, 'conn'):
            try:
                self._db_connection_cache.conn.cursor().execute('SELECT 1')
                return self._db_connection_cache.conn
            except:
                pass

        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                conn = psycopg2.connect(
                    host=self.host, 
                    port=self.port,
                    dbname=self.db, 
                    user=self.user, 
                    password=self.password,
                    connect_timeout=10
                )
                self._db_connection_cache.conn = conn
                logger.debug("Postgres connection established")
                return conn
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Postgres connection failed after {max_retries} attempts: {e}")
                    raise CustomException(e, sys)
                logger.warning(f"DB connection attempt {attempt + 1} failed, retrying in {retry_delay}s...")
                time.sleep(retry_delay)

    def init_local_db(self):
        """Initialize database with proper error handling and indexes"""
        ddl = """
        CREATE TABLE IF NOT EXISTS reviews (
            id SERIAL PRIMARY KEY, 
            review_id VARCHAR(128) UNIQUE NOT NULL,
            user_id VARCHAR(255),
            content TEXT,
            app_id VARCHAR(128),
            scraped_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_reviews_review_id ON reviews(review_id);
        CREATE INDEX IF NOT EXISTS idx_reviews_app_id ON reviews(app_id);
        CREATE INDEX IF NOT EXISTS idx_reviews_scraped_at ON reviews(scraped_at);
        """
        
        conn = None
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()
            logger.info("Initialized reviews table with indexes")
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Failed to initialize reviews table: {e}")
            raise CustomException(e, sys)
        finally:
            if conn:
                conn.close()
                if hasattr(self._db_connection_cache, 'conn'):
                    delattr(self._db_connection_cache, 'conn')

    def insert_batch(self, rows):
        """Insert batch with better error handling and performance"""
        if not rows:
            return 0
            
        conn = None
        inserted_count = 0
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                insert_query = """
                    INSERT INTO reviews (review_id, user_id, content, app_id, scraped_at)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (review_id) DO NOTHING;
                """
                
                batch_data = []
                for r in rows:
                    scraped_at = None
                    if r.get("scraped_at"):
                        try:
                            scraped_at = datetime.fromisoformat(r["scraped_at"].replace('Z', '+00:00'))
                        except:
                            scraped_at = datetime.now(timezone.utc)
                    
                    batch_data.append((
                        r["review_id"], 
                        r.get("user_id"), 
                        r.get("content", ""),
                        r.get("app_id"),
                        scraped_at
                    ))
                
                cur.executemany(insert_query, batch_data)
                inserted_count = cur.rowcount
                
            conn.commit()
            logger.info(f"Inserted batch of {inserted_count}/{len(rows)} reviews")
            return inserted_count
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Batch insert failed: {e}")
            raise CustomException(e, sys)
        finally:
            if conn:
                conn.close()
                if hasattr(self._db_connection_cache, 'conn'):
                    delattr(self._db_connection_cache, 'conn')

    # [Keep all the cleaning methods as they were - is_quality_content, clean_text, is_english_no_other_lang]
    def is_quality_content(self, text: str) -> bool:
        if not text or not isinstance(text, str):
            return False
        words = text.split()
        return (
            len(words) >= 5 and
            not text.isnumeric() and
            len(text.strip()) >= 20 and
            not all(len(word) <= 2 for word in words)
        )

    def clean_text(self, content: str) -> str:
        if not content:
            return ""
        s = content.strip()
        s = self.emoji_pattern.sub("", s)
        s = re.sub(r'\b\d+\b', "<NUM>", s)
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(r'[.]{3,}', '...', s)
        s = re.sub(r'[!]{2,}', '!', s)
        s = re.sub(r'[?]{2,}', '?', s)
        return s.strip().lower()

    def is_english_no_other_lang(self, text: str) -> bool:
        if not text:
            return False
        return bool(self.allowed_chars.fullmatch(text)) and len(text) >= 10

    def build_producer(self):
        try:
            return KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
                key_serializer=lambda k: str(k).encode("utf-8"),
                linger_ms=5,
                acks="all",
                retries=5,
                retry_backoff_ms=100,
                request_timeout_ms=30000,
                batch_size=16384,
                buffer_memory=33554432,
                compression_type='gzip'
            )
        except Exception as e:
            logger.error(f"Failed to create Kafka producer: {e}")
            raise CustomException(e, sys)

    def build_consumer(self, group_id: str = "storer"):
        try:
            return KafkaConsumer(
                self.topic,
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                auto_offset_reset="earliest",
                enable_auto_commit=False,
                group_id=group_id,
                consumer_timeout_ms=1000,
                max_poll_records=50,
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000,
                fetch_min_bytes=1,
                fetch_max_wait_ms=500
            )
        except Exception as e:
            logger.error(f"Failed to create Kafka consumer: {e}")
            raise CustomException(e, sys)

    def produce_reviews(self):
        """CONTINUOUS producer - no limits, runs indefinitely"""
        if not self.app_id:
            raise RuntimeError("APP_ID is required")
            
        producer = None
        try:
            producer = self.build_producer()
            continuation_token = None
            consecutive_errors = 0
            max_consecutive_errors = 5
            cycle_count = 0
            
            logger.info(f"Starting CONTINUOUS scraping for app: {self.app_id}")

            while not self.stop_event.is_set():
                try:
                    cycle_count += 1
                    logger.info(f"Starting scraping cycle #{cycle_count}")
                    
                    # Reset continuation token for new cycle
                    if cycle_count > 1:
                        continuation_token = None
                        logger.info(f"Waiting {self.cycle_sleep}s before next cycle...")
                        if self.stop_event.wait(self.cycle_sleep):
                            break
                    
                    cycle_reviews_sent = 0
                    empty_results_count = 0
                    max_empty_results = 3  # Stop cycle after 3 empty results
                    
                    # Scrape within this cycle
                    while not self.stop_event.is_set() and empty_results_count < max_empty_results:
                        try:
                            result, continuation_token = reviews(
                                self.app_id, 
                                lang=self.lang, 
                                country=self.country,
                                sort=Sort.NEWEST, 
                                count=self.page_size,
                                continuation_token=continuation_token
                            )
                            
                            if not result:
                                empty_results_count += 1
                                logger.info(f"No reviews found (attempt {empty_results_count}/{max_empty_results})")
                                time.sleep(self.idle_sleep)
                                continue

                            sent = 0
                            skipped = 0
                            
                            for r in result:
                                if self.stop_event.is_set():
                                    break
                                    
                                review_id = r.get("reviewId")
                                raw_content = (r.get("content") or "").strip()
                                user = r.get("userName") or "Unknown"

                                if not review_id or not raw_content:
                                    skipped += 1
                                    continue
                                    
                                if not self.is_quality_content(raw_content):
                                    skipped += 1
                                    continue
                                    
                                content = self.clean_text(raw_content)
                                if not self.is_english_no_other_lang(content):
                                    skipped += 1
                                    continue

                                payload = {
                                    "review_id": review_id,
                                    "user_id": user,
                                    "content": content,
                                    "app_id": self.app_id,
                                    "scraped_at": datetime.now(timezone.utc).isoformat()
                                }
                                
                                try:
                                    future = producer.send(self.topic, key=review_id, value=payload)
                                    future.get(timeout=10)
                                    sent += 1
                                except KafkaTimeoutError:
                                    logger.warning(f"Timeout sending review {review_id}")
                                    skipped += 1
                                except Exception as send_error:
                                    logger.error(f"Failed to send review {review_id}: {send_error}")
                                    skipped += 1

                            if sent > 0:
                                producer.flush(timeout=10)
                                cycle_reviews_sent += sent
                                with self.review_count_lock:
                                    self.total_reviews_sent += sent
                                logger.info(f"Sent {sent} reviews, skipped {skipped} (cycle total: {cycle_reviews_sent})")
                                empty_results_count = 0  # Reset on successful batch
                            
                            consecutive_errors = 0  # Reset on success
                            
                            if self.stop_event.is_set():
                                break
                                
                            time.sleep(self.page_sleep)
                            
                        except Exception as e:
                            consecutive_errors += 1
                            logger.error(f"Scraping error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                            
                            if consecutive_errors >= max_consecutive_errors:
                                logger.error("Too many consecutive errors, stopping producer")
                                self.stop_event.set()
                                break
                                
                            time.sleep(5)
                            continuation_token = None  # Reset on error
                    
                    logger.info(f"Completed cycle #{cycle_count}: sent {cycle_reviews_sent} reviews (total: {self.total_reviews_sent})")
                    
                except Exception as cycle_error:
                    logger.error(f"Error in scraping cycle #{cycle_count}: {cycle_error}")
                    time.sleep(60)  # Wait 1 minute before next cycle

        except Exception as e:
            logger.error(f"Critical producer error: {e}")
            self.stop_event.set()
            raise CustomException(e, sys)
        finally:
            if producer:
                try:
                    producer.flush(timeout=10)
                    producer.close(timeout=10)
                except Exception as e:
                    logger.warning(f"Error closing producer: {e}")
            logger.info(f"Producer shutdown complete. Total reviews sent: {self.total_reviews_sent}")

    def consumer_store(self, batch_size=50, flush_interval=5):
        """Enhanced consumer with FIXED OffsetAndMetadata"""
        consumer = None
        try:
            consumer = self.build_consumer()
            buffer = []
            last_flush = time.time()
            consecutive_errors = 0
            max_consecutive_errors = 5

            logger.info("Starting consumer for storing reviews")

            while not self.stop_event.is_set():
                try:
                    records = consumer.poll(timeout_ms=500)
                    
                    if not records:
                        if buffer and (time.time() - last_flush >= flush_interval):
                            self.insert_batch(buffer)
                            consumer.commit()
                            buffer.clear()
                            last_flush = time.time()
                        continue

                    for tp, msgs in records.items():
                        for msg in msgs:
                            if self.stop_event.is_set():
                                break
                                
                            try:
                                data = msg.value
                                if not data or "review_id" not in data:
                                    continue
                                    
                                buffer.append(data)
                                
                                if len(buffer) >= batch_size:
                                    inserted = self.insert_batch(buffer)
                                    if inserted > 0:
                                        offset_metadata = OffsetAndMetadata(
                                            offset=msg.offset + 1, leader_epoch=-1,
                                            metadata=""
                                        )
                                        consumer.commit({tp: offset_metadata})
                                    buffer.clear()
                                    last_flush = time.time()
                                    
                            except Exception as msg_error:
                                logger.error(f"Error processing message: {msg_error}")
                                continue
                    
                    consecutive_errors = 0
                    
                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"Consumer error ({consecutive_errors}/{max_consecutive_errors}): {e}")
                    
                    if consecutive_errors >= max_consecutive_errors:
                        logger.error("Too many consecutive errors, stopping consumer")
                        self.stop_event.set()
                        break
                        
                    time.sleep(2)

            # Final flush
            if buffer:
                try:
                    self.insert_batch(buffer)
                    consumer.commit()
                    logger.info(f"Final flush: inserted {len(buffer)} reviews")
                except Exception as e:
                    logger.error(f"Error in final flush: {e}")

        except Exception as e:
            logger.error(f"Critical consumer error: {e}")
            raise CustomException(e, sys)
        finally:
            if consumer:
                try:
                    consumer.close()
                except Exception as e:
                    logger.warning(f"Error closing consumer: {e}")
            logger.info("Consumer shutdown complete")

    def handle_signal(self, signum, frame):
        signal_names = {
            signal.SIGINT: "SIGINT (Ctrl+C)",
            signal.SIGTERM: "SIGTERM"
        }
        signal_name = signal_names.get(signum, f"Signal {signum}")
        
        logger.info(f"{signal_name} received, stopping pipeline gracefully...")
        self.stop_event.set()

    def get_stats(self):
        """Get pipeline statistics from database"""
        try:
            conn = self.get_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_reviews,
                        COUNT(DISTINCT app_id) as unique_apps,
                        MIN(created_at) as first_review,
                        MAX(created_at) as latest_review
                    FROM reviews;
                """)
                stats = cur.fetchone()
                return {
                    'total_reviews': stats[0],
                    'unique_apps': stats[1], 
                    'first_review': stats[2],
                    'latest_review': stats[3],
                    'reviews_sent': self.total_reviews_sent
                }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'total_reviews': 0, 'reviews_sent': self.total_reviews_sent}
        finally:
            if conn:
                conn.close()


# Runner remains the same
if __name__ == "__main__":
    pipeline = ReviewPipeline()
    
    signal.signal(signal.SIGINT, pipeline.handle_signal)
    signal.signal(signal.SIGTERM, pipeline.handle_signal)
    
    try:
        pipeline.init_local_db()
        
        producer_thread = threading.Thread(
            target=pipeline.produce_reviews, 
            daemon=True,
            name="ReviewProducer"
        )
        
        consumer_thread = threading.Thread(
            target=pipeline.consumer_store,
            daemon=True, 
            name="ReviewConsumer"
        )
        
        producer_thread.start()
        consumer_thread.start()
        
        logger.info("Continuous pipeline started successfully")
        
        while not pipeline.stop_event.is_set():
            time.sleep(30)
            
            if not producer_thread.is_alive():
                logger.error("Producer thread died")
                pipeline.stop_event.set()
                break
                
            if not consumer_thread.is_alive():
                logger.error("Consumer thread died") 
                pipeline.stop_event.set()
                break
            
            stats = pipeline.get_stats()
            logger.info(f"Pipeline stats: {stats}")
        
        logger.info("Waiting for threads to complete...")
        producer_thread.join(timeout=30)
        consumer_thread.join(timeout=30)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        pipeline.handle_signal(signal.SIGINT, None)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
    finally:
        logger.info("Pipeline shutdown complete")
