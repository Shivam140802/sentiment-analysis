import sys
import threading
import time
import signal
from predict_pipeline.kafka_pipeline import ReviewPipeline
from predict_pipeline.predict import ReviewPredictorPipeline
from utils.logger import logger
from utils.exception import CustomException

class EndToEndPipeline:
    """End-to-end pipeline: Scrape → Store → Predict → Dashboard"""

    def __init__(self, predict_interval=60, batch_size=200):
        self.kafka_pipeline = ReviewPipeline()
        self.predict_pipeline = ReviewPredictorPipeline()
        self.predict_interval = predict_interval
        self.batch_size = batch_size
        self.last_processed_id = 0
        self.shutdown_event = threading.Event()
        self.prediction_thread = None
        
        logger.info(f"Pipeline initialized with {predict_interval}s interval, batch size: {batch_size}")

    def start_kafka_threads(self):
        """Start Kafka producer and consumer threads"""
        try:
            producer_thread = threading.Thread(
                target=self.kafka_pipeline.produce_reviews, 
                daemon=True,
                name="KafkaProducer"
            )
            consumer_thread = threading.Thread(
                target=self.kafka_pipeline.consumer_store, 
                daemon=True,
                name="KafkaConsumer"
            )
            
            producer_thread.start()
            consumer_thread.start()
            
            logger.info("Kafka producer and consumer threads started")
            return producer_thread, consumer_thread
            
        except Exception as e:
            logger.error(f"Failed to start Kafka threads: {e}")
            raise CustomException(e, sys)

    def wait_for_initial_reviews(self, timeout=30):
        """Wait for initial reviews to appear in the database"""
        logger.info("Waiting for initial reviews to appear in database...")
        
        for i in range(timeout):
            try:
                df = self.predict_pipeline.fetch_new_reviews(0, limit=1)
                if not df.empty:
                    self.last_processed_id = df['id'].max()
                    logger.info(f"Found initial reviews. Starting from ID: {self.last_processed_id}")
                    return True
            except Exception as e:
                logger.warning(f"Error checking for reviews: {e}")
                
            if self.shutdown_event.is_set():
                return False
                
            time.sleep(1)
            if i % 10 == 0:
                logger.info(f"Still waiting for reviews... ({i}/{timeout}s)")
        
        logger.info("No initial reviews found, will process as they arrive")
        return False

    def prediction_loop(self):
        """Main prediction processing loop"""
        logger.info("Starting prediction loop")
        
        # Create prediction table
        self.predict_pipeline.create_table_if_not_exists()
        
        # Wait for initial reviews
        self.wait_for_initial_reviews()
        
        consecutive_empty_batches = 0
        max_empty_batches = 5  # Increase sleep time after several empty batches
        
        while not self.shutdown_event.is_set():
            try:
                # Fetch new reviews
                df_new = self.predict_pipeline.fetch_new_reviews(
                    self.last_processed_id, 
                    limit=self.batch_size
                )
                
                if not df_new.empty:
                    logger.info(f"Processing {len(df_new)} new reviews")
                    
                    # Generate embeddings
                    embeddings = self.predict_pipeline.embed_reviews(df_new)
                    
                    # Make predictions
                    df_predictions = self.predict_pipeline.predict_reviews(df_new, embeddings)
                    
                    # Save predictions
                    self.predict_pipeline.save_predictions(df_predictions)
                    
                    # Update tracking
                    self.last_processed_id = df_new['id'].max()
                    consecutive_empty_batches = 0
                    
                    logger.info(f"Completed batch processing. Last processed ID: {self.last_processed_id}")
                    
                    # Log prediction stats periodically
                    if self.last_processed_id % 100 == 0:
                        stats = self.predict_pipeline.get_prediction_stats()
                        logger.info(f"Prediction stats: {stats}")
                        
                else:
                    consecutive_empty_batches += 1
                    logger.debug(f"No new reviews to process ({consecutive_empty_batches} consecutive empty batches)")
                
                # Dynamic sleep based on activity
                if consecutive_empty_batches > max_empty_batches:
                    sleep_time = min(self.predict_interval * 2, 300)  # Max 5 minutes
                else:
                    sleep_time = self.predict_interval
                    
                # Wait with ability to interrupt
                if self.shutdown_event.wait(sleep_time):
                    break
                    
            except Exception as e:
                logger.error(f"Error in prediction loop: {e}")
                # Don't raise here, continue processing
                if not self.shutdown_event.wait(self.predict_interval):
                    continue
                else:
                    break
        
        logger.info("Prediction loop stopped")

    def handle_shutdown(self, signum=None, frame=None):
        """Handle shutdown signals"""
        logger.info(f"Shutdown signal received: {signum}")
        self.shutdown_event.set()
        
        # Signal Kafka pipeline to stop
        if hasattr(self.kafka_pipeline, 'stop_event'):
            self.kafka_pipeline.stop_event.set()

    def run(self):
        """Run the complete end-to-end pipeline"""
        try:
            logger.info("Starting End-to-End Pipeline")
            
            # Initialize database
            self.kafka_pipeline.init_local_db()
            
            # Setup signal handlers
            signal.signal(signal.SIGINT, self.handle_shutdown)
            signal.signal(signal.SIGTERM, self.handle_shutdown)
            
            # Start Kafka threads
            producer_thread, consumer_thread = self.start_kafka_threads()
            
            # Start prediction thread
            self.prediction_thread = threading.Thread(
                target=self.prediction_loop,
                daemon=True,
                name="PredictionLoop"
            )
            self.prediction_thread.start()
            
            logger.info("All pipeline components started successfully")
            
            # Main loop - keep the main thread alive
            try:
                while not self.shutdown_event.is_set():
                    # Check if critical threads are still alive
                    if not producer_thread.is_alive():
                        logger.error("Producer thread died unexpectedly")
                        break
                    if not consumer_thread.is_alive():
                        logger.error("Consumer thread died unexpectedly")
                        break
                    if not self.prediction_thread.is_alive():
                        logger.error("Prediction thread died unexpectedly")
                        break
                    
                    # Use signal.pause() on Unix systems, or sleep on Windows
                    try:
                        signal.pause()
                    except AttributeError:
                        # Windows doesn't have signal.pause()
                        time.sleep(1)
                    
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                self.handle_shutdown(signal.SIGINT)
            
        except Exception as e:
            logger.error(f"Pipeline startup failed: {e}")
            raise CustomException(e, sys)
        
        finally:
            self.shutdown()

    def shutdown(self):
        """Graceful shutdown of all components"""
        logger.info("Initiating pipeline shutdown...")
        
        # Set shutdown event
        self.shutdown_event.set()
        
        # Wait for threads to complete (with timeout)
        threads_to_wait = []
        
        if hasattr(self, 'prediction_thread') and self.prediction_thread:
            threads_to_wait.append(('Prediction', self.prediction_thread))
        
        for thread_name, thread in threads_to_wait:
            try:
                thread.join(timeout=30)  # 30 second timeout
                if thread.is_alive():
                    logger.warning(f"{thread_name} thread did not stop gracefully")
                else:
                    logger.info(f"{thread_name} thread stopped successfully")
            except Exception as e:
                logger.error(f"Error stopping {thread_name} thread: {e}")
        
        logger.info("Pipeline shutdown completed")

if __name__ == "__main__":
    # Parse command line arguments if needed
    predict_interval = 60  # Default 60 seconds
    batch_size = 200       # Default batch size
    
    if len(sys.argv) > 1:
        try:
            predict_interval = int(sys.argv[1])
        except ValueError:
            logger.warning("Invalid predict_interval argument, using default 60s")
    
    if len(sys.argv) > 2:
        try:
            batch_size = int(sys.argv[2])
        except ValueError:
            logger.warning("Invalid batch_size argument, using default 200")
    
    # Run pipeline
    pipeline = EndToEndPipeline(
        predict_interval=predict_interval,
        batch_size=batch_size
    )
    pipeline.run()
