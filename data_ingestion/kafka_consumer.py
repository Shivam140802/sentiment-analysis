from confluent_kafka import Consumer
import boto3, os, json, threading
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import io
import time

load_dotenv()

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
AWS_S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
S3_REGION = 'eu-north-1'
FOLDER = 'structured_data/'
MAX_EMPTY_POLLS = 10  # max number of consecutive empty polls before stopping

apps = [
    'com.snapchat.android', 'com.linkedin.android', 'org.telegram.messenger',
    'com.whatsapp', 'com.facebook.katana', 'com.instagram.android',
    'com.twitter.android'
]

def upload_to_s3(app_id, records):
    s3 = boto3.client('s3', region_name=S3_REGION)
    file_name = f"{FOLDER}{app_id.replace('.', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"

    df = pd.DataFrame(records)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    s3.put_object(Bucket=AWS_S3_BUCKET_NAME, Key=file_name, Body=csv_buffer.getvalue())
    print(f"[{app_id}] Uploaded: {file_name}")

def consume_topic(app_id):
    topic = app_id.replace(".", "_")
    consumer = Consumer({
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        'group.id': f'{topic}-group',
        'auto.offset.reset': 'earliest'
    })

    consumer.subscribe([topic])
    buffer = []
    empty_polls = 0

    try:
        print(f"[{app_id}] Consumer started for topic '{topic}'")
        while empty_polls < MAX_EMPTY_POLLS:
            msg = consumer.poll(1.0)
            if msg is None:
                empty_polls += 1
                continue
            if msg.error():
                print(f"[{app_id}] Consumer error: {msg.error()}")
                continue

            review = json.loads(msg.value().decode('utf-8'))
            buffer.append(review)
            empty_polls = 0  # reset since we got a valid message

            if len(buffer) >= 20000:
                upload_to_s3(app_id, buffer)
                buffer.clear()

    except KeyboardInterrupt:
        print(f"[{app_id}] Stopping consumer manually...")
    finally:
        if buffer:
            upload_to_s3(app_id, buffer)
        consumer.close()
        print(f"[{app_id}] Consumer finished.")

def run_consumer():
    threads = []
    for app_id in apps:
        t = threading.Thread(target=consume_topic, args=(app_id,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

if __name__ == "__main__":
    run_consumer()