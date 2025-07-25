from google_play_scraper import reviews, Sort
from confluent_kafka import Producer
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, json, time

def run_producer():
    load_dotenv()

    producer = Producer({
        'bootstrap.servers': os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    })

    apps = [
        'com.snapchat.android', 'com.linkedin.android', 'org.telegram.messenger',
        'com.whatsapp', 'com.facebook.katana', 'com.instagram.android',
        'com.twitter.android'
    ]

    def delivery_report(err, msg):
        if err is not None:
            print(f"Delivery failed for topic {msg.topic()}: {err}")

    def scrape_reviews(app_id):
        count = 0
        token = None
        seen_reviews = set()
        topic = app_id.replace(".", "_")  

        while count < 20000:
            result, token = reviews(
                app_id, lang='en', country='us', sort=Sort.NEWEST,
                count=200, continuation_token=token
            )

            if not result:
                break

            for r in result:
                unique_id = (r['userName'], r['content'])
                if unique_id in seen_reviews:
                    continue

                seen_reviews.add(unique_id)

                review = {
                    "appId": app_id,
                    "userName": r["userName"],
                    "content": r["content"],
                    "score": r["score"]
                }

                producer.produce(topic, json.dumps(review), callback=delivery_report)
                count += 1

                if count >= 20000:
                    break

            time.sleep(1)  

        producer.flush()
        print(f"[{app_id}] Finished sending {count} reviews to topic '{topic}'")

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(scrape_reviews, app) for app in apps]
        for future in as_completed(futures):
            future.result()

    producer.flush()


if __name__ == "__main__":
    run_producer()
