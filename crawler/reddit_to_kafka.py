import praw
from kafka import KafkaProducer
import json
import time

# Thiết lập PRAW (Reddit API)
reddit = praw.Reddit(
    client_id='-azzbUReAtS9-o41ZbGNrQ',
    client_secret='Vb9gsVUPKL1nViZkF0Tp35rz9K7Ruw',
    user_agent='reddit_kafka_streamer'
)

# Thiết lập Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Cấu hình
TOPIC = 'reddit_stream'
SUBREDDIT_NAME = 'technology'  # Có thể thay bằng lĩnh vực khác

def get_reddit_post_data(post):
    return {
        'id': post.id,
        'title': post.title,
        'score': post.score,
        'created_utc': post.created_utc,
        'subreddit': post.subreddit.display_name,
        'url': post.url,
        'num_comments': post.num_comments,
        'selftext': post.selftext,
        'author': str(post.author)
    }

def stream_reddit_posts():
    print(f"Streaming from r/{SUBREDDIT_NAME} to Kafka topic '{TOPIC}'...")
    for post in reddit.subreddit(SUBREDDIT_NAME).stream.submissions(skip_existing=True):
        try:
            data = get_reddit_post_data(post)
            producer.send(TOPIC, value=data)
            print(f"[+] Sent: {data['title'][:60]}...")
        except Exception as e:
            print(f"[!] Error: {e}")
        time.sleep(1)

if __name__ == '__main__':
    stream_reddit_posts()
