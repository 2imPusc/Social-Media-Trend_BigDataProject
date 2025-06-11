import os
import json
import time
import logging
from datetime import datetime, date
import praw
import prawcore
from kafka import KafkaProducer

# Cấu hình
DATA_DIR = "Data"
os.makedirs(DATA_DIR, exist_ok=True)

CRAWLED_POST_IDS_FILE = os.path.join(DATA_DIR, "crawled_post_ids_stream.json")
KAFKA_BOOTSTRAP_SERVERS = 'host.docker.internal:29092'
KAFKA_TOPIC = 'reddit_stream'

LIMIT_COMMENTS = 50

# Logging
logging.basicConfig(
    filename=os.path.join(DATA_DIR, 'reddit_stream.log'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
def log(msg):
    logging.info(msg)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# Kafka
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Reddit API
reddit = praw.Reddit(
    client_id='-azzbUReAtS9-o41ZbGNrQ',
    client_secret='Vb9gsVUPKL1nViZkF0Tp35rz9K7Ruw',
    user_agent='2imPusc_'
)

# Subreddit danh mục
subreddits_by_field = {
    "Technology": ["technology", "gadgets", "programming", "science", "todayilearned"],
    "News": ["worldnews", "news", "VietNam", "AskReddit"],
    "Entertainment": ["movies", "gaming", "music", "funny", "memes"],
    "Lifestyle": ["fitness", "food", "travel", "aww"]
}

# Load crawled post ids theo ngày
def load_crawled_ids():
    today_str = date.today().isoformat()
    if os.path.exists(CRAWLED_POST_IDS_FILE):
        with open(CRAWLED_POST_IDS_FILE, 'r') as f:
            try:
                data = json.load(f)
                if data.get("date") == today_str:
                    return set(data.get("ids", [])), today_str
            except Exception:
                pass
    # Nếu chưa có file hoặc khác ngày thì reset
    return set(), today_str

def save_crawled_ids(crawled_ids, today_str):
    with open(CRAWLED_POST_IDS_FILE, 'w') as f:
        json.dump({"date": today_str, "ids": list(crawled_ids)}, f)

# Tạo list subreddit
all_subreddits = []
subreddit_to_field = {}
for field, subs in subreddits_by_field.items():
    all_subreddits.extend(subs)
    for sub in subs:
        subreddit_to_field[sub] = field

# Khởi tạo biến toàn cục cho danh sách đã crawl và ngày
crawled_post_ids, crawled_date = load_crawled_ids()

# Stream
def stream_reddit():
    global crawled_post_ids, crawled_date
    log("Bắt đầu stream dữ liệu từ Reddit...")
    try:
        subreddit = reddit.subreddit("+".join(all_subreddits))
        for post in subreddit.stream.submissions(skip_existing=True):
            # Kiểm tra ngày, nếu sang ngày mới thì reset
            today_str = date.today().isoformat()
            if today_str != crawled_date:
                crawled_post_ids = set()
                crawled_date = today_str
                log("Reset danh sách bài viết đã crawl cho ngày mới.")

            if post.id in crawled_post_ids:
                continue

            subreddit_name = post.subreddit.display_name
            field = subreddit_to_field.get(subreddit_name, "Unknown")
            created_time = datetime.utcfromtimestamp(post.created_utc).isoformat() + "Z"

            post_data = {
                "content_id": post.id,
                "platform": "reddit",
                "title": post.title,
                "content": post.selftext,
                "created_at": created_time,
                "source_id": subreddit_name,
                "source_name": subreddit_name,
                "category_id": field.lower(),
                "category_name": field,
                "tags": json.dumps([]),
                "views": 0,
                "score": post.score,
                "comment_count": post.num_comments,
                "duration": "",
                "upvote_ratio": post.upvote_ratio,
                "url": post.url,
                "author": str(post.author) if post.author else "N/A",
                "crawl_time": datetime.utcnow().isoformat() + "Z"
            }

            # Gửi bài viết vào Kafka
            producer.send(KAFKA_TOPIC, value={"type": "post", "data": post_data})
            producer.flush()
            log(f"Đã gửi post {post.id} từ r/{subreddit_name}")

            # Lấy comment (top comment)
            post.comments.replace_more(limit=0)
            for comment in post.comments[:LIMIT_COMMENTS]:
                comment_data = {
                    "comment_id": comment.id,
                    "content_id": post.id,
                    "platform": "reddit",
                    "content": comment.body,
                    "created_at": datetime.utcfromtimestamp(comment.created_utc).isoformat() + "Z",
                    "score": comment.score,
                    "author": str(comment.author) if comment.author else "N/A",
                    "source_name": subreddit_name,
                    "crawl_time": datetime.utcnow().isoformat() + "Z"
                }
                producer.send(KAFKA_TOPIC, value={"type": "comment", "data": comment_data})
            producer.flush()

            # Lưu ID đã thu
            crawled_post_ids.add(post.id)
            save_crawled_ids(crawled_post_ids, crawled_date)

            time.sleep(1)

    except Exception as e:
        log(f"Lỗi khi stream: {e}")
        time.sleep(30)
        stream_reddit()

if __name__ == "__main__":
    stream_reddit()
