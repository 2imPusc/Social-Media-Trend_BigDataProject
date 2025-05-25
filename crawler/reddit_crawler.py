import os
import time
import logging
import pandas as pd
import json
import praw
from datetime import datetime
import subprocess
from concurrent.futures import ThreadPoolExecutor
import praw.exceptions
import prawcore.exceptions
from kafka import KafkaProducer

# Cấu hình
DATA_DIR = "Data"
MAX_THREADS = 2
LIMIT_POSTS = 1000
LIMIT_COMMENTS = 100
BATCH_SIZE = 500

# Cấu hình Kafka
KAFKA_BOOTSTRAP_SERVERS = 'host.docker.internal:29092'
KAFKA_TOPIC = 'reddit_data'
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Thiết lập logging
logging.basicConfig(
    filename=os.path.join(DATA_DIR, 'reddit_crawler.log'),
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

def log(msg):
    logging.info(msg)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# Đảm bảo thư mục Data tồn tại
os.makedirs(DATA_DIR, exist_ok=True)

# Khởi tạo Reddit instance
reddit = praw.Reddit(
    client_id='-azzbUReAtS9-o41ZbGNrQ',
    client_secret='Vb9gsVUPKL1nViZkF0Tp35rz9K7Ruw',
    user_agent='2imPusc_'
)

# Danh sách subreddit
subreddits_by_field = {
    "Technology": ["technology", "gadgets", "programming", "science", "todayilearned"],
    "News": ["worldnews", "news", "VietNam", "AskReddit"],
    "Entertainment": ["movies", "gaming", "music", "funny", "memes"],
    "Lifestyle": ["fitness", "food", "travel", "aww"]
}

# Kiểm tra giới hạn API
def check_rate_limit(post_count):
    if post_count % 50 != 0:
        return
    limits = reddit.auth.limits
    remaining = limits.get('remaining', float('inf'))
    reset = limits.get('reset_timestamp', time.time() + 60)
    log(f"API còn lại: {remaining} yêu cầu, reset sau {(reset - time.time())/60:.2f} phút")
    if remaining < 50:
        wait_time = reset - time.time() + 10
        if wait_time > 0:
            log(f"Đang chờ {wait_time:.2f}s do giới hạn API...")
            time.sleep(wait_time)

# Hàm lưu dữ liệu
def save_data(data, data_type):
    if not data:
        logging.warning(f"Không có dữ liệu để lưu vào {data_type}.")
        return
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = os.path.join(DATA_DIR, f"{data_type}_{timestamp}.csv")
    pd.DataFrame(data).to_csv(filename, index=False, encoding='utf-8-sig')
    log(f"Đã lưu {len(data)} bản ghi vào {filename}")

    # Đẩy vào HDFS
    hdfs_path = f"hdfs://host.docker.internal:9000/Data/{data_type.capitalize()}/{data_type}_{timestamp}.csv"
    try:
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", f"hdfs://host.docker.internal:9000/Data/{data_type.capitalize()}"], check=True)
        subprocess.run(["hdfs", "dfs", "-put", filename, hdfs_path], check=True)
        log(f"Đã đẩy {filename} vào {hdfs_path}")
    except subprocess.CalledProcessError as e:
        log(f"Lỗi khi đẩy vào HDFS: {e}")

    # Đẩy vào Kafka
    try:
        for record in data:
            producer.send(KAFKA_TOPIC, value=record)
        producer.flush()
        log(f"Đã gửi {len(data)} bản ghi vào Kafka topic {KAFKA_TOPIC}")
    except Exception as e:
        log(f"Lỗi khi gửi vào Kafka: {e}")

# Thu thập dữ liệu từ subreddit
def collect_data(field, subreddit_name):
    content_data = []
    comment_data = []
    retries = 3
    post_count = 0
    for attempt in range(retries):
        try:
            log(f"Đang thu thập từ r/{subreddit_name}... (Lần thử {attempt + 1})")
            subreddit = reddit.subreddit(subreddit_name)
            subreddit.display_name
            for post in subreddit.top(limit=LIMIT_POSTS, time_filter="month"):
                check_rate_limit(post_count)
                post_count += 1
                time.sleep(0.5)

                content_data.append({
                    "content_id": post.id,
                    "platform": "reddit",
                    "title": post.title,
                    "content": post.selftext,
                    "created_at": datetime.utcfromtimestamp(post.created_utc).isoformat() + "Z",
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
                })

                post.comments.replace_more(limit=0)
                for comment in post.comments[:LIMIT_COMMENTS]:
                    comment_data.append({
                        "comment_id": comment.id,
                        "content_id": post.id,
                        "platform": "reddit",
                        "content": comment.body,
                        "created_at": datetime.utcfromtimestamp(comment.created_utc).isoformat() + "Z",
                        "score": comment.score,
                        "author": str(comment.author) if comment.author else "N/A",
                        "source_name": subreddit_name,
                        "crawl_time": datetime.utcnow().isoformat() + "Z"
                    })

                if len(content_data) >= BATCH_SIZE or post_count == LIMIT_POSTS:
                    save_data(content_data, "contents")
                    save_data(comment_data, "comments")

                    log(f"Đã thu thập {len(content_data)} bài đăng từ r/{subreddit_name}")
                    content_data, comment_data = [], []

            if content_data:
                save_data(content_data, "contents")
                save_data(comment_data, "comments")
                log(f"Đã thu thập {len(content_data)} bài đăng từ r/{subreddit_name}")

            return content_data, comment_data

        except prawcore.exceptions.Forbidden:
            log(f"Subreddit r/{subreddit_name} bị chặn hoặc riêng tư")
            return [], []
        except prawcore.exceptions.NotFound:
            log(f"Subreddit r/{subreddit_name} không tồn tại")
            return [], []
        except (praw.exceptions.RedditAPIException, prawcore.exceptions.RequestException) as e:
            log(f"Lỗi API khi thu thập r/{subreddit_name}: {e}")
            if attempt < retries - 1:
                wait_time = 2 ** attempt * 60
                log(f"Thử lại sau {wait_time}s...")
                time.sleep(wait_time)
            else:
                log(f"Đã thử {retries} lần, bỏ qua r/{subreddit_name}")
                return [], []
        except Exception as e:
            log(f"Lỗi không xác định khi thu thập r/{subreddit_name}: {e}")
            return [], []

# Chạy chương trình
start_time = datetime.now()
log(f"BẮT ĐẦU THU THẬP DỮ LIỆU: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

content_data_all = []
comment_data_all = []
for field, subreddit_list in subreddits_by_field.items():
    log(f"\n--- LĨNH VỰC: {field} ---")
    try:
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = [executor.submit(collect_data, field, sub) for sub in subreddit_list]
            for future in futures:
                try:
                    content_data, comment_data = future.result()
                    content_data_all.extend(content_data)
                    comment_data_all.extend(comment_data)
                except Exception as e:
                    log(f"Lỗi trong thread: {e}")
    except Exception as e:
        log(f"Lỗi cấp cao trong lĩnh vực {field}: {e}")

if content_data_all:
    save_data(content_data_all, "contents")

if comment_data_all:
    save_data(comment_data_all, "comments")

end_time = datetime.now()
log(f"\nHOÀN TẤT THU THẬP DỮ LIỆU: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Tổng thời gian chạy: {(end_time - start_time).total_seconds() / 3600:.2f} giờ")