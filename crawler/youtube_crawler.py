import os
import logging
import pandas as pd
import json
import time
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from dotenv import load_dotenv
import subprocess
from kafka import KafkaProducer
from kafka.errors import KafkaError

# Thiết lập logging
logging.basicConfig(
    filename=os.path.join("Data", "youtube_crawler.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def log(msg, level="INFO"):
    getattr(logging, level.lower())(msg)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

# Cấu hình
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
DATA_DIR = "Data"
os.makedirs(DATA_DIR, exist_ok=True)
youtube = build("youtube", "v3", developerKey=API_KEY)

# Cấu hình Kafka
KAFKA_BOOTSTRAP_SERVERS = "localhost:29092"
KAFKA_TOPIC = "youtube_data"
try:
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        max_block_ms=120000,
        retries=5,
        acks="all",
        max_request_size=1048576
    )
    log("Kết nối Kafka thành công")
except KafkaError as e:
    log(f"Lỗi khởi tạo KafkaProducer: {e}", level="ERROR")
    raise

def save_to_csv(data, data_type, batch_number=None):
    if not data:
        log(f"Không có dữ liệu để lưu vào {data_type}", level="WARNING")
        return
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_suffix = f"_batch{batch_number}" if batch_number is not None else ""
        filename = os.path.join(DATA_DIR, f"{data_type}_{timestamp}{batch_suffix}.csv")
        pd.DataFrame(data).to_csv(filename, index=False, encoding="utf-8-sig")
        log(f"Đã lưu {len(data)} bản ghi vào {filename}")

        # Đẩy vào HDFS
        hdfs_path = f"hdfs://host.docker.internal:9000/Data/{data_type.capitalize()}/{data_type}_{timestamp}{batch_suffix}.csv"
        subprocess.run(["hdfs", "dfs", "-mkdir", "-p", f"hdfs://host.docker.internal:9000/Data/{data_type.capitalize()}"], check=True)
        subprocess.run(["hdfs", "dfs", "-put", filename, hdfs_path], check=True)
        log(f"Đã đẩy {filename} vào {hdfs_path}")

        # Đẩy vào Kafka
        for record in data:
            future = producer.send(KAFKA_TOPIC, value=record)
            future.get(timeout=30)
        producer.flush()
        log(f"Đã gửi {len(data)} bản ghi vào Kafka topic {KAFKA_TOPIC}")
    except subprocess.CalledProcessError as e:
        log(f"Lỗi khi đẩy vào HDFS: {e}", level="ERROR")
    except KafkaError as e:
        log(f"Lỗi khi gửi Kafka: {e}", level="ERROR")
        raise
    except Exception as e:
        log(f"Lỗi không xác định: {e}", level="ERROR")
        raise

def get_videos(region_code, max_videos):
    content_data = []
    video_ids = []
    quota_used = 0
    batch_size = 50
    batch_number = 0
    try:
        request = youtube.videos().list(
            part="id,snippet,statistics,contentDetails",
            chart="mostPopular",
            regionCode=region_code,
            maxResults=50
        )
        while request and len(video_ids) < max_videos:
            response = request.execute()
            quota_used += 100  # videos.list ~100 unit
            log(f"Quota API đã sử dụng: {quota_used} unit")
            if quota_used > 9000:
                log("Gần vượt quota API YouTube (10,000 unit/ngày), dừng thu thập video", level="WARNING")
                break
            for item in response["items"]:
                video_ids.append(item["id"])
                content_data.append({
                    "content_id": item["id"],
                    "platform": "youtube",
                    "title": item["snippet"]["title"],
                    "content": item["snippet"].get("description", "")[:1000],
                    "created_at": datetime.strptime(item["snippet"]["publishedAt"], "%Y-%m-%dT%H:%M:%SZ").isoformat() + "Z",
                    "source_id": item["snippet"]["channelId"],
                    "source_name": item["snippet"]["channelTitle"],
                    "category_id": item["snippet"].get("categoryId", "unknown"),
                    "category_name": item["snippet"].get("categoryId", "unknown"),
                    "tags": json.dumps(item["snippet"].get("tags", [])),
                    "views": int(item["statistics"].get("viewCount", 0)),
                    "score": int(item["statistics"].get("likeCount", 0)),
                    "comment_count": int(item["statistics"].get("commentCount", 0)),
                    "duration": item["contentDetails"].get("duration", ""),
                    "upvote_ratio": 0.0,
                    "url": f"https://www.youtube.com/watch?v={item['id']}",
                    "author": item["snippet"]["channelTitle"]
                })
                if len(content_data) >= batch_size:
                    save_to_csv(content_data, "contents", batch_number)
                    content_data = []
                    batch_number += 1
            if len(video_ids) >= max_videos:
                break
            request = youtube.videos().list_next(request, response)
            time.sleep(1)  # Nghỉ 1 giây mỗi request
        if content_data:  # Lưu batch cuối
            save_to_csv(content_data, "contents", batch_number)
    except HttpError as e:
        log(f"Lỗi khi thu thập video: {e}", level="ERROR")
    return video_ids

def get_comments(video_id, max_comments):
    comment_data = []
    quota_used = 0
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=50
        )
        count = 0
        while request and count < max_comments:
            response = request.execute()
            quota_used += 50  # commentThreads.list ~50 unit
            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]
                content = comment["textDisplay"][:1000].encode("utf-8", errors="ignore").decode("utf-8")
                comment_data.append({
                    "comment_id": item["id"],
                    "content_id": video_id,
                    "platform": "youtube",
                    "content": content,
                    "created_at": datetime.strptime(comment["publishedAt"], "%Y-%m-%dT%H:%M:%SZ").isoformat() + "Z",
                    "score": int(comment.get("likeCount", 0)),
                    "author": comment["authorDisplayName"],
                    "source_name": ""
                })
                count += 1
                if count >= max_comments:
                    break
            request = youtube.commentThreads().list_next(request, response)
            time.sleep(1)  # Nghỉ 1 giây mỗi request
        log(f"Quota API cho bình luận video {video_id}: {quota_used} unit")
    except HttpError as e:
        log(f"Lỗi khi thu thập bình luận cho video {video_id}: {e}", level="ERROR")
    return comment_data

def process_comments_batch(video_ids, max_comments, batch_size=500):
    comment_data_all = []
    batch_number = 0
    with ProcessPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(get_comments, video_id, max_comments) for video_id in video_ids]
        for future in futures:
            comment_data_all.extend(future.result())
            if len(comment_data_all) >= batch_size:
                save_to_csv(comment_data_all, "comments")
                comment_data_all = []
                batch_number += 1
    if comment_data_all:  # Lưu batch cuối
        save_to_csv(comment_data_all, "comments")
    return batch_number + 1

def main(region_code, max_videos, max_comments):
    start_time = datetime.now()
    log(f"BẮT ĐẦU THU THẬP DỮ LIỆU YOUTUBE: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    video_ids = get_videos(region_code, max_videos)
    if video_ids:
        batch_count = process_comments_batch(video_ids, max_comments)

    end_time = datetime.now()
    log(f"HOÀN TẤT THU THẬP DỮ LIỆU: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Tổng thời gian chạy: {(end_time - start_time).total_seconds() / 3600:.2f} giờ")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", default="VN", help="Mã vùng (e.g., VN, US)")
    parser.add_argument("--max-videos", type=int, default=10, help="Số lượng video tối đa")
    parser.add_argument("--max-comments", type=int, default=10, help="Số lượng bình luận tối đa mỗi video")
    args = parser.parse_args()
    main(args.region, args.max_videos, args.max_comments)