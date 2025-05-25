from pyspark.sql import SparkSession
from pyspark.sql.functions import log1p
from pyspark.sql.functions import (
    col, lower, explode, split, regexp_replace, avg, count,
    coalesce, to_timestamp, when, udf, lit, trim, length
)
from pyspark.sql.types import IntegerType, FloatType, ArrayType, StringType
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
import nltk

# Tải NLTK stopwords nếu cần thiết và tạo set để broadcast
try:
    from nltk.corpus import stopwords
    nltk_stop_words_list = stopwords.words('english')
except LookupError:
    print("Tài nguyên 'stopwords' không tìm thấy trên driver. Đang tải xuống...")
    nltk.download('stopwords', quiet=True) # quiet=True để không in nhiều log
    from nltk.corpus import stopwords
    nltk_stop_words_list = stopwords.words('english')

stop_words_set_for_broadcast = set(nltk_stop_words_list)


# Khởi tạo Spark session
spark = SparkSession.builder \
    .appName("RedditTrendAnalysis") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# Broadcast stop_words
broadcast_stop_words = spark.sparkContext.broadcast(stop_words_set_for_broadcast)

# Đọc dữ liệu từ HDFS
contents_df = spark.read.option("header", "true").option("escape", "\"").option("multiline", "true").csv("hdfs://host.docker.internal:9000/Data/Contents/*.csv")
comments_df = spark.read.option("header", "true").option("escape", "\"").option("multiline", "true").csv("hdfs://host.docker.internal:9000/Data/Comments/*.csv")

# Chuyển đổi kiểu dữ liệu
contents_df = contents_df.withColumn("score", col("score").cast(IntegerType())) \
                        .withColumn("comment_count", col("comment_count").cast(IntegerType())) \
                        .withColumn("created_at", to_timestamp(col("created_at"))) \
                        .withColumn("upvote_ratio", col("upvote_ratio").cast(FloatType()))
comments_df = comments_df.withColumn("score", col("score").cast(IntegerType()))

# Làm sạch dữ liệu
contents_df = contents_df.filter(col("content").isNotNull() & (length(trim(col("content"))) > 0))
comments_df = comments_df.filter(
    col("comment_id").rlike(r"^[a-z0-9]+$") &
    col("content").isNotNull() &
    (length(trim(col("content"))) > 0)
)

# Hàm đánh giá cảm xúc
def get_sentiment_func(text): # Đổi tên hàm để tránh trùng với tên cột "sentiment"
    if not text:
        return 0.0
    try:
        from textblob import TextBlob # Import bên trong UDF
        return TextBlob(text).sentiment.polarity
    except Exception: # Bắt lỗi cụ thể hơn nếu có thể, hoặc log lỗi
        return 0.0

# Không cần spark.udf.register nếu bạn dùng @udf decorator hoặc gán trực tiếp
sentiment_udf = udf(get_sentiment_func, FloatType())

# Phân tích cảm xúc
contents_df = contents_df.withColumn("sentiment_score", sentiment_udf(col("content"))) # Đổi tên cột để rõ ràng hơn
comments_df = comments_df.withColumn("sentiment_score", sentiment_udf(col("content")))

# Kết hợp bình luận với nội dung
joined_df = comments_df.groupBy("content_id").agg(
    avg("sentiment_score").alias("avg_comment_sentiment"),
    count("comment_id").alias("total_comments")
).join(contents_df, on="content_id", how="inner")

# Điền giá trị mặc định cho score và comment_count
joined_df = joined_df.withColumn("score", coalesce(col("score").cast("float"), lit(0.0))) \
                    .withColumn("comment_count", coalesce(col("comment_count").cast("float"), lit(0.0)))

# Tính base_sentiment_score
result_df_intermediate = joined_df.withColumn( # Sử dụng tên biến trung gian khác để chắc chắn
    "base_sentiment_score",
    (coalesce(col("sentiment_score"), lit(0.0)) * 0.6 + coalesce(col("avg_comment_sentiment"), lit(0.0)) * 0.4)
)

# Tính interaction_factor (là một biểu thức cột)
interaction_factor_expr = (log1p(col("score")) + log1p(col("comment_count")))

result_df = result_df_intermediate.withColumn(
    "total_sentiment_score",
    col("base_sentiment_score") * (lit(1.0) + interaction_factor_expr * lit(0.1))
)

# UDF loại bỏ stopwords sử dụng broadcast variable
def remove_stopwords_udf_logic(words_list):
    if not words_list: # Handles None or empty list
        return []
    # Lấy giá trị từ biến broadcast
    current_stop_words = broadcast_stop_words.value
    # Lọc thêm các từ rỗng có thể sinh ra từ split nếu có nhiều khoảng trắng liên tiếp
    return [word for word in words_list if word and word.lower() not in current_stop_words]

remove_stopwords_udf = udf(remove_stopwords_udf_logic, ArrayType(StringType()))

result_df = result_df.withColumn(
    "keywords",
    remove_stopwords_udf(
        split(regexp_replace(lower(coalesce(col("title"), lit(""))), r"[^\w\s]", ""), r"\s+") # Đảm bảo title không NULL
    )
)

# Gỡ explode thành từng từ khóa riêng biệt
exploded_df = result_df.select(
    "category_name", "source_name", "keywords", "total_sentiment_score"
).withColumn("keyword", explode(col("keywords")))

# Loại bỏ từ ngắn và từ rỗng (keyword đã được kiểm tra trong UDF rồi, nhưng rlike thêm một lớp bảo vệ)
filtered_df = exploded_df.filter(col("keyword").rlike(r"^\w{4,}$") & (col("keyword") != ""))

# Tổng hợp từ khóa
final_df = filtered_df.groupBy("category_name", "source_name", "keyword") \
    .agg(
        count("*").alias("mention_count"),
        avg("total_sentiment_score").alias("avg_sentiment_value") # Đổi tên để tránh nhầm lẫn
    )

final_df = final_df.withColumn(
    "sentiment_label", # Đổi tên cột
    when(col("avg_sentiment_value") >= 0.05, "positive")
    .when(col("avg_sentiment_value") <= -0.05, "negative")
    .otherwise("neutral")
)

# Đẩy dữ liệu lên Elasticsearch sử dụng foreachPartition
def send_partition_to_elasticsearch(partition_iterator):
    # Khởi tạo ES client cho mỗi partition
    # Cần import lại các thư viện cần thiết bên trong hàm này vì nó chạy trên executor
    import os
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
    import json

    es_client = None
    try:
        es_client = Elasticsearch(
            "http://host.docker.internal:9200",
            timeout=30,  # Tăng timeout nếu cần
            max_retries=3,
            retry_on_timeout=True
        )
        if not es_client.ping():
            print(f"PID {os.getpid()}: Không thể ping Elasticsearch từ partition.")
            return # Hoặc raise Exception để task fail và được thử lại
    except Exception as e_conn:
        print(f"PID {os.getpid()}: Lỗi kết nối ES trong partition: {e_conn}")
        return

    actions = []
    for json_row_string in partition_iterator:
        try:
            doc = json.loads(json_row_string)
            actions.append({
                "_index": "reddit_sentiment", # Đảm bảo index này có mapping phù hợp
                "_source": doc
            })
        except json.JSONDecodeError:
            print(f"PID {os.getpid()}: Lỗi decode JSON: {json_row_string}")
            continue # Bỏ qua bản ghi lỗi

        # Gửi theo batch nhỏ để tránh request quá lớn
        if len(actions) >= 100: # Gửi mỗi 500 documents
            try:
                bulk(es_client, actions, raise_on_error=True)
                actions = []
            except Exception as e_bulk:
                print(f"PID {os.getpid()}: Lỗi khi bulk insert: {e_bulk}")
                # Có thể xử lý lỗi chi tiết hơn ở đây (ví dụ: ghi lại các doc lỗi)
                actions = [] # Xóa batch đã thử gửi lỗi

    if actions: # Gửi phần còn lại
        try:
            bulk(es_client, actions, raise_on_error=True)
        except Exception as e_bulk_final:
            print(f"PID {os.getpid()}: Lỗi khi bulk insert (final batch): {e_bulk_final}")


def send_to_elasticsearch_by_partition(df_to_send):
    print("Bắt đầu gửi dữ liệu lên Elasticsearch theo partition...")
    try:
        # df.toJSON() trả về một RDD[String]
        df_to_send.toJSON().foreachPartition(send_partition_to_elasticsearch)
        print(f"✅ Hoàn tất việc gửi dữ liệu lên Elasticsearch (hoặc đã cố gắng gửi). Kiểm tra log của executor để biết chi tiết.")
    except Exception as e:
        print(f"❌ Lỗi tổng thể khi cấu hình gửi dữ liệu lên Elasticsearch bằng foreachPartition: {str(e)}")

send_to_elasticsearch_by_partition(final_df)


# Ghi kết quả ra HDFS
try:
    final_df.write.mode("overwrite").parquet("hdfs://host.docker.internal:9000/outputs/processed_sentiments/")
    print("✅ Đã lưu kết quả sentiment + keywords vào HDFS: /outputs/processed_sentiments/")
except Exception as e:
    print(f"❌ Lỗi khi ghi dữ liệu ra HDFS: {str(e)}")

# Hiển thị dữ liệu (có thể comment lại nếu không cần thiết trong production)
contents_df.printSchema()
print("Contents DF Sample:")
contents_df.show(5, truncate=False)
print("Comments DF Sample:")
comments_df.show(5, truncate=False)
print("Final DF Sample:")
final_df.show(5, truncate=False)

print("Kiểm tra joined_df trước khi tính total_sentiment_score:")
joined_df.select("content_id", "score", "comment_count", "sentiment_score", "avg_comment_sentiment") \
         .orderBy(col("score").desc_nulls_last()) \
         .show(10, truncate=False)

joined_df.select("content_id", "score", "comment_count", "sentiment_score", "avg_comment_sentiment") \
         .orderBy(col("comment_count").desc_nulls_last()) \
         .show(10, truncate=False)

print("Thống kê của score và comment_count trong joined_df:")
joined_df.selectExpr("min(score) as min_score", "avg(score) as avg_score", "max(score) as max_score",
                     "min(comment_count) as min_cc", "avg(comment_count) as avg_cc", "max(comment_count) as max_cc") \
         .show()

print("Final DF Sample:")
final_df.show(5, truncate=False)

print("Kiểm tra result_df với total_sentiment_score:")
result_df.select("content_id", "score", "comment_count", "base_sentiment_score", "total_sentiment_score") \
         .orderBy(col("total_sentiment_score").desc_nulls_last()) \
         .show(20, truncate=False)

result_df.selectExpr("min(total_sentiment_score) as min_total_sent",
                     "avg(total_sentiment_score) as avg_total_sent",
                     "max(total_sentiment_score) as max_total_sent") \
         .show()

# Dừng Spark session
spark.stop()
print("Spark session stopped.")