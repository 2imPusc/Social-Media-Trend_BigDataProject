from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, lower, explode, split, regexp_replace, avg, count,
    coalesce, to_timestamp, when, udf, lit, trim, length, log1p, date_trunc
)
from pyspark.sql.types import IntegerType, FloatType, ArrayType, StringType
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
import nltk
import sparknlp
from sparknlp.pretrained import PretrainedPipeline

# Tải NLTK stopwords nếu cần thiết và tạo set để broadcast
try:
    from nltk.corpus import stopwords
    nltk_stop_words_list = stopwords.words('english')
except LookupError:
    print("Tài nguyên 'stopwords' không tìm thấy trên driver. Đang tải xuống...")
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    nltk_stop_words_list = stopwords.words('english')

stop_words_set_for_broadcast = set(nltk_stop_words_list)

# Khởi tạo Spark session với Spark NLP
spark = SparkSession.builder \
    .appName("RedditTrendAnalysis") \
    .config("spark.driver.memory", "4g") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.0") \
    .getOrCreate()

# Broadcast stop_words
broadcast_stop_words = spark.sparkContext.broadcast(stop_words_set_for_broadcast)

# Tải pipeline Spark NLP
sentiment_pipeline = PretrainedPipeline("analyze_sentiment", lang="en")

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

# Phân tích cảm xúc với Spark NLP
# Tạo cột text cho pipeline NLP (ưu tiên content, fallback sang title)
contents_df = contents_df.withColumn(
    "text",
    when(length(col("content")) > 0, col("content")).otherwise(col("title"))
)
# Đối với comments_df: sử dụng cột content
comments_df = comments_df.withColumn(
    "text",
    col("content")
)

# Áp dụng pipeline Spark NLP
contents_df = sentiment_pipeline.transform(contents_df) \
    .withColumn("sentiment_score", col("sentiment.result")[0].cast("string"))

comments_df = sentiment_pipeline.transform(comments_df) \
    .withColumn("sentiment_score", col("sentiment.result")[0].cast("string"))

# Chuyển đổi nhãn cảm xúc thành điểm số
def label_to_score(label):
    if label == "positive":
        return 1.0
    elif label == "negative":
        return -1.0
    return 0.0

label_to_score_udf = udf(label_to_score, FloatType())

contents_df = contents_df.withColumn("sentiment_score", label_to_score_udf(col("sentiment_score")))
comments_df = comments_df.withColumn("sentiment_score", label_to_score_udf(col("sentiment_score")))

# Kết hợp bình luận với nội dung
joined_df = comments_df.groupBy("content_id").agg(
    avg("sentiment_score").alias("avg_comment_sentiment"),
    count("comment_id").alias("total_comments")
).join(contents_df, on="content_id", how="inner")

# Điền giá trị mặc định cho score và comment_count
joined_df = joined_df.withColumn("score", coalesce(col("score").cast("float"), lit(0.0))) \
                    .withColumn("comment_count", coalesce(col("comment_count").cast("float"), lit(0.0)))

# Tính base_sentiment_score
result_df_intermediate = joined_df.withColumn(
    "base_sentiment_score",
    (coalesce(col("sentiment_score"), lit(0.0)) * 0.6 + coalesce(col("avg_comment_sentiment"), lit(0.0)) * 0.4)
)

# Tính interaction_factor
interaction_factor_expr = (log1p(col("score")) + log1p(col("comment_count")))

result_df = result_df_intermediate.withColumn(
    "total_sentiment_score",
    col("base_sentiment_score") * (lit(1.0) + interaction_factor_expr * lit(0.1))
)

# UDF loại bỏ stopwords
def remove_stopwords_udf_logic(words_list):
    if not words_list:
        return []
    current_stop_words = broadcast_stop_words.value
    return [word for word in words_list if word and word.lower() not in current_stop_words]

remove_stopwords_udf = udf(remove_stopwords_udf_logic, ArrayType(StringType()))

result_df = result_df.withColumn(
    "keywords",
    remove_stopwords_udf(
        split(regexp_replace(lower(coalesce(col("title"), lit(""))), r"[^\w\s]", ""), r"\s+")
    )
)

# Gỡ explode thành từng từ khóa riêng biệt
exploded_df = result_df.select(
    "category_name", "source_name", "keywords", "total_sentiment_score", "created_at"
).withColumn("keyword", explode(col("keywords")))

# Loại bỏ từ ngắn và từ rỗng
filtered_df = exploded_df.filter(col("keyword").rlike(r"^\w{4,}$") & (col("keyword") != ""))

# Tổng hợp từ khóa

filtered_df = filtered_df.withColumn("date", date_trunc("day", col("created_at")))

final_df = filtered_df.groupBy("category_name", "source_name", "keyword", "date") \
    .agg(
        count("*").alias("mention_count"),
        avg("total_sentiment_score").alias("avg_sentiment_value")
    )

final_df = final_df.withColumn(
    "sentiment_label",
    when(col("avg_sentiment_value") >= 0.05, "positive")
    .when(col("avg_sentiment_value") <= -0.05, "negative")
    .otherwise("neutral")
)

# Đẩy dữ liệu lên Elasticsearch
def send_partition_to_elasticsearch(partition_iterator):
    import os
    from elasticsearch import Elasticsearch
    from elasticsearch.helpers import bulk
    import json6

    es_client = None
    try:
        es_client = Elasticsearch(
            "http://host.docker.internal:9200",
            request_timeout=30,
            max_retries=3,
            retry_on_timeout=True
        )
        if not es_client.ping():
            print(f"PID {os.getpid()}: Không thể ping Elasticsearch từ partition.")
            return
    except Exception as e_conn:
        print(f"PID {os.getpid()}: Lỗi kết nối ES trong partition: {e_conn}")
        return

    actions = []
    for json_row_string in partition_iterator:
        try:
            doc = json.loads(json_row_string)
            actions.append({
                "_index": "reddit_sentiment_ver4",
                "_source": doc
            })
        except json.JSONDecodeError:
            print(f"PID {os.getpid()}: Lỗi decode JSON: {json_row_string}")
            continue

        if len(actions) >= 100:
            try:
                bulk(es_client, actions, raise_on_error=True)
                actions = []
            except Exception as e_bulk:
                print(f"PID {os.getpid()}: Lỗi khi bulk insert: {e_bulk}")
                actions = []

    if actions:
        try:
            bulk(es_client, actions, raise_on_error=True)
        except Exception as e_bulk_final:
            print(f"PID {os.getpid()}: Lỗi khi bulk insert (final batch): {e_bulk_final}")

def send_to_elasticsearch_by_partition(df_to_send):
    print("Bắt đầu gửi dữ liệu lên Elasticsearch theo partition...")
    try:
        df_to_send.toJSON().foreachPartition(send_partition_to_elasticsearch)
        print(f"✅ Hoàn tất việc gửi dữ liệu lên Elasticsearch.")
    except Exception as e:
        print(f"❌ Lỗi khi gửi dữ liệu lên Elasticsearch: {str(e)}")

send_to_elasticsearch_by_partition(final_df)

# Ghi kết quả ra HDFS
try:
    final_df.write.mode("overwrite").parquet("hdfs://host.docker.internal:9000/outputs/processed_sentiments/")
    print("✅ Đã lưu kết quả sentiment + keywords vào HDFS: /outputs/processed_sentiments/")
except Exception as e:
    print(f"❌ Lỗi khi ghi dữ liệu ra HDFS: {str(e)}")

# Hiển thị dữ liệu (có thể comment lại nếu không cần thiết)
contents_df.printSchema()
comments_df.printSchema()
# print("Contents DF Sample:")
# contents_df.show(5, truncate=False)
# print("Comments DF Sample:")
# comments_df.show(5, truncate=False)
# print("Final DF Sample:")
# final_df.show(5, truncate=False)

# print("Kiểm tra joined_df trước khi tính total_sentiment_score:")
# joined_df.select("content_id", "score", "comment_count", "sentiment_score", "avg_comment_sentiment") \
#          .orderBy(col("score").desc_nulls_last()) \
#          .show(10, truncate=False)

# joined_df.select("content_id", "score", "comment_count", "sentiment_score", "avg_comment_sentiment") \
#          .orderBy(col("comment_count").desc_nulls_last()) \
#          .show(10, truncate=False)

# print("Thống kê của score và comment_count trong joined_df:")
# joined_df.selectExpr("min(score) as min_score", "avg(score) as avg_score", "max(score) as max_score",
#                      "min(comment_count) as min_cc", "avg(comment_count) as avg_cc", "max(comment_count) as max_cc") \
#          .show()

# print("Kiểm tra result_df với total_sentiment_score:")
# result_df.select("content_id", "score", "comment_count", "base_sentiment_score", "total_sentiment_score") \
#          .orderBy(col("total_sentiment_score").desc_nulls_last()) \
#          .show(20, truncate=False)

# result_df.selectExpr("min(total_sentiment_score) as min_total_sent",
#                      "avg(total_sentiment_score) as avg_total_sent",
#                      "max(total_sentiment_score) as max_total_sent") \
#          .show()

# Dừng Spark session
spark.stop()
print("Spark session stopped.")