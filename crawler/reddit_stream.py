from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, current_timestamp, when, length, date_format,
    explode, split, regexp_replace, avg, count, coalesce, lit,
    log1p, date_trunc, to_timestamp
)
from pyspark.sql.types import StructType, StringType, IntegerType, FloatType, ArrayType
from sparknlp.pretrained import PretrainedPipeline
import sparknlp
import nltk
from nltk.corpus import stopwords

# Tải NLTK stopwords
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))

# Khởi động Spark NLP và SparkSession
spark = SparkSession.builder \
    .master("spark://spark-master:7077") \
    .appName("RedditStreamingAnalysis") \
    .config("spark.streaming.stopGracefullyOnShutdown", "true") \
    .config("es.nodes", "elasticsearch") \
    .config("es.port", "9200") \
    .config("es.resource", "reddit_sentiment_ver4/_doc") \
    .config("es.nodes.wan.only", "true") \
    .config("spark.sql.streaming.checkpointLocation", "/opt/bitnami/spark/checkpoints/reddit_streaming") \
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.5.0") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Định nghĩa schema
schema = StructType() \
    .add("type", StringType()) \
    .add("data", StructType() \
        .add("content_id", StringType()) \
        .add("platform", StringType()) \
        .add("title", StringType()) \
        .add("content", StringType()) \
        .add("created_at", StringType()) \
        .add("source_id", StringType()) \
        .add("source_name", StringType()) \
        .add("category_id", StringType()) \
        .add("category_name", StringType()) \
        .add("tags", StringType()) \
        .add("views", IntegerType()) \
        .add("score", IntegerType()) \
        .add("comment_count", IntegerType()) \
        .add("duration", StringType()) \
        .add("upvote_ratio", FloatType()) \
        .add("url", StringType()) \
        .add("author", StringType()) \
        .add("crawl_time", StringType()) \
        .add("sentiment_score", FloatType()) \
        .add("keywords", ArrayType(StringType())) \
        .add("data_source", StringType()) \
        .add("comment_id", StringType())  
    )

# Đọc stream từ Kafka
df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "reddit_stream") \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

# Parse JSON và xử lý dữ liệu
df_parsed = df_raw.selectExpr("CAST(value AS STRING) as json") \
    .select(from_json(col("json"), schema).alias("data")) \
    .select("data.type", "data.data.*") \
    .withColumn("timestamp", date_format(current_timestamp(), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'")) \
    .withColumn("created_at", to_timestamp(col("created_at")))

# Lọc posts
posts_df = df_parsed.filter(col("type") == "post").select(
    col("content_id"),
    col("title"),
    col("content"),
    col("source_name"),
    col("category_name"),
    col("score").cast(IntegerType()),
    col("comment_count").cast(IntegerType()),
    col("created_at"),
    col("sentiment_score"),
    col("keywords"),
    col("data_source"),
    col("timestamp")
).withWatermark("created_at", "1 hour")  # Giới hạn trạng thái streaming

# Lọc comments
comments_df = df_parsed.filter(col("type") == "comment").select(
    col("content_id"),
    col("comment_id"),
    col("content"),
    col("source_name"),
    col("sentiment_score"),
    col("keywords"),
    col("data_source"),
    col("timestamp")
).withWatermark("timestamp", "1 hour")

# Tổng hợp bình luận theo content_id
comments_agg_df = comments_df.groupBy("content_id").agg(
    avg("sentiment_score").alias("avg_comment_sentiment"),
    count("comment_id").alias("total_comments")
)

# Kết hợp bài viết và bình luận
joined_df = posts_df.join(comments_agg_df, "content_id", "left_outer") \
    .withColumn("score", coalesce(col("score").cast("float"), lit(0.0))) \
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

# Tổng hợp từ khóa
result_df = result_df.withColumn(
    "keyword",
    explode(col("keywords"))
).filter(col("keyword").rlike(r"^\w{4,}$") & (col("keyword") != ""))

# Nhóm theo từ khóa
final_df = result_df.groupBy("content_id", "category_name", "source_name", "keyword", "data_source") \
    .agg(
        count("*").alias("mention_count"),
        avg("total_sentiment_score").alias("avg_sentiment_value")
    ).withColumn(
        "sentiment_label",
        when(col("avg_sentiment_value") >= 0.05, "positive")
        .when(col("avg_sentiment_value") <= -0.05, "negative")
        .otherwise("neutral")
    )

# Ghi vào Elasticsearch
query = final_df.writeStream \
    .outputMode("update") \
    .format("es") \
    .option("es.resource", "reddit_sentiment_ver4") \
    .option("es.write.operation", "upsert") \
    .option("es.mapping.id", "content_id") \
    .option("checkpointLocation", "/opt/bitnami/spark/checkpoints/reddit_streaming") \
    .start()

query.awaitTermination()