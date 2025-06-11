from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, current_timestamp, when, length, date_format
from pyspark.sql.types import StructType, StringType, IntegerType, FloatType
from sparknlp.pretrained import PretrainedPipeline
import sparknlp

# Khởi động Spark NLP và SparkSession
import sparknlp

# Khởi động Spark NLP và SparkSession với cấu hình tùy chỉnh
spark = SparkSession.builder \
    .master("spark://spark-master:7077") \
    .appName("RedditStreamingAnalysis") \
    .config("spark.streaming.stopGracefullyOnShutdown", "true") \
    .config("es.nodes", "elasticsearch") \
    .config("es.port", "9200") \
    .config("es.resource", "reddit-stream/_doc") \
    .config("es.nodes.wan.only", "true") \
    .config("spark.sql.streaming.checkpointLocation", "/opt/bitnami/spark/checkpoints/reddit_streaming") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Định nghĩa schema giống batch (cho contents)
schema = StructType() \
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
    .add("crawl_time", StringType())

# Tải pipeline phân tích cảm xúc từ Spark NLP
sentiment_pipeline = PretrainedPipeline("analyze_sentiment", lang="en")

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
    .select("data.*") \
    .withColumn("timestamp", date_format(current_timestamp(), "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'"))
# Tạo cột text cho pipeline NLP (ưu tiên content, fallback sang title)
df_parsed = df_parsed.withColumn(
    "text",
    when(length(col("content")) > 0, col("content")).otherwise(col("title"))
)

# Phân tích cảm xúc với Spark NLP
df_annotated = sentiment_pipeline.transform(df_parsed) \
    .select(
        col("content_id"),
        col("title"),
        col("content"),
        col("source_name"),
        col("category_name"),
        col("score"),
        col("timestamp"),
        col("sentiment.result")[0].alias("sentiment")
    )

# Ghi vào Elasticsearch
query = df_annotated.writeStream \
    .outputMode("append") \
    .format("es") \
    .option("es.resource", "reddit-stream") \
    .option("checkpointLocation", "/opt/bitnami/spark/checkpoints/reddit_streaming") \
    .start()

query.awaitTermination()