# query.awaitTermination()
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, current_timestamp, when, length,
    avg, coalesce, log1p, lit, udf, count, to_timestamp, date_trunc, max, unix_timestamp, abs
)
from pyspark.sql.types import StructType, StringType, IntegerType, FloatType
import sparknlp
from sparknlp.pretrained import PretrainedPipeline

# Khởi động SparkSession
spark = SparkSession.builder \
    .master("spark://spark-master:7077") \
    .appName("RedditStreamingAnalysis") \
    .config("spark.streaming.stopGracefullyOnShutdown", "true") \
    .config("es.nodes", "elasticsearch") \
    .config("es.port", "9200") \
    .config("es.resource", "reddit-stream-ver3/_doc") \
    .config("es.nodes.data.only", "true") \
    .config("spark.sql.streaming.checkpointLocation", "/opt/bitnami/spark/checkpoints/reddit_streaming") \
    .config("spark.sql.debug.maxToStringFields", "1000") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Định nghĩa schema
content_schema = StructType() \
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

comment_schema = StructType() \
    .add("comment_id", StringType()) \
    .add("content_id", StringType()) \
    .add("platform", StringType()) \
    .add("content", StringType()) \
    .add("created_at", StringType()) \
    .add("score", IntegerType()) \
    .add("author", StringType()) \
    .add("source_name", StringType()) \
    .add("crawl_time", StringType())

type_schema = StructType().add("type", StringType())

# Tải pipeline NLP
sentiment_pipeline = PretrainedPipeline("analyze_sentiment", lang="en")

# Đọc stream từ Kafka
df_raw = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "reddit_stream") \
    .option("startingOffsets", "latest") \
    .option("failOnDataLoss", "false") \
    .load()

# Parse JSON và phân biệt content/comment
df_raw = df_raw.selectExpr("CAST(value AS STRING) as json")
df_with_type = df_raw.withColumn("parsed", from_json(col("json"), type_schema))

# Lọc content
df_content = df_with_type.filter(col("parsed.type") == "content") \
    .select(from_json(col("json"), content_schema).alias("data")) \
    .select("data.*") \
    .withColumn("type", lit("content")) \
    .withColumn("timestamp", current_timestamp()) \
    .withColumn("created_at", to_timestamp(col("created_at"))) \
    .filter(col("created_at").isNotNull())

# Lọc comment
df_comment = df_with_type.filter(col("parsed.type") == "comment") \
    .select(from_json(col("json"), comment_schema).alias("data")) \
    .select("data.*") \
    .withColumn("type", lit("comment")) \
    .withColumn("timestamp", current_timestamp()) \
    .withColumn("created_at", to_timestamp(col("created_at"))) \
    .filter(col("created_at").isNotNull())

# Tạo cột text cho NLP
df_content = df_content.withColumn(
    "text",
    when(length(col("content")) > 0, col("content")).otherwise(col("title"))
)
df_comment = df_comment.withColumn("text", col("content"))

# Phân tích cảm xúc
df_content_annotated = sentiment_pipeline.transform(df_content) \
    .select(
        col("content_id"),
        col("title"),
        col("content"),
        col("source_name"),
        col("category_name"),
        col("score"),
        col("comment_count"),
        col("timestamp"),
        col("created_at"),
        col("sentiment.result").alias("sentiment")
    )

df_comment_annotated = sentiment_pipeline.transform(df_comment) \
    .select(
        col("content_id"),
        col("comment_id"),
        col("content"),
        col("source_name"),
        col("timestamp"),
        col("created_at"),
        col("sentiment.result").alias("sentiment")
    )

# Chuyển đổi nhãn cảm xúc thành điểm số
def label_to_score(label):
    if label == "positive":
        return 1.0
    elif label == "negative":
        return -1.0
    return 0.0

label_to_score_udf = udf(label_to_score, FloatType())

df_content_annotated = df_content_annotated.withColumn(
    "sentiment_score",
    label_to_score_udf(col("sentiment")[0])
)
df_comment_annotated = df_comment_annotated.withColumn(
    "sentiment_score",
    label_to_score_udf(col("sentiment")[0])
)

# Thêm watermark
df_content_annotated = df_content_annotated.withWatermark("created_at", "10 minutes")
df_comment_annotated = df_comment_annotated.withWatermark("created_at", "10 minutes")

# Tổng hợp cảm xúc bình luận
df_comment_agg = df_comment_annotated.groupBy(
    col("content_id")
).agg(
    avg("sentiment_score").alias("avg_comment_sentiment"),
    count("comment_id").alias("total_comments"),
    max("created_at").alias("max_created_at")
).withWatermark("max_created_at", "10 minutes")

# Join với inner join
df_joined = df_content_annotated.join(
    df_comment_agg,
    (df_content_annotated.content_id == df_comment_agg.content_id) & 
    (abs(unix_timestamp(df_content_annotated.created_at) - unix_timestamp(df_comment_agg.max_created_at)) <= 600000),
    "inner"
).withColumn("score", coalesce(col("score").cast("float"), lit(0.0))) \
 .withColumn("comment_count", coalesce(col("comment_count").cast("float"), lit(0.0)))

# Thêm cột date và tính toán sentiment score
df_joined = df_joined.withColumn("date", date_trunc("day", col("created_at")))
df_joined = df_joined.withColumn(
    "base_sentiment_score",
    (coalesce(col("sentiment_score"), lit(0.0)) * 0.6 + 
     coalesce(col("avg_comment_sentiment"), lit(0.0)) * 0.4)
)
df_joined = df_joined.withColumn(
    "total_sentiment_score",
    col("base_sentiment_score") * (lit(1.0) + (log1p(col("score")) + log1p(col("comment_count"))) * lit(0.1))
)
df_joined = df_joined.withColumn(
    "sentiment_label",
    when(col("total_sentiment_score") >= 0.05, "positive")
    .when(col("total_sentiment_score") <= -0.05, "negative")
    .otherwise("neutral")
)

df_joined = df_joined.withWatermark("created_at", "10 minutes")

# Ghi vào Elasticsearch
query = df_joined.writeStream \
    .outputMode("update") \
    .format("es") \
    .option("es.resource", "reddit-stream-ver3") \
    .option("checkpointLocation", "/opt/bitnami/spark/checkpoints/reddit_streaming") \
    .option("es.mapping.timestamp", "created_at") \
    .option("es.mapping.id", "content_id")\
    .start()

query.awaitTermination()