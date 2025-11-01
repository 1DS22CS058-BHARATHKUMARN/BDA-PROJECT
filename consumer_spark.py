from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
print("🔥 Spark consumer starting...", flush=True)


# # Initialize Spark session
# spark = SparkSession.builder \
#     .appName("HeartDiseaseConsumer") \
#     .getOrCreate()
spark = SparkSession.builder \
    .appName("HeartDiseaseConsumer") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1") \
    .getOrCreate()
# Define schema matching producer.py
schema = StructType([
    StructField("age", IntegerType(), True),
    StructField("sex", IntegerType(), True),
    StructField("cp", IntegerType(), True),
    StructField("trestbps", IntegerType(), True),
    StructField("chol", IntegerType(), True),
    StructField("fbs", IntegerType(), True),
    StructField("restecg", IntegerType(), True),
    StructField("thalach", IntegerType(), True),
    StructField("exang", IntegerType(), True),
    StructField("oldpeak", DoubleType(), True),
    StructField("slope", IntegerType(), True),
    StructField("ca", IntegerType(), True),
    StructField("thal", IntegerType(), True),
    StructField("target", IntegerType(), True)
])

# Read streaming data from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "HeartData") \
    .option("startingOffsets", "earliest") \
    .load()

# Extract JSON from Kafka 'value' column
parsed_df = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# # Show the data on console (streaming)
# query = parsed_df.writeStream \
#     .outputMode("append") \
#     .format("console") \
#     .option("truncate", "false") \
#     .start()

# Write the streaming data to HDFS (CSV format)
query = parsed_df.writeStream \
    .outputMode("append") \
    .format("csv") \
    .option("header", "true") \
    .option("path", "hdfs://localhost:9000/heart_data/") \
    .option("checkpointLocation", "hdfs://localhost:9000/heart_checkpoint/") \
    .start()

query.awaitTermination()
