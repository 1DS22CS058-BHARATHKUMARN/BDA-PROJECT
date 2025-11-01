from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf
from pyspark.sql.types import StructType, StructField, IntegerType, DoubleType
import joblib
import numpy as np

from pyspark.sql import SparkSession
import logging
import os

# Silence Kafka and Spark warnings
os.environ["PYSPARK_SUBMIT_ARGS"] = "--conf spark.ui.showConsoleProgress=false pyspark-shell"

# Reduce Spark logging level
spark = SparkSession.builder \
    .appName("HeartDiseaseRealTimePrediction") \
    .config("spark.sql.streaming.schemaInference", "true") \
    .getOrCreate()

# Set log level to ERROR instead of WARN
spark.sparkContext.setLogLevel("ERROR")

# Also silence Py4J gateway logs (Java backend)
logger = logging.getLogger('py4j')
logger.setLevel(logging.ERROR)


# --------------------------------------------
# 2. Define schema (must match producer JSON)
# --------------------------------------------
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
    StructField("thal", IntegerType(), True)
])

# --------------------------------------------
# 3. Read streaming data from Kafka
# --------------------------------------------
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "HeartData") \
    .option("startingOffsets", "latest") \
    .load()

# Convert Kafka value to JSON
json_df = kafka_df.selectExpr("CAST(value AS STRING)")

# Parse JSON according to schema
parsed_df = json_df.select(from_json(col("value"), schema).alias("data")).select("data.*")

# --------------------------------------------
# 4. Load trained model (scikit-learn)
# --------------------------------------------
model = joblib.load("heart_model.pkl")

# --------------------------------------------
# 5. Define UDF for prediction
# --------------------------------------------
def predict_heart(*cols):
    features = np.array(cols).reshape(1, -1)
    pred = model.predict(features)
    return int(pred[0])

predict_udf = udf(predict_heart, IntegerType())

# --------------------------------------------
# 6. Apply UDF to create 'prediction' column
# --------------------------------------------
predicted_df = parsed_df.withColumn(
    "prediction",
    predict_udf(
        col("age"), col("sex"), col("cp"), col("trestbps"),
        col("chol"), col("fbs"), col("restecg"), col("thalach"),
        col("exang"), col("oldpeak"), col("slope"), col("ca"), col("thal")
    )
)

# --------------------------------------------
# 7A. Print predictions to console (live view)
# --------------------------------------------
console_query = predicted_df.select("prediction").writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .start()
# def print_predictions(batch_df, batch_id):
#     preds = batch_df.select("prediction").collect()
#     for row in preds:
#         print(f"Prediction: {row['prediction']}")

# predicted_df.writeStream \
#     .foreachBatch(print_predictions) \
#     .start() \
#     .awaitTermination()

# --------------------------------------------
# 7B. Write predictions to a single local CSV file
# --------------------------------------------
def write_to_single_csv(batch_df, batch_id):
    pdf = batch_df.select("prediction").toPandas()
    pdf.to_csv("predictions_final.csv", mode='a', index=False, header=False)

file_query = predicted_df.writeStream \
    .outputMode("append") \
    .foreachBatch(write_to_single_csv) \
    .option("checkpointLocation", "checkpoints_predictions/") \
    .start()

console_query.awaitTermination()
file_query.awaitTermination()
