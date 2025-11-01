from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegressionModel

# ✅ 1. Initialize Spark Session
spark = SparkSession.builder \
    .appName("HeartDiseaseModelPrediction") \
    .getOrCreate()

# ✅ 2. Define data and model paths
data_path = "hdfs://localhost:9000/heart_data/"
model_path = "hdfs://localhost:9000/models/heart_lr_model/"

# ✅ 3. Define feature columns (same as in train_model.py)
feature_cols = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]
target_col = "target"  # Not required for prediction but helpful for testing

# ✅ 4. Read CSV data (new/unseen heart data)
df = spark.read.csv(data_path, header=True, inferSchema=True)

# ✅ 5. Assemble features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
data = assembler.transform(df)

# ✅ 6. Load the trained model
model = LogisticRegressionModel.load(model_path)

# ✅ 7. Generate predictions
predictions = model.transform(data)

# ✅ 8. Show a few results
predictions.select("features", "prediction", "probability").show(10, truncate=False)

# ✅ 9. (Optional) Save predictions back to HDFS
predictions.select("prediction").write.mode("overwrite").csv("hdfs://localhost:9000/predictions/heart_predictions")

spark.stop()
