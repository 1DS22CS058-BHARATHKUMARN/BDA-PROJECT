from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import joblib
import numpy as np

# Initialize Spark session
spark = SparkSession.builder \
    .appName("HeartDiseaseModelTraining") \
    .getOrCreate()

# Load data from HDFS
df = spark.read.csv("hdfs://localhost:9000/heart_data/", header=True, inferSchema=True)

# Show sample
df.show(5)

# Feature columns
feature_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
                'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Target column
target_col = 'target'

# Assemble features into vector
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
final_df = assembler.transform(df).select("features", target_col)

# Split into training and test sets
train_df, test_df = final_df.randomSplit([0.8, 0.2], seed=42)

# Train model
lr = LogisticRegression(labelCol=target_col, featuresCol="features")
model = lr.fit(train_df)

# Evaluate accuracy
predictions = model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(
    labelCol=target_col, predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# ✅ Save Spark model to HDFS
model.write().overwrite().save("hdfs://localhost:9000/models/heart_lr_model")
print("✅ Spark ML model saved to HDFS at /models/heart_lr_model")

# ✅ Also export local joblib model (for consumer_predict.py)
# Collect small dataset to pandas for sklearn-compatible training
pandas_df = df.toPandas()

from sklearn.linear_model import LogisticRegression as SklearnLR
X = pandas_df[feature_cols].values
y = pandas_df[target_col].values
sk_model = SklearnLR(max_iter=1000)
sk_model.fit(X, y)

# Save locally
joblib.dump(sk_model, "heart_model.pkl")
print("✅ Joblib model saved locally as heart_model.pkl")

# Stop Spark
spark.stop()
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")
print("✅ Spark ML model saved to HDFS at /models/heart_lr_model")
print("✅ Joblib model saved locally as heart_model.pkl")