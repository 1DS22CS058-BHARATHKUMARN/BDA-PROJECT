from kafka import KafkaProducer
import pandas as pd
import time
import json
import math

print("📡 Connecting to Kafka...", flush=True)

# Load dataset
df = pd.read_csv("data/heart_disease_uci.csv")

# Define mappings
sex_map = {"Male": 1, "Female": 0}
cp_map = {
    "typical angina": 0,
    "atypical angina": 1,
    "non-anginal pain": 2,
    "asymptomatic": 3
}
restecg_map = {"normal": 0, "ST-T wave abnormality": 1, "left ventricular hypertrophy": 2}
slope_map = {"upsloping": 0, "flat": 1, "downsloping": 2}
thal_map = {"normal": 1, "fixed defect": 2, "reversible defect": 3}

# Initialize Kafka Producer
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Function to safely convert values
def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default

def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default

# Send each row as a message
for _, row in df.iterrows():
    message = {
        "age": safe_int(row["age"]),
        "sex": sex_map.get(row["sex"], -1),
        "cp": cp_map.get(row["cp"], -1),
        "trestbps": safe_int(row["trestbps"]),
        "chol": safe_int(row["chol"]),
        "fbs": int(row["fbs"]) if not pd.isna(row["fbs"]) else 0,
        "restecg": restecg_map.get(row["restecg"], -1),
        "thalach": safe_int(row["thalch"]),
        "exang": int(row["exang"]) if not pd.isna(row["exang"]) else 0,
        "oldpeak": safe_float(row["oldpeak"]),
        "slope": slope_map.get(row["slope"], -1),
        "ca": safe_int(row["ca"]),
        "thal": thal_map.get(row["thal"], -1),
        "target": safe_int(row["num"])
    }

    producer.send('HeartData', value=message)
    print(f"Sent: {message}")
    time.sleep(1)  # simulate streaming

producer.close()
