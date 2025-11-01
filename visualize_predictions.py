import pandas as pd
import matplotlib.pyplot as plt

# Read predictions file
df = pd.read_csv("predictions_final.csv", header=0)

# If your file doesn't have column headers, rename manually
if 'prediction' not in df.columns:
    df.columns = ['prediction']

# Count frequency of each prediction
counts = df['prediction'].value_counts().sort_index()

# Plot bar chart
plt.figure(figsize=(8, 5))
counts.plot(kind='bar', color='skyblue', edgecolor='black')

plt.title("Heart Disease Prediction Distribution")
plt.xlabel("Prediction Category")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()
