import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------
# 1. Load predictions data
# --------------------------------------------
df = pd.read_csv("predictions_final.csv", header=0)

# If the file has no header, rename manually
if 'prediction' not in df.columns:
    df.columns = ['prediction']

# --------------------------------------------
# 2. Prepare summary statistics
# --------------------------------------------
counts = df['prediction'].value_counts().sort_index()
total = len(df)

print("Prediction Summary:")
for label, count in counts.items():
    print(f"  Class {label}: {count} ({count/total:.2%})")

# --------------------------------------------
# 3. Visualization (Enhanced)
# --------------------------------------------
sns.set(style="whitegrid")

plt.figure(figsize=(10, 6))
bar = sns.barplot(
    x=counts.index,
    y=counts.values,
    palette="viridis",
    edgecolor='black'
)

# Add data labels on bars
for i, v in enumerate(counts.values):
    plt.text(i, v + 0.5, f"{v}\n({v/total:.1%})", 
             ha='center', fontweight='bold', color='black', fontsize=11)

# Titles and labels
plt.title("Heart Disease Prediction Distribution", fontsize=16, weight='bold', pad=20)
plt.xlabel("Prediction Category (0 = No Disease, 1–3 = Severity Levels)", fontsize=12)
plt.ylabel("Number of Predictions", fontsize=12)

plt.tight_layout()
plt.show()
