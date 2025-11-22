import pandas as pd
import numpy as np

df = pd.read_csv('./Dataset/session_based_dataset.csv')

print("Dataset Info:")
print(f"Total samples: {len(df)}")
print(f"\nLabel distribution:")
print(df['label'].value_counts())
print(f"Anomaly %: {(df['label'].sum()/len(df))*100:.2f}%")

# Check if there's actually a difference between normal and anomaly
print("\n" + "="*70)
print("Feature Statistics Comparison")
print("="*70)

# Get numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col not in ['label', 'unique_link_mark']]

# Compare means
for col in numeric_cols[:10]:  # First 10 features
    normal_mean = df[df['label']==0][col].mean()
    anomaly_mean = df[df['label']==1][col].mean()
    diff_pct = abs(normal_mean - anomaly_mean) / (normal_mean + 1e-10) * 100
    print(f"{col:30s} | Normal: {normal_mean:10.2f} | Anomaly: {anomaly_mean:10.2f} | Diff: {diff_pct:6.1f}%")