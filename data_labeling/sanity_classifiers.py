import polars as pl
from datetime import date
import altair as alt
from rich.progress import track
from pathlib import Path
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
# from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.colors as pColors
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from time import time

from common import (
    find_data_all_lazy,
    data_path,
    centrally_smoothed_path,
    windows,
    trend_windows,
    intermediate_path,
)
# check_path = centrally_smoothed_path / "AMN.parquet"


# find the eligible tickers
all_tickers = list(centrally_smoothed_path.glob("*.parquet"))
# subset = all_tickers[list(range(0,len(all_tickers), 100))]

# print(my_tckrs)
# for i in track(range(0,len(all_tickers), 1000), description="Plotting result..."):
#     check_path = Path(all_tickers[i])
sum_q = []
time_start = time()
for check_path in track(all_tickers, description="Collecting result..."):
    q = pl.scan_parquet(check_path).filter(pl.col("date") > date(year=2023, month=1, day=1))
    q = q.group_by("savgol_p2_11_rel_diff_5_trend").len()

    sum_q.append(q.collect())

# sum_q = pl.concat(sum_q, how="vertical").group_by("savgol_p2_11_rel_diff_5_trend").sum()
sum_q = pl.concat(sum_q, how="vertical").group_by("savgol_p2_11_rel_diff_5_trend").sum()

# sum_q.collect(streaming=True).write_parquet(data_path / "trend_counts.parquet")
sum_q.write_parquet(data_path / "trend_counts.parquet")
# print(sum_q.collect())
print(sum_q)
time_end = time()
print(f"Time taken: {time_end - time_start} seconds")

# Sample with different ratios (x and y) based on the label value
x_ratio = 0.3  # Example ratio for labels 0 and 1
y_ratio = 0.7  # Example ratio for other label values

# First approach: Split and recombine
df_label_0_1 = q.filter((pl.col("label") == 0) | (pl.col("label") == 1))
df_other_labels = q.filter(~((pl.col("label") == 0) | (pl.col("label") == 1)))

# Sample each part with different ratios
sampled_0_1 = df_label_0_1.sample(fraction=x_ratio, seed=42)
sampled_other = df_other_labels.sample(fraction=y_ratio, seed=42)

# Combine the sampled parts
undersampled_df = pl.concat([sampled_0_1, sampled_other], how="vertical")

# Apply operations and collect with streaming
result_df = undersampled_df.collect(streaming=True).write_parquet(data_path / "undersampled_data.parquet")


# subset of all data
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.model_selection import train_test_split

# Example dataset (replace with your actual data)
from sklearn.datasets import make_classification
X, y = make_classification(n_classes=6, weights=[0.6, 0.1, 0.1, 0.1, 0.05, 0.05], n_samples=100000, random_state=42)

# Parameters for random sampling
n_iterations = 5  # Number of random samples
sample_size = 10000  # Size of each sample
results = []  # To store performance metrics

for i in range(n_iterations):
    # Random sampling
    indices = np.random.choice(len(X), size=sample_size, replace=False)
    X_sample, y_sample = X[indices], y[indices]
    
    # Train-test split for the sample
    X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.3, random_state=i)
    
    # Train the model
    rf = RandomForestClassifier(random_state=i)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    y_probs = rf.predict_proba(X_test)
    
    # Evaluate metrics
    accuracy = accuracy_score(y_test, y_pred)
    ap_score = average_precision_score(y_test, y_probs, average='weighted')
    results.append({'iteration': i, 'accuracy': accuracy, 'average_precision': ap_score})

# Aggregate results
avg_accuracy = np.mean([r['accuracy'] for r in results])
avg_ap_score = np.mean([r['average_precision'] for r in results])

print(f"Average Accuracy: {avg_accuracy:.2f}")
print(f"Average Precision Score: {avg_ap_score:.2f}")

## all data set
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# Example dataset (replace with your data)
from sklearn.datasets import make_classification
X, y = make_classification(n_classes=6, weights=[0.6, 0.1, 0.1, 0.1, 0.05, 0.05], n_informative=10, n_samples=1000, random_state=42)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Make predictions
y_pred = rf.predict(X_test)

# Evaluation metrics
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optional: Check specific metrics like F1-score or balanced accuracy
f1 = f1_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)
print(f"\nF1-Score (weighted): {f1:.2f}")
print(f"Accuracy: {accuracy:.2f}")

# different metrics
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

# Get probabilities for each class (needed for AUC and AP)
y_probs = rf.predict_proba(X_test)

# For multiclass AUC, use the 'ovr' (one-vs-rest) approach
auc_score = roc_auc_score(y_test, y_probs, multi_class='ovr')
print(f"Multiclass AUC Score: {auc_score:.2f}")

# Average Precision Score (binary or multiclass)
ap_score = average_precision_score(y_test, y_probs, average='weighted')
print(f"Weighted Average Precision Score: {ap_score:.2f}")