import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

breast_cancer = datasets.load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target
feature_names = breast_cancer.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

df_train = pd.DataFrame(X_train, columns=feature_names)
df_train['target'] = y_train

dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train, y_train)

feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 20 Most Important Features:")
print(feature_importance.head(20).to_string(index=False))
print("\n" + "="*80 + "\n")

print("Feature Statistics (Train Set):")
print("="*80)
for idx in range(min(20, len(feature_names))):
    feat_name = feature_importance.iloc[idx]['feature']
    feat_idx = list(feature_names).index(feat_name)

    benign_vals = X_train[y_train == 1, feat_idx]
    malignant_vals = X_train[y_train == 0, feat_idx]

    print(f"\n{feat_name}:")
    print(f"  Benign    - Mean: {benign_vals.mean():.4f}, Median: {np.median(benign_vals):.4f}, Q1: {np.percentile(benign_vals, 25):.4f}, Q3: {np.percentile(benign_vals, 75):.4f}")
    print(f"  Malignant - Mean: {malignant_vals.mean():.4f}, Median: {np.median(malignant_vals):.4f}, Q1: {np.percentile(malignant_vals, 25):.4f}, Q3: {np.percentile(malignant_vals, 75):.4f}")

    threshold_median = (np.median(benign_vals) + np.median(malignant_vals)) / 2
    threshold_mean = (benign_vals.mean() + malignant_vals.mean()) / 2
    threshold_q3_q1 = (np.percentile(benign_vals, 75) + np.percentile(malignant_vals, 25)) / 2

    print(f"  Threshold (median): {threshold_median:.4f}")
    print(f"  Threshold (mean):   {threshold_mean:.4f}")
    print(f"  Threshold (Q3-Q1):  {threshold_q3_q1:.4f}")

    separation = abs(np.median(malignant_vals) - np.median(benign_vals)) / (benign_vals.std() + malignant_vals.std())
    print(f"  Separation score:   {separation:.4f}")

print("\n" + "="*80)
print("Decision Tree Accuracy:", dt.score(X_test, y_test))
print("="*80 + "\n")

print("\nOptimal Single-Feature Thresholds (Test Accuracy):")
print("="*80)
results = []
for feat_idx, feat_name in enumerate(feature_names):
    best_acc = 0
    best_threshold = 0
    best_direction = ">"

    thresholds = np.percentile(X_train[:, feat_idx], np.linspace(10, 90, 20))

    for threshold in thresholds:
        for direction in [">", "<"]:
            if direction == ">":
                y_pred = (X_test[:, feat_idx] > threshold).astype(int)
            else:
                y_pred = (X_test[:, feat_idx] < threshold).astype(int)

            acc = accuracy_score(y_test, y_pred)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
                best_direction = direction

    results.append({
        'feature': feat_name,
        'threshold': best_threshold,
        'direction': best_direction,
        'accuracy': best_acc
    })

results_df = pd.DataFrame(results).sort_values('accuracy', ascending=False)
print(results_df.head(15).to_string(index=False))

print("\n" + "="*80)
print("\nTop 5 Features for Rule-Based Classification:")
print("="*80)
top_features = results_df.head(5)
for _, row in top_features.iterrows():
    print(f"{row['feature']:30s}: if feature {row['direction']} {row['threshold']:.4f} => accuracy {row['accuracy']:.4f}")
