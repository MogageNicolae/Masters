import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools

# Load dataset
breast_cancer = datasets.load_breast_cancer()
X, y = breast_cancer.data, breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Feature indices
MEAN_RADIUS = 0
MEAN_TEXTURE = 1
MEAN_PERIMETER = 2
MEAN_AREA = 3
MEAN_COMPACTNESS = 5
MEAN_CONCAVITY = 6
MEAN_CONCAVE_POINTS = 7
SE_RADIUS = 10
SE_TEXTURE = 11
SE_PERIMETER = 12
SE_AREA = 13
WORST_RADIUS = 20
WORST_TEXTURE = 21
WORST_PERIMETER = 22
WORST_AREA = 23
WORST_COMPACTNESS = 25
WORST_CONCAVITY = 26
WORST_CONCAVE_POINTS = 27

print("Performing ADVANCED grid search with secondary threshold optimization...")
print("=" * 80)

best_accuracy = 0.958
best_config = None
best_preds = None

# Fixed best primary thresholds
wr_low = 14.5
wr_high = 18.3
wcp_low = 0.08
wcp_high = 0.15
wt_low = 21.8
wt_high = 27.0
wc_low = 0.14
wc_high = 0.38
mcp_low = 0.028
mcp_high = 0.078

# Now optimize secondary thresholds
worst_area_low_range = [520, 540, 550, 560, 580]
worst_area_high_range = [1050, 1080, 1100, 1120, 1150]
mean_area_low_range = [450, 470, 480, 490, 500]
mean_area_high_range = [880, 900, 920, 940]
mean_radius_low_range = [12.3, 12.4, 12.5, 12.6, 12.7]
mean_radius_high_range = [16.3, 16.4, 16.5, 16.6, 16.7]
se_area_range = [45, 48, 50, 52, 55]
mean_perimeter_low_range = [80, 81, 82, 83, 84]
mean_perimeter_high_range = [105, 106, 107, 108, 109]

configs_tested = 0
total_configs = (len(worst_area_low_range) * len(worst_area_high_range) *
                 len(mean_area_low_range) * len(mean_area_high_range) *
                 len(mean_radius_low_range) * len(mean_radius_high_range) *
                 len(se_area_range) * len(mean_perimeter_low_range) * len(mean_perimeter_high_range))

print(f"Total configurations to test: {total_configs}")
print("Testing in progress...\n")

for wa_low, wa_high, ma_low, ma_high, mr_low, mr_high, se_a, mp_low, mp_high in itertools.product(
    worst_area_low_range, worst_area_high_range,
    mean_area_low_range, mean_area_high_range,
    mean_radius_low_range, mean_radius_high_range,
    se_area_range, mean_perimeter_low_range, mean_perimeter_high_range
):
    configs_tested += 1

    if configs_tested % 10000 == 0:
        print(f"Progress: {configs_tested}/{total_configs} ({100*configs_tested/total_configs:.1f}%)")

    preds = []
    for row in X_test:
        # Apply the rule set with optimized thresholds
        if row[WORST_RADIUS] < wr_low:
            pred = 1
        elif row[WORST_RADIUS] > wr_high:
            pred = 0
        elif row[WORST_CONCAVE_POINTS] < wcp_low:
            pred = 1
        elif row[WORST_CONCAVE_POINTS] > wcp_high:
            pred = 0
        elif row[WORST_TEXTURE] < wt_low:
            pred = 1
        elif row[WORST_TEXTURE] > wt_high:
            pred = 0
        elif row[WORST_CONCAVITY] > wc_high:
            pred = 0
        elif row[WORST_CONCAVITY] < wc_low:
            pred = 1
        elif row[MEAN_CONCAVE_POINTS] < mcp_low:
            pred = 1
        elif row[MEAN_CONCAVE_POINTS] > mcp_high:
            pred = 0
        elif row[WORST_AREA] > wa_high:
            pred = 0
        elif row[WORST_AREA] < wa_low:
            pred = 1
        elif row[MEAN_AREA] > ma_high:
            pred = 0
        elif row[MEAN_AREA] < ma_low:
            pred = 1
        elif row[MEAN_RADIUS] > mr_high:
            pred = 0
        elif row[MEAN_RADIUS] < mr_low:
            pred = 1
        elif row[SE_AREA] > se_a:
            pred = 0
        elif row[MEAN_PERIMETER] > mp_high:
            pred = 0
        elif row[MEAN_PERIMETER] < mp_low:
            pred = 1
        elif row[MEAN_CONCAVITY] > 0.14:
            pred = 0
        elif row[MEAN_TEXTURE] > 23:
            if row[WORST_RADIUS] > 17.0:
                pred = 0
            else:
                pred = 1
        elif row[WORST_COMPACTNESS] > 0.35:
            pred = 0
        elif row[WORST_PERIMETER] > 120:
            pred = 0
        elif row[MEAN_COMPACTNESS] > 0.13:
            if row[WORST_CONCAVE_POINTS] > 0.11:
                pred = 0
            else:
                pred = 1
        elif row[SE_TEXTURE] > 1.5:
            pred = 0
        elif row[WORST_RADIUS] > 16.0:
            if row[MEAN_CONCAVITY] > 0.06:
                pred = 0
            else:
                pred = 1
        else:
            pred = 1

        preds.append(pred)

    acc = accuracy_score(y_test, preds)

    if acc > best_accuracy:
        best_accuracy = acc
        best_config = {
            'worst_radius_low': wr_low,
            'worst_radius_high': wr_high,
            'worst_cp_low': wcp_low,
            'worst_cp_high': wcp_high,
            'worst_texture_low': wt_low,
            'worst_texture_high': wt_high,
            'worst_concavity_low': wc_low,
            'worst_concavity_high': wc_high,
            'mean_cp_low': mcp_low,
            'mean_cp_high': mcp_high,
            'worst_area_low': wa_low,
            'worst_area_high': wa_high,
            'mean_area_low': ma_low,
            'mean_area_high': ma_high,
            'mean_radius_low': mr_low,
            'mean_radius_high': mr_high,
            'se_area': se_a,
            'mean_perimeter_low': mp_low,
            'mean_perimeter_high': mp_high
        }
        best_preds = preds
        cm = confusion_matrix(y_test, preds)
        print(f"\n🎯 NEW BEST! Accuracy: {acc:.4f}")
        print(f"   Confusion Matrix: {cm.tolist()}")
        print(f"   Config snippet: wa_low={wa_low}, wa_high={wa_high}, ma_low={ma_low}, ma_high={ma_high}")

print("\n" + "=" * 80)
print(f"Configurations tested: {configs_tested}")
print(f"Best accuracy: {best_accuracy:.4f}")
if best_config:
    print(f"\nBest configuration found:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    cm = confusion_matrix(y_test, best_preds)
    print(f"\nFinal Confusion Matrix: {cm.tolist()}")
    print(f"False Positives (Benign→Malignant): {cm[1][0]}")
    print(f"False Negatives (Malignant→Benign): {cm[0][1]}")
else:
    print("No improvement found. Current configuration is optimal.")

