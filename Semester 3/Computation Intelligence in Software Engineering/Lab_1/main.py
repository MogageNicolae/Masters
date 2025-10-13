import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class RuleBasedBreastCancerClassifier:
    def __init__(self):
        # Indices for Breast Cancer features (30 total)
        self.MEAN_RADIUS_IDX = 0
        self.MEAN_TEXTURE_IDX = 1
        self.MEAN_PERIMETER_IDX = 2
        self.MEAN_AREA_IDX = 3
        self.MEAN_SMOOTHNESS_IDX = 4
        self.MEAN_COMPACTNESS_IDX = 5
        self.MEAN_CONCAVITY_IDX = 6
        self.MEAN_CONCAVE_POINTS_IDX = 7
        self.MEAN_SYMMETRY_IDX = 8
        self.MEAN_FRACTAL_DIM_IDX = 9

        self.SE_RADIUS_IDX = 10
        self.SE_TEXTURE_IDX = 11
        self.SE_PERIMETER_IDX = 12
        self.SE_AREA_IDX = 13

        self.WORST_RADIUS_IDX = 20
        self.WORST_TEXTURE_IDX = 21
        self.WORST_PERIMETER_IDX = 22
        self.WORST_AREA_IDX = 23
        self.WORST_SMOOTHNESS_IDX = 24
        self.WORST_COMPACTNESS_IDX = 25
        self.WORST_CONCAVITY_IDX = 26
        self.WORST_CONCAVE_POINTS_IDX = 27

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for row in X:
            mean_radius = row[self.MEAN_RADIUS_IDX]
            mean_texture = row[self.MEAN_TEXTURE_IDX]
            mean_perimeter = row[self.MEAN_PERIMETER_IDX]
            mean_area = row[self.MEAN_AREA_IDX]
            mean_compactness = row[self.MEAN_COMPACTNESS_IDX]
            mean_concavity = row[self.MEAN_CONCAVITY_IDX]
            mean_concave_points = row[self.MEAN_CONCAVE_POINTS_IDX]

            se_texture = row[self.SE_TEXTURE_IDX]
            se_area = row[self.SE_AREA_IDX]

            worst_radius = row[self.WORST_RADIUS_IDX]
            worst_texture = row[self.WORST_TEXTURE_IDX]
            worst_perimeter = row[self.WORST_PERIMETER_IDX]
            worst_area = row[self.WORST_AREA_IDX]
            worst_smoothness = row[self.WORST_SMOOTHNESS_IDX]
            worst_compactness = row[self.WORST_COMPACTNESS_IDX]
            worst_concavity = row[self.WORST_CONCAVITY_IDX]
            worst_concave_points = row[self.WORST_CONCAVE_POINTS_IDX]

            # Rule 1: Most important feature - worst radius (strong benign indicator)
            if worst_radius < 14.5:
                preds.append(1)  # benign
            # Rule 2: Most important feature - worst radius (strong malignant indicator)
            elif worst_radius > 18.5:
                preds.append(0)  # malignant
            # Rule 3: Second most important - worst concave points (benign)
            elif worst_concave_points < 0.08:
                preds.append(1)  # benign
            # Rule 4: Second most important - worst concave points (malignant)
            elif worst_concave_points > 0.15:
                preds.append(0)  # malignant
            # Rule 5: Third most important - worst texture (benign)
            elif worst_texture < 22.0:
                preds.append(1)  # benign
            # Rule 6: Third most important - worst texture (malignant)
            elif worst_texture > 27.5:
                preds.append(0)  # malignant
            # Rule 7: Worst concavity check
            elif worst_concavity > 0.40:
                preds.append(0)  # malignant
            # Rule 8: Worst concavity (benign)
            elif worst_concavity < 0.15:
                preds.append(1)  # benign
            # Rule 9: Mean concave points (benign)
            elif mean_concave_points < 0.03:
                preds.append(1)  # benign
            # Rule 10: Mean concave points (malignant)
            elif mean_concave_points > 0.08:
                preds.append(0)  # malignant
            # Rule 11: Worst area check (large area = malignant)
            elif worst_area > 1100:
                preds.append(0)  # malignant
            # Rule 12: Worst area (small area = benign)
            elif worst_area < 550:
                preds.append(1)  # benign
            # Rule 13: Mean area check
            elif mean_area > 900:
                preds.append(0)  # malignant
            # Rule 14: Mean area (benign)
            elif mean_area < 480:
                preds.append(1)  # benign
            # Rule 15: Mean radius check
            elif mean_radius > 16.5:
                preds.append(0)  # malignant
            # Rule 16: Mean radius (benign)
            elif mean_radius < 12.5:
                preds.append(1)  # benign
            # Rule 17: Area error (high variability = malignant)
            elif se_area > 50:
                preds.append(0)  # malignant
            # Rule 18: Mean perimeter check
            elif mean_perimeter > 107:
                preds.append(0)  # malignant
            # Rule 19: Mean perimeter (benign)
            elif mean_perimeter < 82:
                preds.append(1)  # benign
            # Rule 20: Mean concavity check
            elif mean_concavity > 0.14:
                preds.append(0)  # malignant
            # Rule 21: Mean texture (malignant)
            elif mean_texture > 23:
                # Rule 22: Combined check with worst radius
                if worst_radius > 17.0:
                    preds.append(0)  # malignant
                else:
                    preds.append(1)  # benign
            # Rule 23: Worst compactness check
            elif worst_compactness > 0.35:
                preds.append(0)  # malignant
            # Rule 24: Worst perimeter check
            elif worst_perimeter > 120:
                preds.append(0)  # malignant
            # Rule 25: Mean compactness with nested check
            elif mean_compactness > 0.13:
                # Rule 26: Check worst concave points
                if worst_concave_points > 0.11:
                    preds.append(0)  # malignant
                else:
                    preds.append(1)  # benign
            # Rule 27: Texture error check
            elif se_texture > 1.5:
                preds.append(0)  # malignant
            # Rule 28: Final checks with worst radius in middle range
            elif worst_radius > 16.0:
                # Rule 29: Nested check with mean concavity
                if mean_concavity > 0.06:
                    preds.append(0)  # malignant
                else:
                    preds.append(1)  # benign
            # Rule 30: Default to benign for remaining cases
            else:
                preds.append(1)  # benign

        return np.array(preds, dtype=int)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def main():
    breast_cancer = datasets.load_breast_cancer()
    X, y = breast_cancer.data, breast_cancer.target
    target_names = breast_cancer.target_names

    n_rows, n_features = X.shape
    print(f"Dataset: Breast Cancer | Samples: {n_rows} | Features: {n_features} | Classes: {len(target_names)}")
    assert n_rows >= 100, "Dataset must have at least 100 rows."

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    svm_model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    metrics_svm = evaluate(y_test, y_pred_svm)

    rule_model = RuleBasedBreastCancerClassifier()
    y_pred_rule = rule_model.predict(X_test)
    metrics_rule = evaluate(y_test, y_pred_rule)

    def fmt(m):
        return (
            f"  Accuracy         : {m['accuracy']:.4f}\n"
            f"  Precision (macro): {m['precision_macro']:.4f}\n"
            f"  Recall (macro)   : {m['recall_macro']:.4f}\n"
            f"  F1 (macro)       : {m['f1_macro']:.4f}\n"
            f"  Confusion Matrix : {m['confusion_matrix']}\n"
        )

    print("Breast Cancer classification — Support Vector Machine (part a)")
    print(fmt(metrics_svm))

    print("Breast Cancer classification — Rule-based (<= #attributes ifs) (part b)")
    print(fmt(metrics_rule))

    def better(a, b):
        return "part (a)" if a >= b else "part (b)"

    print("Comparison summary (higher is better):")
    print(f"- Accuracy         : {better(metrics_svm['accuracy'], metrics_rule['accuracy'])}")
    print(f"- Precision (macro): {better(metrics_svm['precision_macro'], metrics_rule['precision_macro'])}")
    print(f"- Recall (macro)   : {better(metrics_svm['recall_macro'], metrics_rule['recall_macro'])}")
    print(f"- F1 (macro)       : {better(metrics_svm['f1_macro'], metrics_rule['f1_macro'])}")


if __name__ == "__main__":
    main()
