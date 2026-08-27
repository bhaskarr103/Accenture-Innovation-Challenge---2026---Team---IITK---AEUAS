import numpy as np
import pandas as pd

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_FILE = "secom_sae_features.npz"

RANDOM_STATE = 42

THRESHOLDS = np.arange(
    0.05,
    0.51,
    0.01
)


# ============================================================
# LOAD DATA
# ============================================================

print("\n========== LOADING SECOM FEATURES ==========")

data = np.load(FEATURE_FILE)

X_train = data["X_train"]
X_val = data["X_val"]
X_test = data["X_test"]

Z_train = data["Z_train"]
Z_val = data["Z_val"]
Z_test = data["Z_test"]

y_train = data["y_train"]
y_val = data["y_val"]
y_test = data["y_test"]


print("Original training:", X_train.shape)
print("SAE training     :", Z_train.shape)

print(
    "Training failures:",
    np.sum(y_train == 1)
)

print(
    "Validation failures:",
    np.sum(y_val == 1)
)

print(
    "Test failures:",
    np.sum(y_test == 1)
)


# ============================================================
# BUILD THREE FEATURE SETS
# ============================================================

feature_sets = {

    "Original 458": (
        X_train,
        X_val,
        X_test
    ),

    "SAE 8": (
        Z_train,
        Z_val,
        Z_test
    ),

    "Hybrid 466": (
        np.hstack([
            X_train,
            Z_train
        ]),

        np.hstack([
            X_val,
            Z_val
        ]),

        np.hstack([
            X_test,
            Z_test
        ])
    )
}


# ============================================================
# BAGGING FACTORY
# ============================================================

def create_bagging():

    return BaggingClassifier(
        estimator=DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# RUN ABLATION
# ============================================================

for feature_name, (
    Xtr,
    Xv,
    Xte
) in feature_sets.items():

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"EXPERIMENT: {feature_name}"
    )

    print(
        "=" * 80
    )

    print(
        "Input dimension:",
        Xtr.shape[1]
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print(
        "\n========== TRAINING BAGGING =========="
    )

    clf = create_bagging()

    clf.fit(
        Xtr,
        y_train
    )


    # --------------------------------------------------------
    # PROBABILITIES
    # --------------------------------------------------------

    val_probability = clf.predict_proba(
        Xv
    )[:, 1]

    test_probability = clf.predict_proba(
        Xte
    )[:, 1]


    # --------------------------------------------------------
    # PROBABILITY DIAGNOSTICS
    # --------------------------------------------------------

    print(
        "\n========== PROBABILITY DIAGNOSTICS =========="
    )

    print(
        f"Validation min : {val_probability.min():.4f}"
    )

    print(
        f"Validation max : {val_probability.max():.4f}"
    )

    print(
        f"Validation mean: {val_probability.mean():.4f}"
    )

    print(
        f"Test min       : {test_probability.min():.4f}"
    )

    print(
        f"Test max       : {test_probability.max():.4f}"
    )

    print(
        f"Test mean      : {test_probability.mean():.4f}"
    )


    # --------------------------------------------------------
    # LEARN THRESHOLD ON VALIDATION
    # --------------------------------------------------------

    print(
        "\n========== LEARNING THRESHOLD =========="
    )

    threshold_rows = []


    for threshold in THRESHOLDS:

        prediction = (
            val_probability >= threshold
        ).astype(int)


        accuracy = accuracy_score(
            y_val,
            prediction
        )

        precision = precision_score(
            y_val,
            prediction,
            zero_division=0
        )

        recall = recall_score(
            y_val,
            prediction,
            zero_division=0
        )

        f1 = f1_score(
            y_val,
            prediction,
            zero_division=0
        )


        threshold_rows.append({

            "threshold": threshold,

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1": f1

        })


    threshold_df = pd.DataFrame(
        threshold_rows
    )


    best_idx = (
        threshold_df["f1"]
        .idxmax()
    )

    best_row = (
        threshold_df
        .loc[best_idx]
    )


    best_threshold = (
        best_row["threshold"]
    )


    print(
        f"Best threshold : "
        f"{best_threshold:.2f}"
    )

    print(
        f"Validation F1  : "
        f"{best_row['f1'] * 100:.2f}%"
    )

    print(
        f"Validation recall: "
        f"{best_row['recall'] * 100:.2f}%"
    )

    print(
        f"Validation precision: "
        f"{best_row['precision'] * 100:.2f}%"
    )


    # --------------------------------------------------------
    # TOP VALIDATION THRESHOLDS
    # --------------------------------------------------------

    print(
        "\nTop thresholds:"
    )

    print(
        f"{'Threshold':<12}"
        f"{'F1':<12}"
        f"{'Recall':<12}"
        f"{'Precision':<12}"
    )

    top_rows = (
        threshold_df
        .sort_values(
            "f1",
            ascending=False
        )
        .head(5)
    )


    for _, row in top_rows.iterrows():

        print(
            f"{row['threshold']:<12.2f}"
            f"{row['f1'] * 100:<12.2f}"
            f"{row['recall'] * 100:<12.2f}"
            f"{row['precision'] * 100:<12.2f}"
        )


    # --------------------------------------------------------
    # TEST WITH LEARNED THRESHOLD
    # --------------------------------------------------------

    test_prediction = (
        test_probability
        >= best_threshold
    ).astype(int)


    accuracy = accuracy_score(
        y_test,
        test_prediction
    )

    precision = precision_score(
        y_test,
        test_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        test_prediction,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        test_prediction
    )


    predicted_failures = np.sum(
        test_prediction == 1
    )

    detected_failures = np.sum(
        (y_test == 1)
        &
        (test_prediction == 1)
    )


    # --------------------------------------------------------
    # TEST RESULTS
    # --------------------------------------------------------

    print(
        "\n========== TEST RESULT =========="
    )

    print(
        f"Threshold : {best_threshold:.2f}"
    )

    print(
        f"Accuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision : {precision * 100:.2f}%"
    )

    print(
        f"Recall    : {recall * 100:.2f}%"
    )

    print(
        f"F1        : {f1 * 100:.2f}%"
    )

    print(
        "Confusion matrix:"
    )

    print(cm)

    print(
        "Predicted failures:",
        predicted_failures
    )

    print(
        "Correctly detected:",
        detected_failures
    )


    results.append({

        "Feature Set": feature_name,

        "Input Dimension": Xtr.shape[1],

        "Threshold": best_threshold,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "Predicted Failures": predicted_failures,

        "Detected Failures": detected_failures

    })


# ============================================================
# FINAL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


print(
    "\n"
    + "=" * 100
)

print(
    "EXPERIMENT 2D — SAE ABLATION RESULTS"
)

print(
    "=" * 100
)

print(
    f"{'Feature Set':<20}"
    f"{'Dim':>7}"
    f"{'Threshold':>12}"
    f"{'Accuracy':>12}"
    f"{'Precision':>13}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
)

print(
    "-" * 100
)


for _, row in results_df.iterrows():

    print(
        f"{row['Feature Set']:<20}"
        f"{row['Input Dimension']:>7}"
        f"{row['Threshold']:>12.2f}"
        f"{row['Accuracy'] * 100:>11.2f}%"
        f"{row['Precision'] * 100:>12.2f}%"
        f"{row['Recall'] * 100:>11.2f}%"
        f"{row['F1'] * 100:>11.2f}%"
    )


# ============================================================
# FAILURE DETECTION
# ============================================================

print(
    "\n========== FAILURE DETECTION =========="
)

print(
    "Actual test failures:",
    np.sum(y_test == 1)
)


for _, row in results_df.iterrows():

    print(
        f"{row['Feature Set']:<20}"
        f"Predicted = {int(row['Predicted Failures']):3d}"
        f" | Detected = {int(row['Detected Failures']):3d}"
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    "secom_experiment_2d_ablation.csv",
    index=False
)


print(
    "\nResults saved to:"
)

print(
    "secom_experiment_2d_ablation.csv"
)


print(
    "\n========== EXPERIMENT 2D COMPLETE =========="
)