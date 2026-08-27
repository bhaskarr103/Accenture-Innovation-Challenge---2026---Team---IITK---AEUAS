import numpy as np
import pandas as pd

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
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

# Thresholds to test
THRESHOLDS = np.arange(
    0.05,
    0.51,
    0.01
)


# ============================================================
# LOAD
# ============================================================

print(
    "\n========== LOADING SAE FEATURES =========="
)

data = np.load(
    FEATURE_FILE
)

X_train = data["X_train"]
X_val = data["X_val"]
X_test = data["X_test"]

Z_train = data["Z_train"]
Z_val = data["Z_val"]
Z_test = data["Z_test"]

y_train = data["y_train"]
y_val = data["y_val"]
y_test = data["y_test"]


# ============================================================
# HYBRID FEATURES
# ============================================================

X_train_hybrid = np.hstack(
    [X_train, Z_train]
)

X_val_hybrid = np.hstack(
    [X_val, Z_val]
)

X_test_hybrid = np.hstack(
    [X_test, Z_test]
)


print(
    "Original dimension:",
    X_train.shape[1]
)

print(
    "Latent dimension:",
    Z_train.shape[1]
)

print(
    "Hybrid dimension:",
    X_train_hybrid.shape[1]
)


# ============================================================
# CLASSIFIERS
# ============================================================

classifiers = {

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=7
    ),

    "GaussianNB": GaussianNB(),

    "Bagging": BaggingClassifier(
        estimator=DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
}


# ============================================================
# TRAIN
# ============================================================

print(
    "\n========== TRAINING CLASSIFIERS =========="
)

for name, clf in classifiers.items():

    print(
        "Training:",
        name
    )

    clf.fit(
        X_train_hybrid,
        y_train
    )


# ============================================================
# GET PROBABILITIES
# ============================================================

print(
    "\n========== GENERATING PROBABILITIES =========="
)

val_probabilities = {}
test_probabilities = {}

for name, clf in classifiers.items():

    val_probabilities[name] = (
        clf.predict_proba(
            X_val_hybrid
        )[:, 1]
    )

    test_probabilities[name] = (
        clf.predict_proba(
            X_test_hybrid
        )[:, 1]
    )


# ============================================================
# PROBABILITY DIAGNOSTICS
# ============================================================

print(
    "\n========== PROBABILITY DIAGNOSTICS =========="
)

print(
    f"{'Classifier':<15}"
    f"{'Val Min':>10}"
    f"{'Val Max':>10}"
    f"{'Val Mean':>11}"
    f"{'Test Min':>11}"
    f"{'Test Max':>11}"
    f"{'Test Mean':>12}"
)

for name in classifiers:

    vp = val_probabilities[name]
    tp = test_probabilities[name]

    print(
        f"{name:<15}"
        f"{vp.min():>10.4f}"
        f"{vp.max():>10.4f}"
        f"{vp.mean():>11.4f}"
        f"{tp.min():>11.4f}"
        f"{tp.max():>11.4f}"
        f"{tp.mean():>12.4f}"
    )


# ============================================================
# VALIDATION THRESHOLD SEARCH
# ============================================================

print(
    "\n========== LEARNING FAILURE THRESHOLDS =========="
)

best_thresholds = {}

threshold_results = {}


for name in classifiers:

    probabilities = val_probabilities[name]

    rows = []

    for threshold in THRESHOLDS:

        prediction = (
            probabilities >= threshold
        ).astype(int)


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

        accuracy = accuracy_score(
            y_val,
            prediction
        )


        rows.append({

            "threshold": threshold,

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1": f1

        })


    result_df = pd.DataFrame(
        rows
    )

    threshold_results[name] = result_df


    # --------------------------------------------------------
    # Select threshold using F1
    # --------------------------------------------------------

    best_index = (
        result_df["f1"]
        .idxmax()
    )

    best_row = (
        result_df
        .loc[best_index]
    )

    best_threshold = (
        best_row["threshold"]
    )

    best_thresholds[name] = (
        best_threshold
    )


    print(
        f"\n{name}"
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

    print(
        f"Validation accuracy: "
        f"{best_row['accuracy'] * 100:.2f}%"
    )


# ============================================================
# TOP THRESHOLDS
# ============================================================

print(
    "\n========== TOP VALIDATION THRESHOLDS =========="
)


for name in classifiers:

    print(
        f"\n{name}"
    )

    df = threshold_results[name]

    top = (
        df.sort_values(
            by="f1",
            ascending=False
        )
        .head(10)
    )

    print(
        f"{'Threshold':<12}"
        f"{'F1':<12}"
        f"{'Recall':<12}"
        f"{'Precision':<12}"
    )

    for _, row in top.iterrows():

        print(
            f"{row['threshold']:<12.2f}"
            f"{row['f1'] * 100:<12.2f}"
            f"{row['recall'] * 100:<12.2f}"
            f"{row['precision'] * 100:<12.2f}"
        )


# ============================================================
# TEST: DEFAULT 0.50 VS LEARNED THRESHOLD
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "DEFAULT 0.50 VS LEARNED THRESHOLD"
)

print(
    "=" * 90
)


test_results = []


for name in classifiers:

    probabilities = test_probabilities[name]


    # --------------------------------------------------------
    # Default 0.50
    # --------------------------------------------------------

    default_prediction = (
        probabilities >= 0.50
    ).astype(int)


    # --------------------------------------------------------
    # Learned threshold
    # --------------------------------------------------------

    learned_prediction = (
        probabilities
        >=
        best_thresholds[name]
    ).astype(int)


    # --------------------------------------------------------
    # Evaluate default
    # --------------------------------------------------------

    default_precision = precision_score(
        y_test,
        default_prediction,
        zero_division=0
    )

    default_recall = recall_score(
        y_test,
        default_prediction,
        zero_division=0
    )

    default_f1 = f1_score(
        y_test,
        default_prediction,
        zero_division=0
    )

    default_accuracy = accuracy_score(
        y_test,
        default_prediction
    )


    # --------------------------------------------------------
    # Evaluate learned
    # --------------------------------------------------------

    learned_precision = precision_score(
        y_test,
        learned_prediction,
        zero_division=0
    )

    learned_recall = recall_score(
        y_test,
        learned_prediction,
        zero_division=0
    )

    learned_f1 = f1_score(
        y_test,
        learned_prediction,
        zero_division=0
    )

    learned_accuracy = accuracy_score(
        y_test,
        learned_prediction
    )


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    test_results.append({

        "Classifier": name,

        "Threshold": "0.50",

        "Accuracy": default_accuracy,

        "Precision": default_precision,

        "Recall": default_recall,

        "F1": default_f1,

        "PredictedFailures": np.sum(
            default_prediction == 1
        ),

        "DetectedFailures": np.sum(
            (y_test == 1)
            &
            (default_prediction == 1)
        )

    })


    test_results.append({

        "Classifier": name,

        "Threshold": (
            f"{best_thresholds[name]:.2f}"
        ),

        "Accuracy": learned_accuracy,

        "Precision": learned_precision,

        "Recall": learned_recall,

        "F1": learned_f1,

        "PredictedFailures": np.sum(
            learned_prediction == 1
        ),

        "DetectedFailures": np.sum(
            (y_test == 1)
            &
            (learned_prediction == 1)
        )

    })


# ============================================================
# PRINT RESULTS
# ============================================================

results_df = pd.DataFrame(
    test_results
)


print(
    f"\n{'Classifier':<15}"
    f"{'Threshold':<12}"
    f"{'Accuracy':>11}"
    f"{'Precision':>12}"
    f"{'Recall':>11}"
    f"{'F1':>11}"
    f"{'Pred Fail':>12}"
    f"{'Detected':>11}"
)

print(
    "-" * 105
)


for _, row in results_df.iterrows():

    print(
        f"{row['Classifier']:<15}"
        f"{row['Threshold']:<12}"
        f"{row['Accuracy'] * 100:>10.2f}%"
        f"{row['Precision'] * 100:>11.2f}%"
        f"{row['Recall'] * 100:>10.2f}%"
        f"{row['F1'] * 100:>10.2f}%"
        f"{row['PredictedFailures']:>12}"
        f"{row['DetectedFailures']:>11}"
    )


# ============================================================
# CONFUSION MATRICES FOR LEARNED THRESHOLDS
# ============================================================

print(
    "\n========== LEARNED-THRESHOLD CONFUSION MATRICES =========="
)


for name in classifiers:

    threshold = best_thresholds[name]

    prediction = (
        test_probabilities[name]
        >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_test,
        prediction
    )

    print(
        f"\n{name}"
    )

    print(
        f"Threshold = {threshold:.2f}"
    )

    print(
        cm
    )


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    "secom_threshold_results.csv",
    index=False
)

print(
    "\nResults saved to:"
)

print(
    "secom_threshold_results.csv"
)


print(
    "\n"
    + "=" * 70
)

print(
    "SECOM THRESHOLD EXPERIMENT COMPLETE"
)

print(
    "=" * 70
)