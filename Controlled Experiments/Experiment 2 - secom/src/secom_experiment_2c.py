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

from sklearn.linear_model import LogisticRegression


# ============================================================
# CONFIGURATION
# ============================================================

FEATURE_FILE = "secom_sae_features.npz"

RANDOM_STATE = 42

THRESHOLDS = np.arange(
    0.05,
    0.96,
    0.01
)


# ============================================================
# LOAD SAE FEATURES
# ============================================================

print("\n========== LOADING SECOM SAE FEATURES ==========")

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


print("Original dimension :", X_train.shape[1])
print("Latent dimension   :", Z_train.shape[1])


# ============================================================
# HYBRID FEATURES
# ============================================================

X_train_hybrid = np.hstack([
    X_train,
    Z_train
])

X_val_hybrid = np.hstack([
    X_val,
    Z_val
])

X_test_hybrid = np.hstack([
    X_test,
    Z_test
])


print(
    "Hybrid dimension   :",
    X_train_hybrid.shape[1]
)

print(
    "Training samples   :",
    X_train.shape[0]
)

print(
    "Validation samples :",
    X_val.shape[0]
)

print(
    "Test samples       :",
    X_test.shape[0]
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
# TRAIN CLASSIFIERS
# ============================================================

print("\n========== TRAINING CLASSIFIERS ==========")

for name, clf in classifiers.items():

    print("Training:", name)

    clf.fit(
        X_train_hybrid,
        y_train
    )


# ============================================================
# RAW PROBABILITIES
# ============================================================

print("\n========== GENERATING RAW PROBABILITIES ==========")

val_raw = {}
test_raw = {}

for name, clf in classifiers.items():

    val_raw[name] = clf.predict_proba(
        X_val_hybrid
    )[:, 1]

    test_raw[name] = clf.predict_proba(
        X_test_hybrid
    )[:, 1]


# ============================================================
# PROBABILITY CALIBRATION
# ============================================================
#
# We calibrate the probability scale using ONLY:
#
#   Training predictions -> calibration model training
#   Validation -> threshold selection
#
# IMPORTANT:
# The test set is never used here.
#
# We use a simple logistic calibration model:
#
# calibrated_p = sigmoid(a * raw_p + b)
#
# ============================================================

print(
    "\n========== PROBABILITY CALIBRATION =========="
)


calibrators = {}

val_calibrated = {}
test_calibrated = {}


for name in classifiers:

    print(
        "Calibrating:",
        name
    )

    # --------------------------------------------------------
    # Generate out-of-fold-like training probabilities
    # using a fresh classifier with 5-fold manual splitting.
    #
    # This prevents fitting the calibration model directly
    # on predictions generated from the same training samples.
    # --------------------------------------------------------

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    oof_probability = np.zeros(
        len(y_train)
    )


    for train_idx, calibration_idx in skf.split(
        X_train_hybrid,
        y_train
    ):

        base_clf = classifiers[name]

        # clone-like fresh model
        if name == "SVM":

            clf_fold = SVC(
                kernel="rbf",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE
            )

        elif name == "KNN":

            clf_fold = KNeighborsClassifier(
                n_neighbors=7
            )

        elif name == "GaussianNB":

            clf_fold = GaussianNB()

        else:

            clf_fold = BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                ),
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )


        clf_fold.fit(
            X_train_hybrid[train_idx],
            y_train[train_idx]
        )

        oof_probability[calibration_idx] = (
            clf_fold.predict_proba(
                X_train_hybrid[calibration_idx]
            )[:, 1]
        )


    # --------------------------------------------------------
    # Fit calibration model
    # --------------------------------------------------------

    calibrator = LogisticRegression(
        max_iter=1000,
        class_weight=None,
        random_state=RANDOM_STATE
    )

    calibrator.fit(
        oof_probability.reshape(-1, 1),
        y_train
    )

    calibrators[name] = calibrator


    # --------------------------------------------------------
    # Calibrate validation/test probabilities
    # --------------------------------------------------------

    val_calibrated[name] = (
        calibrator.predict_proba(
            val_raw[name].reshape(-1, 1)
        )[:, 1]
    )

    test_calibrated[name] = (
        calibrator.predict_proba(
            test_raw[name].reshape(-1, 1)
        )[:, 1]
    )


# ============================================================
# CALIBRATION DIAGNOSTICS
# ============================================================

print(
    "\n========== CALIBRATED PROBABILITY DIAGNOSTICS =========="
)

print(
    f"{'Classifier':<15}"
    f"{'Val Mean':>12}"
    f"{'Val Min':>12}"
    f"{'Val Max':>12}"
    f"{'Test Mean':>13}"
    f"{'Test Min':>12}"
    f"{'Test Max':>12}"
)

for name in classifiers:

    vp = val_calibrated[name]
    tp = test_calibrated[name]

    print(
        f"{name:<15}"
        f"{vp.mean():>12.4f}"
        f"{vp.min():>12.4f}"
        f"{vp.max():>12.4f}"
        f"{tp.mean():>13.4f}"
        f"{tp.min():>12.4f}"
        f"{tp.max():>12.4f}"
    )


# ============================================================
# LEARN CLASSIFIER-SPECIFIC THRESHOLDS
# ============================================================

print(
    "\n========== LEARNING CALIBRATED THRESHOLDS =========="
)


best_thresholds = {}

competence = {}


for name in classifiers:

    probabilities = val_calibrated[name]

    best_f1 = -1
    best_threshold = 0.50

    best_precision = 0.0
    best_recall = 0.0


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


        if f1 > best_f1:

            best_f1 = f1

            best_threshold = threshold

            best_precision = precision

            best_recall = recall


    best_thresholds[name] = best_threshold


    competence[name] = {

        "failure_f1": best_f1,

        "failure_precision": best_precision,

        "failure_recall": best_recall

    }


    print(
        f"\n{name}"
    )

    print(
        f"Threshold : {best_threshold:.2f}"
    )

    print(
        f"F1        : {best_f1 * 100:.2f}%"
    )

    print(
        f"Recall    : {best_recall * 100:.2f}%"
    )

    print(
        f"Precision : {best_precision * 100:.2f}%"
    )


# ============================================================
# COMPETENCE WEIGHTS
# ============================================================

failure_f1 = np.array([
    competence[name]["failure_f1"]
    for name in classifiers
])


failure_weights = (
    failure_f1 /
    (failure_f1.sum() + 1e-12)
)


print(
    "\n========== FAILURE COMPETENCE WEIGHTS =========="
)


for name, weight in zip(
    classifiers.keys(),
    failure_weights
):

    print(
        f"{name:<15}{weight:.4f}"
    )


# ============================================================
# BUILD LEARNED-THRESHOLD TEST PREDICTIONS
# ============================================================

print(
    "\n========== LEARNED-THRESHOLD TEST RESULTS =========="
)


individual_results = []


for name in classifiers:

    threshold = best_thresholds[name]

    prediction = (
        test_calibrated[name]
        >= threshold
    ).astype(int)


    accuracy = accuracy_score(
        y_test,
        prediction
    )

    precision = precision_score(
        y_test,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediction,
        zero_division=0
    )

    detected = np.sum(
        (y_test == 1)
        &
        (prediction == 1)
    )

    predicted = np.sum(
        prediction == 1
    )


    individual_results.append({

        "Method": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "PredictedFailures": predicted,

        "DetectedFailures": detected

    })


    print(
        f"\n{name}"
    )

    print(
        f"Threshold = {threshold:.2f}"
    )

    print(
        f"Accuracy  = {accuracy * 100:.2f}%"
    )

    print(
        f"Precision = {precision * 100:.2f}%"
    )

    print(
        f"Recall    = {recall * 100:.2f}%"
    )

    print(
        f"F1        = {f1 * 100:.2f}%"
    )

    print(
        "Confusion matrix:"
    )

    print(
        confusion_matrix(
            y_test,
            prediction
        )
    )


# ============================================================
# MAJORITY VOTING
# ============================================================

print(
    "\n========== MAJORITY VOTING =========="
)


thresholded_predictions = np.column_stack([

    (
        test_calibrated[name]
        >= best_thresholds[name]
    ).astype(int)

    for name in classifiers

])


majority_prediction = (
    thresholded_predictions.sum(axis=1)
    >=
    2
).astype(int)


# ============================================================
# ORIGINAL CSWE
# ============================================================

print(
    "\n========== ORIGINAL CSWE =========="
)


# Use hard predictions + validation failure F1.

W = np.array([

    [
        f1_score(
            y_val,
            (
                val_calibrated[name]
                >= best_thresholds[name]
            ).astype(int),
            pos_label=0,
            zero_division=0
        ),

        competence[name]["failure_f1"]

    ]

    for name in classifiers

])


def original_cswe(
    predictions,
    W
):

    output = np.zeros(
        len(predictions),
        dtype=int
    )


    for i in range(
        len(predictions)
    ):

        normal_score = 0.0
        failure_score = 0.0


        for j in range(
            predictions.shape[1]
        ):

            if predictions[i, j] == 0:

                normal_score += W[j, 0]

            else:

                failure_score += W[j, 1]


        output[i] = int(
            failure_score
            >
            normal_score
        )


    return output


original_cswe = original_cswe(
    thresholded_predictions,
    W
)


# ============================================================
# CONFIDENCE-AWARE CSWE
# ============================================================

print(
    "\n========== CALIBRATED CONFIDENCE-AWARE CSWE =========="
)


# ------------------------------------------------------------
# Important:
#
# We use calibrated probability AND competence.
#
# failure score =
#
#       calibrated probability
#       ×
#       failure competence
#
# normal score =
#
#       calibrated normal probability
#       ×
#       normal competence
#
# ------------------------------------------------------------


def confidence_cswe_scores(
    probabilities,
    W
):

    normal_scores = np.zeros(
        len(y_test)
    )

    failure_scores = np.zeros(
        len(y_test)
    )


    for j in range(
        probabilities.shape[1]
    ):

        p_failure = probabilities[:, j]

        p_normal = (
            1.0
            -
            p_failure
        )


        normal_scores += (
            p_normal
            *
            W[j, 0]
        )

        failure_scores += (
            p_failure
            *
            W[j, 1]
        )


    return (
        normal_scores,
        failure_scores
    )


probability_matrix = np.column_stack([

    test_calibrated[name]

    for name in classifiers

])


normal_scores, failure_scores = (
    confidence_cswe_scores(
        probability_matrix,
        W
    )
)


# ============================================================
# LEARN ENSEMBLE THRESHOLD
# ============================================================
#
# We MUST learn this threshold using validation only.
#
# Therefore calculate validation ensemble scores first.
# ============================================================


val_probability_matrix = np.column_stack([

    val_calibrated[name]

    for name in classifiers

])


val_normal_scores = np.zeros(
    len(y_val)
)

val_failure_scores = np.zeros(
    len(y_val)
)


for j in range(
    val_probability_matrix.shape[1]
):

    p_failure = (
        val_probability_matrix[:, j]
    )

    p_normal = (
        1.0
        -
        p_failure
    )


    val_normal_scores += (
        p_normal
        *
        W[j, 0]
    )

    val_failure_scores += (
        p_failure
        *
        W[j, 1]
    )


# Convert scores into a normalized failure score.

val_ensemble_score = (
    val_failure_scores
    /
    (
        val_failure_scores
        +
        val_normal_scores
        +
        1e-12
    )
)


test_ensemble_score = (
    failure_scores
    /
    (
        failure_scores
        +
        normal_scores
        +
        1e-12
    )
)


# ============================================================
# SEARCH ENSEMBLE THRESHOLD
# ============================================================

best_ensemble_threshold = 0.50
best_ensemble_f1 = -1


for threshold in THRESHOLDS:

    prediction = (
        val_ensemble_score
        >= threshold
    ).astype(int)


    f1 = f1_score(
        y_val,
        prediction,
        zero_division=0
    )


    if f1 > best_ensemble_f1:

        best_ensemble_f1 = f1

        best_ensemble_threshold = threshold


print(
    "\n========== ENSEMBLE THRESHOLD =========="
)

print(
    f"Best threshold: "
    f"{best_ensemble_threshold:.2f}"
)

print(
    f"Validation F1: "
    f"{best_ensemble_f1 * 100:.2f}%"
)


# ============================================================
# FINAL CONFIDENCE-AWARE CSWE PREDICTION
# ============================================================

confidence_cswe = (
    test_ensemble_score
    >= best_ensemble_threshold
).astype(int)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(
    name,
    prediction
):

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    precision = precision_score(
        y_test,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        prediction,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        prediction
    )

    detected = np.sum(
        (y_test == 1)
        &
        (prediction == 1)
    )

    predicted = np.sum(
        prediction == 1
    )


    print(
        f"\n{name}"
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
        f"Predicted failures : {predicted}"
    )

    print(
        f"Detected failures  : {detected}"
    )


    return {

        "Method": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "PredictedFailures": predicted,

        "DetectedFailures": detected

    }


# ============================================================
# FINAL COMPARISON
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "FINAL SECOM EXPERIMENT 2C"
)

print(
    "=" * 100
)


final_results = []


# Individual Bagging
bagging_prediction = (
    test_calibrated["Bagging"]
    >= best_thresholds["Bagging"]
).astype(int)


final_results.append(
    evaluate(
        "Bagging",
        bagging_prediction
    )
)


# GaussianNB
gnb_prediction = (
    test_calibrated["GaussianNB"]
    >= best_thresholds["GaussianNB"]
).astype(int)


final_results.append(
    evaluate(
        "GaussianNB",
        gnb_prediction
    )
)


# Majority voting
final_results.append(
    evaluate(
        "Majority Voting",
        majority_prediction
    )
)


# Original CSWE
final_results.append(
    evaluate(
        "Original CSWE",
        original_cswe
    )
)


# Confidence-aware CSWE
final_results.append(
    evaluate(
        "Calibrated Confidence-Aware CSWE",
        confidence_cswe
    )
)


# ============================================================
# SUMMARY TABLE
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "SUMMARY"
)

print(
    "=" * 100
)

print(
    f"{'Method':<35}"
    f"{'Accuracy':>12}"
    f"{'Precision':>13}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
)

print(
    "-" * 100
)


for r in final_results:

    print(
        f"{r['Method']:<35}"
        f"{r['Accuracy'] * 100:>11.2f}%"
        f"{r['Precision'] * 100:>12.2f}%"
        f"{r['Recall'] * 100:>11.2f}%"
        f"{r['F1'] * 100:>11.2f}%"
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


for r in final_results:

    print(
        f"{r['Method']:<35}"
        f"Predicted = {r['PredictedFailures']:3d}"
        f" | Detected = {r['DetectedFailures']:3d}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

summary_df = pd.DataFrame(
    final_results
)

summary_df.to_csv(
    "secom_experiment_2c_results.csv",
    index=False
)


print(
    "\nSaved:"
)

print(
    "secom_experiment_2c_results.csv"
)


print(
    "\n========== EXPERIMENT 2C COMPLETE =========="
)