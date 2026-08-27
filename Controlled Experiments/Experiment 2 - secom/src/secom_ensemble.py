import numpy as np

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

CLASSIFIERS = [
    "SVM",
    "KNN",
    "GaussianNB",
    "Bagging"
]


# ============================================================
# LOAD SAE FEATURES
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


print(
    "Original training features:",
    X_train.shape
)

print(
    "SAE training features:",
    Z_train.shape
)

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
# BUILD HYBRID FEATURES
# ============================================================

print(
    "\n========== BUILDING HYBRID FEATURES =========="
)

X_train_hybrid = np.hstack(
    [
        X_train,
        Z_train
    ]
)

X_val_hybrid = np.hstack(
    [
        X_val,
        Z_val
    ]
)

X_test_hybrid = np.hstack(
    [
        X_test,
        Z_test
    ]
)


print(
    "Original dimension:",
    X_train.shape[1]
)

print(
    "SAE latent dimension:",
    Z_train.shape[1]
)

print(
    "Hybrid dimension:",
    X_train_hybrid.shape[1]
)


# ============================================================
# CLASSIFIER DEFINITIONS
# ============================================================

print(
    "\n========== CREATING CLASSIFIERS =========="
)


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
            max_depth=None,
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
# VALIDATION PREDICTIONS
# ============================================================

print(
    "\n========== VALIDATION PREDICTIONS =========="
)


validation_predictions = {}

validation_probabilities = {}

for name, clf in classifiers.items():

    prediction = clf.predict(
        X_val_hybrid
    )

    probability = clf.predict_proba(
        X_val_hybrid
    )[:, 1]

    validation_predictions[name] = prediction

    validation_probabilities[name] = probability


# ============================================================
# VALIDATION COMPETENCE
# ============================================================

print(
    "\n========== VALIDATION COMPETENCE =========="
)

print(
    f"{'Classifier':<15}"
    f"{'Normal F1':>12}"
    f"{'Failure F1':>13}"
    f"{'Failure Recall':>17}"
)


competence = {}

for name in CLASSIFIERS:

    prediction = validation_predictions[name]

    normal_f1 = f1_score(
        y_val,
        prediction,
        pos_label=0,
        zero_division=0
    )

    failure_f1 = f1_score(
        y_val,
        prediction,
        pos_label=1,
        zero_division=0
    )

    failure_recall = recall_score(
        y_val,
        prediction,
        pos_label=1,
        zero_division=0
    )

    competence[name] = {

        "normal_f1": normal_f1,

        "failure_f1": failure_f1,

        "failure_recall": failure_recall
    }

    print(
        f"{name:<15}"
        f"{normal_f1:>12.4f}"
        f"{failure_f1:>13.4f}"
        f"{failure_recall:>17.4f}"
    )


# ============================================================
# COMPETENCE MATRIX
# ============================================================

W = np.array(

    [
        [
            competence[name]["normal_f1"],
            competence[name]["failure_f1"]
        ]

        for name in CLASSIFIERS
    ]

)


print(
    "\n========== COMPETENCE MATRIX =========="
)

print(
    "Rows    = classifiers"
)

print(
    "Columns = normal / failure"
)

print(
    W
)


# ============================================================
# FAILURE COMPETENCE WEIGHTS
# ============================================================

failure_f1 = np.array(

    [
        competence[name]["failure_f1"]

        for name in CLASSIFIERS

    ]

)

failure_weights = (
    failure_f1
    /
    (failure_f1.sum() + 1e-12)
)


print(
    "\n========== FAILURE COMPETENCE WEIGHTS =========="
)

for name, weight in zip(
    CLASSIFIERS,
    failure_weights
):

    print(
        f"{name:<15}{weight:.4f}"
    )


# ============================================================
# TEST PREDICTIONS
# ============================================================

print(
    "\n========== TEST PREDICTIONS =========="
)


test_predictions = {}

test_probabilities = {}


for name, clf in classifiers.items():

    test_predictions[name] = clf.predict(
        X_test_hybrid
    )

    test_probabilities[name] = clf.predict_proba(
        X_test_hybrid
    )[:, 1]


# ============================================================
# MAJORITY VOTING
# ============================================================

prediction_matrix = np.column_stack(

    [
        test_predictions[name]

        for name in CLASSIFIERS

    ]

)

majority_prediction = (

    np.sum(
        prediction_matrix,
        axis=1
    )

    >=

    (
        len(CLASSIFIERS) / 2
    )

).astype(int)


# ============================================================
# ORIGINAL CSWE
# ============================================================
#
# For each sample:
#
# score(class) =
#     sum(
#       classifier prediction
#       *
#       competence
#     )
#
# Select class with highest score.
#

def original_cswe(
    predictions,
    competence_matrix
):

    n_samples = predictions.shape[0]

    output = np.zeros(
        n_samples,
        dtype=int
    )


    for i in range(n_samples):

        class_scores = []

        for target_class in [0, 1]:

            score = 0.0

            for j in range(
                len(CLASSIFIERS)
            ):

                if (
                    predictions[i, j]
                    ==
                    target_class
                ):

                    score += (
                        competence_matrix[
                            j,
                            target_class
                        ]
                    )

            class_scores.append(
                score
            )


        output[i] = int(
            class_scores[1]
            >
            class_scores[0]
        )


    return output


original_cswe_prediction = original_cswe(
    prediction_matrix,
    W
)


# ============================================================
# CONFIDENCE-AWARE CSWE
# ============================================================
#
# Instead of only using the hard classifier output,
# use the classifier's probability of failure.
#
# Each classifier contributes:
#
# failure probability
# ×
# failure competence
#
# Normal probability
# ×
# normal competence
#

def confidence_aware_cswe(
    probabilities,
    competence_matrix
):

    n_samples = probabilities.shape[0]

    output = np.zeros(
        n_samples,
        dtype=int
    )


    for i in range(n_samples):

        normal_score = 0.0

        failure_score = 0.0


        for j in range(
            len(CLASSIFIERS)
        ):

            p_failure = probabilities[
                i,
                j
            ]

            p_normal = (
                1.0
                -
                p_failure
            )


            normal_score += (

                p_normal

                *

                competence_matrix[
                    j,
                    0
                ]

            )


            failure_score += (

                p_failure

                *

                competence_matrix[
                    j,
                    1
                ]

            )


        output[i] = int(
            failure_score
            >
            normal_score
        )


    return output


probability_matrix = np.column_stack(

    [
        test_probabilities[name]

        for name in CLASSIFIERS

    ]

)


confidence_cswe_prediction = (
    confidence_aware_cswe(
        probability_matrix,
        W
    )
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def evaluate(
    name,
    y_true,
    prediction
):

    accuracy = accuracy_score(
        y_true,
        prediction
    )

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        prediction
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
        f"F1 Score  : {f1 * 100:.2f}%"
    )

    print(
        "Confusion matrix:"
    )

    print(
        cm
    )


    return {

        "Method": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ConfusionMatrix": cm

    }


# ============================================================
# BAGGING
# ============================================================

results = []

results.append(

    evaluate(
        "Bagging",
        y_test,
        test_predictions["Bagging"]
    )

)


# ============================================================
# MAJORITY VOTING
# ============================================================

results.append(

    evaluate(
        "Majority Voting",
        y_test,
        majority_prediction
    )

)


# ============================================================
# ORIGINAL CSWE
# ============================================================

results.append(

    evaluate(
        "Original CSWE",
        y_test,
        original_cswe_prediction
    )

)


# ============================================================
# CONFIDENCE-AWARE CSWE
# ============================================================

results.append(

    evaluate(
        "Confidence-Aware CSWE",
        y_test,
        confidence_cswe_prediction
    )

)


# ============================================================
# FINAL COMPARISON
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "FINAL SECOM COMPARISON"
)

print(
    "=" * 90
)

print(
    f"{'Method':<25}"
    f"{'Accuracy':>12}"
    f"{'Precision':>13}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
)

print(
    "-" * 90
)


for result in results:

    print(
        f"{result['Method']:<25}"
        f"{result['Accuracy'] * 100:>11.2f}%"
        f"{result['Precision'] * 100:>12.2f}%"
        f"{result['Recall'] * 100:>11.2f}%"
        f"{result['F1'] * 100:>11.2f}%"
    )


# ============================================================
# CONFUSION MATRICES
# ============================================================

print(
    "\n========== CONFUSION MATRICES =========="
)


for result in results:

    print(
        f"\n{result['Method']}"
    )

    print(
        result["ConfusionMatrix"]
    )


# ============================================================
# FAILURE DETECTION
# ============================================================

print(
    "\n========== FAILURE DETECTION =========="
)

actual_failures = np.sum(
    y_test == 1
)

print(
    "Actual failures:",
    actual_failures
)


for result in results:

    prediction = None

    if result["Method"] == "Bagging":

        prediction = test_predictions[
            "Bagging"
        ]

    elif result["Method"] == "Majority Voting":

        prediction = majority_prediction

    elif result["Method"] == "Original CSWE":

        prediction = original_cswe_prediction

    elif result["Method"] == "Confidence-Aware CSWE":

        prediction = confidence_cswe_prediction


    predicted_failures = np.sum(
        prediction == 1
    )

    correctly_detected = np.sum(

        (
            y_test == 1
        )

        &

        (
            prediction == 1
        )

    )


    print(

        f"{result['Method']:<25}"
        f"Predicted = "
        f"{predicted_failures:3d}"
        f" | Detected = "
        f"{correctly_detected:3d}"

    )


# ============================================================
# SAVE NUMERICAL RESULTS
# ============================================================

import pandas as pd


results_table = pd.DataFrame(

    [

        {
            "Method": r["Method"],
            "Accuracy": r["Accuracy"],
            "Precision": r["Precision"],
            "Recall": r["Recall"],
            "F1": r["F1"]
        }

        for r in results

    ]

)


results_table.to_csv(
    "secom_ensemble_results.csv",
    index=False
)


print(
    "\nResults saved to:"
)

print(
    "secom_ensemble_results.csv"
)


print(
    "\n"
    + "=" * 70
)

print(
    "SECOM ENSEMBLE EXPERIMENT COMPLETE"
)

print(
    "=" * 70
)