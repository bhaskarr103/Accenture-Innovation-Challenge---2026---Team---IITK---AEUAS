# ============================================================
# AI4I EXPERIMENT
#
# Improved Sparse Autoencoder
# +
# Confidence-Aware Competence Weighting
#
# IMPORTANT:
# This is OUR proposed modification.
# It is NOT claimed to be the original paper's CSWE.
# ============================================================

import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier


# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATA_FILE = "ai4i2020.csv"


# ============================================================
# 2. SAE SETTINGS
# ============================================================

# SAME SAE THAT GAVE US:
#
# latent ≈ [0.0685, 0.0711, 0.0584, 0.0606]
#
# We freeze this configuration.

LATENT_DIM = 4
HIDDEN_DIM = 12

RHO = 0.05

BETA_SPARSITY = 0.1

LAMBDA_L2 = 1e-5

LBFGS_MAX_ITER = 500
LBFGS_HISTORY_SIZE = 50
LBFGS_LR = 1.0


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\n========== LOADING AI4I ==========")

df = pd.read_csv(DATA_FILE)

print(
    "Dataset shape:",
    df.shape
)


# ============================================================
# 4. INPUT FEATURES
# ============================================================

NUMERICAL_FEATURES = [

    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"

]

CATEGORICAL_FEATURES = [

    "Type"

]

TARGET = "Machine failure"


print("\n========== INPUT ==========")

for feature in NUMERICAL_FEATURES:

    print(
        "-",
        feature
    )

print(
    "- Type"
)

print(
    "Target:",
    TARGET
)


# ============================================================
# 5. DATA
# ============================================================

X_df = df[
    NUMERICAL_FEATURES
    +
    CATEGORICAL_FEATURES
].copy()

y = df[
    TARGET
].values.astype(np.int64)


# ============================================================
# 6. SPLIT
# ============================================================

X_train_df, X_temp_df, y_train, y_temp = train_test_split(

    X_df,
    y,

    test_size=0.30,

    stratify=y,

    random_state=SEED

)


X_val_df, X_test_df, y_val, y_test = train_test_split(

    X_temp_df,
    y_temp,

    test_size=0.50,

    stratify=y_temp,

    random_state=SEED

)


print("\n========== DATA SPLIT ==========")

print(
    "Training   :",
    X_train_df.shape
)

print(
    "Validation :",
    X_val_df.shape
)

print(
    "Test       :",
    X_test_df.shape
)


# ============================================================
# 7. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",

            StandardScaler(),

            NUMERICAL_FEATURES

        ),

        (
            "cat",

            OneHotEncoder(

                handle_unknown="ignore",

                sparse_output=False

            ),

            CATEGORICAL_FEATURES

        )

    ]

)


X_train = preprocessor.fit_transform(
    X_train_df
)

X_val = preprocessor.transform(
    X_val_df
)

X_test = preprocessor.transform(
    X_test_df
)


print(
    "\nFinal input dimension:",
    X_train.shape[1]
)


# ============================================================
# 8. SPARSE AUTOENCODER
# ============================================================

class SparseAutoencoder(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim
    ):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                latent_dim
            ),

            nn.Sigmoid()

        )

        self.decoder = nn.Sequential(

            nn.Linear(
                latent_dim,
                hidden_dim
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_dim,
                input_dim
            )

        )


    def forward(self, x):

        latent = self.encoder(x)

        reconstruction = self.decoder(
            latent
        )

        return (
            reconstruction,
            latent
        )


# ============================================================
# 9. CREATE SAE
# ============================================================

INPUT_DIM = X_train.shape[1]

sae = SparseAutoencoder(

    INPUT_DIM,

    HIDDEN_DIM,

    LATENT_DIM

)


X_train_tensor = torch.tensor(

    X_train,

    dtype=torch.float32

)


# ============================================================
# 10. KL SPARSITY
# ============================================================

def kl_divergence(
    rho,
    rho_hat
):

    eps = 1e-8

    rho_hat = torch.clamp(

        rho_hat,

        eps,

        1.0 - eps

    )

    rho_tensor = torch.full_like(

        rho_hat,

        rho

    )

    return torch.sum(

        rho_tensor
        *
        torch.log(

            rho_tensor
            /
            rho_hat

        )

        +

        (1.0 - rho_tensor)
        *
        torch.log(

            (1.0 - rho_tensor)
            /
            (1.0 - rho_hat)

        )

    )


# ============================================================
# 11. SAE LOSS
# ============================================================

def calculate_loss(

    reconstruction,

    original,

    latent,

    model

):

    reconstruction_loss = torch.mean(

        (
            reconstruction
            -
            original
        ) ** 2

    )


    l2_loss = torch.tensor(
        0.0
    )

    for parameter in model.parameters():

        if parameter.ndim >= 2:

            l2_loss += torch.sum(
                parameter ** 2
            )


    rho_hat = torch.mean(

        latent,

        dim=0

    )


    sparsity_loss = kl_divergence(

        RHO,

        rho_hat

    )


    total_loss = (

        reconstruction_loss

        +

        (LAMBDA_L2 / 2.0)
        *
        l2_loss

        +

        BETA_SPARSITY
        *
        sparsity_loss

    )


    return (

        total_loss,

        reconstruction_loss,

        l2_loss,

        sparsity_loss,

        rho_hat

    )


# ============================================================
# 12. TRAIN SAE
# ============================================================

print(
    "\n========== TRAINING SAE =========="
)

optimizer = torch.optim.LBFGS(

    sae.parameters(),

    lr=LBFGS_LR,

    max_iter=LBFGS_MAX_ITER,

    history_size=LBFGS_HISTORY_SIZE,

    line_search_fn="strong_wolfe"

)


closure_calls = 0


def closure():

    global closure_calls

    optimizer.zero_grad()

    reconstruction, latent = sae(

        X_train_tensor

    )

    (

        total_loss,

        reconstruction_loss,

        l2_loss,

        sparsity_loss,

        rho_hat

    ) = calculate_loss(

        reconstruction,

        X_train_tensor,

        latent,

        sae

    )

    total_loss.backward()

    closure_calls += 1

    return total_loss


optimizer.step(
    closure
)


# ============================================================
# 13. SAE DIAGNOSTICS
# ============================================================

with torch.no_grad():

    reconstruction, latent = sae(

        X_train_tensor

    )

    (

        total_loss,

        reconstruction_loss,

        l2_loss,

        sparsity_loss,

        rho_hat

    ) = calculate_loss(

        reconstruction,

        X_train_tensor,

        latent,

        sae

    )


print(
    "\n========== SAE RESULT =========="
)

print(
    "Input dimension:",
    INPUT_DIM
)

print(
    "Hidden dimension:",
    HIDDEN_DIM
)

print(
    "Latent dimension:",
    LATENT_DIM
)

print(
    "Rho target:",
    RHO
)

print(
    "Beta sparsity:",
    BETA_SPARSITY
)

print(
    "Reconstruction MSE:",
    round(
        reconstruction_loss.item(),
        6
    )
)

print(
    "KL sparsity loss:",
    round(
        sparsity_loss.item(),
        6
    )
)

print(
    "Mean latent activation:"
)

print(
    np.round(
        rho_hat.numpy(),
        4
    )
)

print(
    "L-BFGS closure calls:",
    closure_calls
)


# ============================================================
# 14. EXTRACT LATENT FEATURES
# ============================================================

def extract_latent(X):

    tensor = torch.tensor(

        X,

        dtype=torch.float32

    )

    with torch.no_grad():

        _, latent = sae(
            tensor
        )

    return latent.numpy()


X_train_sae = extract_latent(
    X_train
)

X_val_sae = extract_latent(
    X_val
)

X_test_sae = extract_latent(
    X_test
)


# ============================================================
# 15. HYBRID FEATURES
# ============================================================

GAMMA = 1

X_train_hybrid = np.concatenate(

    [
        X_train,
        X_train_sae
    ],

    axis=1

)

X_val_hybrid = np.concatenate(

    [
        X_val,
        X_val_sae
    ],

    axis=1

)

X_test_hybrid = np.concatenate(

    [
        X_test,
        X_test_sae
    ],

    axis=1

)


print(
    "\n========== HYBRID FEATURES =========="
)

print(
    "Original:",
    X_train.shape[1]
)

print(
    "Latent:",
    X_train_sae.shape[1]
)

print(
    "Hybrid:",
    X_train_hybrid.shape[1]
)


# ============================================================
# 16. CLASSIFIERS
# ============================================================
#
# probability=True is enabled for SVM.
#
# This allows us to obtain:
#
# P(Failure | x)
#
# instead of only:
#
# 0 / 1
#
# ============================================================

classifiers = {

    "SVM": SVC(

        kernel="rbf",

        C=1.0,

        gamma="scale",

        probability=True,

        random_state=SEED

    ),

    "KNN": KNeighborsClassifier(

        n_neighbors=5

    ),

    "GaussianNB": GaussianNB(),

    "Bagging": BaggingClassifier(

        n_estimators=100,

        random_state=SEED

    )

}


# ============================================================
# 17. TRAIN
# ============================================================

print(
    "\n========== TRAINING CLASSIFIERS =========="
)

for name, classifier in classifiers.items():

    print(
        "Training:",
        name
    )

    classifier.fit(

        X_train_hybrid,

        y_train

    )


# ============================================================
# 18. VALIDATION COMPETENCE
# ============================================================

print(
    "\n========== VALIDATION COMPETENCE =========="
)

W = []

validation_predictions = {}

validation_probabilities = {}


print(

    f"{'Classifier':<15}"
    f"{'Failure F1':>13}"
    f"{'Failure Recall':>17}"

)


for name, classifier in classifiers.items():

    prediction = classifier.predict(

        X_val_hybrid

    )

    probability = classifier.predict_proba(

        X_val_hybrid

    )[:, 1]


    validation_predictions[name] = (
        prediction
    )

    validation_probabilities[name] = (
        probability
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

        zero_division=0

    )


    W.append(
        failure_f1
    )


    print(

        f"{name:<15}"

        f"{failure_f1:>13.4f}"

        f"{failure_recall:>17.4f}"

    )


W = np.array(W)


# ============================================================
# 19. FAILURE COMPETENCE WEIGHTS
# ============================================================
#
# Normalize only the FAILURE competence.
#
# Stronger failure classifier automatically receives
# more influence.
# ============================================================

failure_weights = (

    W
    /
    np.sum(W)

)


print(
    "\n========== FAILURE COMPETENCE WEIGHTS =========="
)

for index, name in enumerate(

    classifiers.keys()

):

    print(

        f"{name:<15}"

        f"{failure_weights[index]:.4f}"

    )


# ============================================================
# 20. VALIDATION CONFIDENCE-AWARE SCORE
# ============================================================
#
# For each validation sample:
#
# weighted_failure_probability =
#
#     sum(
#       classifier_probability
#       *
#       classifier_failure_competence
#     )
#
# ============================================================

validation_probability_matrix = np.array([

    validation_probabilities[name]

    for name in classifiers.keys()

])


validation_failure_score = (

    np.sum(

        validation_probability_matrix
        *
        failure_weights[:, None],

        axis=0

    )

)


# ============================================================
# 21. LEARN THRESHOLD FROM VALIDATION
# ============================================================
#
# IMPORTANT:
#
# Test set is NOT used to choose threshold.
#
# ============================================================

print(
    "\n========== LEARNING CONFIDENCE THRESHOLD =========="
)

thresholds = np.arange(

    0.05,

    0.96,

    0.01

)


best_threshold = 0.50

best_f1 = -1

best_recall = 0

best_precision = 0


threshold_results = []


for threshold in thresholds:

    prediction = (

        validation_failure_score
        >= threshold

    ).astype(int)


    f1 = f1_score(

        y_val,

        prediction,

        zero_division=0

    )


    recall = recall_score(

        y_val,

        prediction,

        zero_division=0

    )


    precision = precision_score(

        y_val,

        prediction,

        zero_division=0

    )


    threshold_results.append(

        (
            threshold,
            f1,
            recall,
            precision
        )

    )


    if f1 > best_f1:

        best_f1 = f1

        best_threshold = threshold

        best_recall = recall

        best_precision = precision


print(
    "Best threshold:",
    round(
        best_threshold,
        2
    )
)

print(
    "Validation F1:",
    round(
        best_f1,
        4
    )
)

print(
    "Validation Recall:",
    round(
        best_recall,
        4
    )
)

print(
    "Validation Precision:",
    round(
        best_precision,
        4
    )
)


# ============================================================
# 22. TOP THRESHOLDS
# ============================================================

threshold_results.sort(

    key=lambda x: x[1],

    reverse=True

)


print(
    "\nTop validation thresholds:"
)

print(

    f"{'Threshold':<12}"
    f"{'F1':<12}"
    f"{'Recall':<12}"
    f"{'Precision':<12}"

)


for threshold, f1, recall, precision in (

    threshold_results[:10]

):

    print(

        f"{threshold:<12.2f}"
        f"{f1:<12.4f}"
        f"{recall:<12.4f}"
        f"{precision:<12.4f}"

    )


# ============================================================
# 23. TEST PROBABILITIES
# ============================================================

test_probability_matrix = np.array([

    classifier.predict_proba(

        X_test_hybrid

    )[:, 1]

    for classifier in classifiers.values()

])


# ============================================================
# 24. CONFIDENCE-AWARE CSWE
# ============================================================
#
# Each classifier contributes:
#
# P(failure | x)
#          ×
# failure competence
#
# ============================================================

test_failure_score = (

    np.sum(

        test_probability_matrix
        *
        failure_weights[:, None],

        axis=0

    )

)


confidence_cswe = (

    test_failure_score
    >=
    best_threshold

).astype(int)


# ============================================================
# 25. ORIGINAL CSWE
# ============================================================

test_predictions = np.array([

    classifier.predict(

        X_test_hybrid

    )

    for classifier in classifiers.values()

])


original_cswe = []


# For comparison only.
#
# This is the original paper-style hard-vote CSWE
# using class-specific F1.

competence_matrix = []


for name, classifier in classifiers.items():

    prediction = classifier.predict(

        X_val_hybrid

    )

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

    competence_matrix.append(

        [
            normal_f1,
            failure_f1
        ]

    )


competence_matrix = np.array(
    competence_matrix
)


for sample_idx in range(

    len(y_test)

):

    scores = np.zeros(2)


    for classifier_idx in range(

        len(classifiers)

    ):

        predicted_class = (

            test_predictions[
                classifier_idx,
                sample_idx
            ]

        )


        scores[
            predicted_class
        ] += competence_matrix[

            classifier_idx,

            predicted_class

        ]


    original_cswe.append(

        np.argmax(scores)

    )


original_cswe = np.array(
    original_cswe
)


# ============================================================
# 26. BASELINES
# ============================================================

classifier_names = list(
    classifiers.keys()
)


bagging_index = classifier_names.index(
    "Bagging"
)


bagging_predictions = (

    test_predictions[
        bagging_index
    ]

)


majority_predictions = (

    np.sum(

        test_predictions,

        axis=0

    )
    >= 2

).astype(int)


# ============================================================
# 27. EVALUATION FUNCTION
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


    print(

        f"{name:<25}"

        f"{accuracy*100:>10.2f}%"

        f"{precision*100:>12.2f}%"

        f"{recall*100:>12.2f}%"

        f"{f1*100:>10.2f}%"

    )


# ============================================================
# 28. FINAL COMPARISON
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "FINAL COMPARISON"
)

print(
    "=" * 90
)

print(

    f"{'Method':<25}"
    f"{'Accuracy':>10}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'F1':>10}"

)

print(
    "-" * 90
)


evaluate(

    "Bagging",

    y_test,

    bagging_predictions

)


evaluate(

    "Majority Voting",

    y_test,

    majority_predictions

)


evaluate(

    "Original CSWE",

    y_test,

    original_cswe

)


evaluate(

    "Confidence-Aware CSWE",

    y_test,

    confidence_cswe

)


# ============================================================
# 29. CONFUSION MATRICES
# ============================================================

print(
    "\n========== CONFUSION MATRICES =========="
)


predictions = {

    "Bagging":
        bagging_predictions,

    "Majority Voting":
        majority_predictions,

    "Original CSWE":
        original_cswe,

    "Confidence-Aware CSWE":
        confidence_cswe

}


for name, prediction in predictions.items():

    print(
        f"\n{name}"
    )

    print(

        confusion_matrix(

            y_test,

            prediction

        )

    )


# ============================================================
# 30. FAILURE DETECTION
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


for name, prediction in predictions.items():

    predicted_failures = np.sum(

        prediction == 1

    )


    detected_failures = np.sum(

        (prediction == 1)
        &
        (y_test == 1)

    )


    print(

        f"{name:<25}"

        f"Predicted = "
        f"{predicted_failures:4d} | "

        f"Detected = "
        f"{detected_failures:4d}"

    )


# ============================================================
# 31. CONFIDENCE SCORE DIAGNOSTICS
# ============================================================

print(
    "\n========== TEST CONFIDENCE DIAGNOSTICS =========="
)

print(
    "Minimum failure score:",
    round(
        test_failure_score.min(),
        4
    )
)

print(
    "Maximum failure score:",
    round(
        test_failure_score.max(),
        4
    )
)

print(
    "Mean failure score:",
    round(
        test_failure_score.mean(),
        4
    )
)

print(
    "Selected threshold:",
    round(
        best_threshold,
        2
    )
)


print(
    "\n========== EXPERIMENT COMPLETE =========="
)