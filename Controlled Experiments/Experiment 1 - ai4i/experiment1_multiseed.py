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
    f1_score
)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "ai4i2020.csv"

SEEDS = [1, 2, 3, 4, 42]

LATENT_DIM = 4
HIDDEN_DIM = 12

RHO = 0.05
BETA_SPARSITY = 0.1
LAMBDA_L2 = 1e-5

LBFGS_MAX_ITER = 500
LBFGS_HISTORY_SIZE = 50
LBFGS_LR = 1.0


# ============================================================
# FEATURES
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


# ============================================================
# SAE
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

        return reconstruction, latent


# ============================================================
# KL SPARSITY
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
            rho_tensor / rho_hat
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
# SAE LOSS
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

    l2_loss = torch.tensor(0.0)

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
# TRAIN SAE
# ============================================================

def train_sae(
    X_train
):

    input_dim = X_train.shape[1]

    sae = SparseAutoencoder(
        input_dim,
        HIDDEN_DIM,
        LATENT_DIM
    )

    X_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    optimizer = torch.optim.LBFGS(

        sae.parameters(),

        lr=LBFGS_LR,

        max_iter=LBFGS_MAX_ITER,

        history_size=LBFGS_HISTORY_SIZE,

        line_search_fn="strong_wolfe"

    )

    def closure():

        optimizer.zero_grad()

        reconstruction, latent = sae(
            X_tensor
        )

        (
            total_loss,
            reconstruction_loss,
            l2_loss,
            sparsity_loss,
            rho_hat
        ) = calculate_loss(

            reconstruction,
            X_tensor,
            latent,
            sae

        )

        total_loss.backward()

        return total_loss

    optimizer.step(closure)

    with torch.no_grad():

        reconstruction, latent = sae(
            X_tensor
        )

        (
            total_loss,
            reconstruction_loss,
            l2_loss,
            sparsity_loss,
            rho_hat
        ) = calculate_loss(

            reconstruction,
            X_tensor,
            latent,
            sae

        )

    return sae, rho_hat.numpy()


# ============================================================
# LATENT EXTRACTION
# ============================================================

def extract_latent(
    sae,
    X
):

    tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    with torch.no_grad():

        _, latent = sae(
            tensor
        )

    return latent.numpy()


# ============================================================
# CLASSIFIERS
# ============================================================

def create_classifiers(seed):

    return {

        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            random_state=seed
        ),

        "KNN": KNeighborsClassifier(
            n_neighbors=5
        ),

        "GaussianNB": GaussianNB(),

        "Bagging": BaggingClassifier(
            n_estimators=100,
            random_state=seed
        )
    }


# ============================================================
# RUN ONE SEED
# ============================================================

def run_experiment(
    df,
    seed,
    verbose=True
):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    X_df = df[
        NUMERICAL_FEATURES
        +
        CATEGORICAL_FEATURES
    ].copy()

    y = df[
        TARGET
    ].values.astype(np.int64)

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    X_train_df, X_temp_df, y_train, y_temp = train_test_split(

        X_df,
        y,

        test_size=0.30,

        stratify=y,

        random_state=seed

    )

    X_val_df, X_test_df, y_val, y_test = train_test_split(

        X_temp_df,
        y_temp,

        test_size=0.50,

        stratify=y_temp,

        random_state=seed

    )

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SAE
    # --------------------------------------------------------

    sae, latent_mean = train_sae(
        X_train
    )

    X_train_sae = extract_latent(
        sae,
        X_train
    )

    X_val_sae = extract_latent(
        sae,
        X_val
    )

    X_test_sae = extract_latent(
        sae,
        X_test
    )

    # --------------------------------------------------------
    # HYBRID
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CLASSIFIERS
    # --------------------------------------------------------

    classifiers = create_classifiers(
        seed
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for classifier in classifiers.values():

        classifier.fit(
            X_train_hybrid,
            y_train
        )

    # --------------------------------------------------------
    # VALIDATION COMPETENCE
    # --------------------------------------------------------

    failure_f1_scores = []

    validation_probabilities = []

    validation_predictions = []

    for classifier in classifiers.values():

        prediction = classifier.predict(
            X_val_hybrid
        )

        probability = classifier.predict_proba(
            X_val_hybrid
        )[:, 1]

        failure_f1 = f1_score(
            y_val,
            prediction,
            pos_label=1,
            zero_division=0
        )

        failure_f1_scores.append(
            failure_f1
        )

        validation_probabilities.append(
            probability
        )

        validation_predictions.append(
            prediction
        )

    failure_f1_scores = np.array(
        failure_f1_scores
    )

    validation_probabilities = np.array(
        validation_probabilities
    )

    # --------------------------------------------------------
    # FAILURE COMPETENCE WEIGHTS
    # --------------------------------------------------------

    weight_sum = np.sum(
        failure_f1_scores
    )

    failure_weights = (
        failure_f1_scores
        /
        weight_sum
    )

    # --------------------------------------------------------
    # VALIDATION FAILURE SCORE
    # --------------------------------------------------------

    validation_failure_score = np.sum(

        validation_probabilities
        *
        failure_weights[:, None],

        axis=0

    )

    # --------------------------------------------------------
    # LEARN THRESHOLD
    # --------------------------------------------------------

    thresholds = np.arange(
        0.05,
        0.96,
        0.01
    )

    best_threshold = 0.50
    best_f1 = -1

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

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    # --------------------------------------------------------
    # TEST PROBABILITIES
    # --------------------------------------------------------

    test_probability_matrix = np.array([

        classifier.predict_proba(
            X_test_hybrid
        )[:, 1]

        for classifier in classifiers.values()

    ])

    # --------------------------------------------------------
    # CONFIDENCE-AWARE CSWE
    # --------------------------------------------------------

    test_failure_score = np.sum(

        test_probability_matrix
        *
        failure_weights[:, None],

        axis=0

    )

    confidence_cswe = (

        test_failure_score
        >=
        best_threshold

    ).astype(int)

    # --------------------------------------------------------
    # INDIVIDUAL PREDICTIONS
    # --------------------------------------------------------

    test_predictions = np.array([

        classifier.predict(
            X_test_hybrid
        )

        for classifier in classifiers.values()

    ])

    # --------------------------------------------------------
    # BAGGING
    # --------------------------------------------------------

    classifier_names = list(
        classifiers.keys()
    )

    bagging_index = classifier_names.index(
        "Bagging"
    )

    bagging_prediction = test_predictions[
        bagging_index
    ]

    # --------------------------------------------------------
    # MAJORITY VOTING
    # --------------------------------------------------------

    majority_prediction = (

        np.sum(
            test_predictions,
            axis=0
        )
        >= 2

    ).astype(int)

    # --------------------------------------------------------
    # ORIGINAL CSWE
    # --------------------------------------------------------

    competence_matrix = []

    for classifier in classifiers.values():

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

    original_cswe = []

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

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    predictions = {

        "Bagging":
            bagging_prediction,

        "Majority Voting":
            majority_prediction,

        "Original CSWE":
            original_cswe,

        "Confidence-Aware CSWE":
            confidence_cswe

    }

    results = {}

    for name, prediction in predictions.items():

        results[name] = {

            "accuracy":
                accuracy_score(
                    y_test,
                    prediction
                ),

            "precision":
                precision_score(
                    y_test,
                    prediction,
                    zero_division=0
                ),

            "recall":
                recall_score(
                    y_test,
                    prediction,
                    zero_division=0
                ),

            "f1":
                f1_score(
                    y_test,
                    prediction,
                    zero_division=0
                ),

            "detected":
                int(
                    np.sum(
                        (prediction == 1)
                        &
                        (y_test == 1)
                    )
                )

        }

    if verbose:

        print(
            f"\n========== SEED {seed} =========="
        )

        print(
            "Latent mean:",
            np.round(
                latent_mean,
                4
            )
        )

        print(
            "Failure competence:"
        )

        for name, weight in zip(
            classifiers.keys(),
            failure_weights
        ):

            print(
                f"  {name:<15} "
                f"{weight:.4f}"
            )

        print(
            "Best threshold:",
            round(
                best_threshold,
                2
            )
        )

        print()

        for name, values in results.items():

            print(
                f"{name:<25}"
                f"F1={values['f1']*100:6.2f}%  "
                f"Recall={values['recall']*100:6.2f}%  "
                f"Detected={values['detected']:2d}"
            )

    return results


# ============================================================
# MAIN
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "EXPERIMENT 1 — MULTI-SEED VALIDATION"
)

print(
    "=" * 80
)

print(
    "Seeds:",
    SEEDS
)


df = pd.read_csv(
    DATA_FILE
)

print(
    "\nDataset:",
    df.shape
)


all_results = []


# ============================================================
# RUN ALL SEEDS
# ============================================================

for seed in SEEDS:

    results = run_experiment(
        df,
        seed,
        verbose=True
    )

    for method, metrics in results.items():

        all_results.append({

            "seed":
                seed,

            "method":
                method,

            "accuracy":
                metrics["accuracy"],

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"],

            "f1":
                metrics["f1"],

            "detected":
                metrics["detected"]

        })


# ============================================================
# DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    all_results
)


# ============================================================
# SAVE RAW RESULTS
# ============================================================

results_df.to_csv(

    "experiment1_multiseed_results.csv",

    index=False

)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "MULTI-SEED SUMMARY"
)

print(
    "=" * 100
)

print(

    f"{'Method':<25}"
    f"{'Accuracy':>18}"
    f"{'Precision':>18}"
    f"{'Recall':>18}"
    f"{'F1':>18}"

)

print(
    "-" * 100
)


methods = results_df[
    "method"
].unique()


for method in methods:

    subset = results_df[
        results_df["method"] == method
    ]

    acc_mean = subset[
        "accuracy"
    ].mean()

    acc_std = subset[
        "accuracy"
    ].std(ddof=1)

    precision_mean = subset[
        "precision"
    ].mean()

    precision_std = subset[
        "precision"
    ].std(ddof=1)

    recall_mean = subset[
        "recall"
    ].mean()

    recall_std = subset[
        "recall"
    ].std(ddof=1)

    f1_mean = subset[
        "f1"
    ].mean()

    f1_std = subset[
        "f1"
    ].std(ddof=1)

    print(

        f"{method:<25}"

        f"{acc_mean*100:7.2f} ± "
        f"{acc_std*100:5.2f}%"

        f"{precision_mean*100:7.2f} ± "
        f"{precision_std*100:5.2f}%"

        f"{recall_mean*100:7.2f} ± "
        f"{recall_std*100:5.2f}%"

        f"{f1_mean*100:7.2f} ± "
        f"{f1_std*100:5.2f}%"

    )


# ============================================================
# FAILURE DETECTION SUMMARY
# ============================================================

print(
    "\n========== FAILURE DETECTION =========="
)

for method in methods:

    subset = results_df[
        results_df["method"] == method
    ]

    print(

        f"{method:<25}"

        f"Mean detected = "
        f"{subset['detected'].mean():.2f} / 51"

    )


# ============================================================
# BEST METHOD BY MEAN F1
# ============================================================

summary = (

    results_df
    .groupby("method")["f1"]
    .mean()
    .sort_values(
        ascending=False
    )

)


print(
    "\n========== RANKING BY MEAN F1 =========="
)

for rank, (method, score) in enumerate(

    summary.items(),

    start=1

):

    print(

        f"{rank}. "
        f"{method:<25}"
        f"{score*100:.2f}%"

    )


print(
    "\nResults saved to:"
)

print(
    "experiment1_multiseed_results.csv"
)

print(
    "\n========== EXPERIMENT 1 COMPLETE =========="
)