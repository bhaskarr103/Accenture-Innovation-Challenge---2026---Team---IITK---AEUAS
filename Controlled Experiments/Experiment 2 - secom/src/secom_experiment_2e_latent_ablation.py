import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

# Existing preprocessed SECOM file
DATA_FILE = "secom_sae_features.npz"

# Latent dimensions to test
LATENT_DIMS = [8, 16, 32]

# SAE architecture
HIDDEN_1 = 128
HIDDEN_2 = 32

# Sparsity
RHO = 0.05
BETA = 0.01

# L2
L2_WEIGHT = 1e-5

# Training
MAX_ITER = 500

# Threshold search
THRESHOLDS = np.arange(
    0.05,
    0.51,
    0.01
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# SAE
# ============================================================

class SparseAutoencoder(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_1,
        hidden_2,
        latent_dim
    ):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                hidden_1
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_1,
                hidden_2
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_2,
                latent_dim
            ),

            nn.Sigmoid()
        )


        self.decoder = nn.Sequential(

            nn.Linear(
                latent_dim,
                hidden_2
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_2,
                hidden_1
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_1,
                input_dim
            )
        )


    def encode(self, x):

        return self.encoder(x)


    def forward(self, x):

        z = self.encode(x)

        reconstruction = self.decoder(z)

        return reconstruction, z


# ============================================================
# KL SPARSITY
# ============================================================

def kl_sparsity(
    rho,
    rho_hat
):

    rho = torch.tensor(
        rho,
        dtype=torch.float32,
        device=rho_hat.device
    )

    eps = 1e-8

    rho_hat = torch.clamp(
        rho_hat,
        eps,
        1.0 - eps
    )

    term_1 = (
        rho *
        torch.log(
            rho / rho_hat
        )
    )

    term_2 = (
        (1 - rho) *
        torch.log(
            (1 - rho) /
            (1 - rho_hat)
        )
    )

    return torch.sum(
        term_1 + term_2
    )


# ============================================================
# TRAIN SAE
# ============================================================

def train_sae(
    X_train,
    latent_dim
):

    input_dim = X_train.shape[1]

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"TRAINING SAE — LATENT DIMENSION = {latent_dim}"
    )

    print(
        "=" * 80
    )

    print(
        f"Input dimension  : {input_dim}"
    )

    print(
        f"Hidden dimension 1: {HIDDEN_1}"
    )

    print(
        f"Hidden dimension 2: {HIDDEN_2}"
    )

    print(
        f"Latent dimension  : {latent_dim}"
    )

    print(
        f"Rho               : {RHO}"
    )

    print(
        f"Beta              : {BETA}"
    )


    model = SparseAutoencoder(
        input_dim=input_dim,
        hidden_1=HIDDEN_1,
        hidden_2=HIDDEN_2,
        latent_dim=latent_dim
    ).to(DEVICE)


    X_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
        device=DEVICE
    )


    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=MAX_ITER,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=20,
        line_search_fn="strong_wolfe"
    )


    closure_calls = 0


    def closure():

        nonlocal closure_calls

        optimizer.zero_grad()

        reconstruction, z = model(
            X_tensor
        )

        reconstruction_loss = torch.mean(
            (reconstruction - X_tensor) ** 2
        )


        mean_activation = torch.mean(
            z,
            dim=0
        )


        sparsity_loss = kl_sparsity(
            RHO,
            mean_activation
        )


        l2_loss = torch.tensor(
            0.0,
            device=DEVICE
        )

        for parameter in model.parameters():

            l2_loss = (
                l2_loss +
                torch.sum(parameter ** 2)
            )


        total_loss = (
            reconstruction_loss
            +
            BETA * sparsity_loss
            +
            L2_WEIGHT * l2_loss
        )


        total_loss.backward()

        closure_calls += 1

        return total_loss


    optimizer.step(
        closure
    )


    # --------------------------------------------------------
    # FINAL DIAGNOSTICS
    # --------------------------------------------------------

    with torch.no_grad():

        reconstruction, z = model(
            X_tensor
        )

        reconstruction_loss = torch.mean(
            (reconstruction - X_tensor) ** 2
        ).item()


        mean_activation = torch.mean(
            z,
            dim=0
        ).cpu().numpy()


        sparsity_loss = kl_sparsity(
            RHO,
            torch.mean(z, dim=0)
        ).item()


        l2_loss = 0.0

        for parameter in model.parameters():

            l2_loss += torch.sum(
                parameter ** 2
            ).item()


        total_loss = (
            reconstruction_loss
            +
            BETA * sparsity_loss
            +
            L2_WEIGHT * l2_loss
        )


    print(
        "\n========== SAE RESULT =========="
    )

    print(
        f"Final total loss      : {total_loss:.6f}"
    )

    print(
        f"Reconstruction MSE    : "
        f"{reconstruction_loss:.6f}"
    )

    print(
        f"L2 loss               : "
        f"{l2_loss:.6f}"
    )

    print(
        f"KL sparsity loss      : "
        f"{sparsity_loss:.6f}"
    )

    print(
        "Mean latent activation:"
    )

    print(
        np.round(
            mean_activation,
            4
        )
    )

    print(
        f"L-BFGS closure calls  : "
        f"{closure_calls}"
    )


    return model


# ============================================================
# EXTRACT LATENT FEATURES
# ============================================================

def extract_latent(
    model,
    X
):

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE
    )

    model.eval()

    with torch.no_grad():

        z = model.encode(
            X_tensor
        ).cpu().numpy()

    return z


# ============================================================
# BAGGING
# ============================================================

def create_bagging():

    return BaggingClassifier(

        estimator=DecisionTreeClassifier(
            class_weight="balanced",
            random_state=SEED
        ),

        n_estimators=100,

        random_state=SEED,

        n_jobs=-1
    )


# ============================================================
# LEARN THRESHOLD
# ============================================================

def learn_threshold(
    probabilities,
    y
):

    rows = []


    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)


        precision = precision_score(
            y,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y,
            predictions,
            zero_division=0
        )

        accuracy = accuracy_score(
            y,
            predictions
        )


        rows.append({

            "threshold": threshold,

            "accuracy": accuracy,

            "precision": precision,

            "recall": recall,

            "f1": f1

        })


    threshold_df = pd.DataFrame(
        rows
    )


    best_index = (
        threshold_df["f1"]
        .idxmax()
    )


    best = threshold_df.loc[
        best_index
    ]


    return (
        best["threshold"],
        threshold_df
    )


# ============================================================
# EVALUATE
# ============================================================

def evaluate(
    name,
    probabilities,
    y_test,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)


    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )


    predicted_failures = np.sum(
        predictions == 1
    )

    detected_failures = np.sum(
        (y_test == 1)
        &
        (predictions == 1)
    )


    print(
        "\n========== TEST RESULT =========="
    )

    print(
        f"Method       : {name}"
    )

    print(
        f"Threshold    : {threshold:.2f}"
    )

    print(
        f"Accuracy     : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision    : {precision * 100:.2f}%"
    )

    print(
        f"Recall       : {recall * 100:.2f}%"
    )

    print(
        f"F1           : {f1 * 100:.2f}%"
    )

    print(
        "Confusion matrix:"
    )

    print(cm)

    print(
        f"Predicted failures : "
        f"{predicted_failures}"
    )

    print(
        f"Detected failures  : "
        f"{detected_failures}"
    )


    return {

        "Method": name,

        "Dimension": None,

        "Threshold": threshold,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "Predicted Failures":
            predicted_failures,

        "Detected Failures":
            detected_failures

    }


# ============================================================
# LOAD DATA
# ============================================================

print(
    "\n========== LOADING SECOM =========="
)

data = np.load(
    DATA_FILE
)


X_train = data["X_train"]
X_val = data["X_val"]
X_test = data["X_test"]

y_train = data["y_train"]
y_val = data["y_val"]
y_test = data["y_test"]


print(
    "Original training:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Test:",
    X_test.shape
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


results = []


# ============================================================
# BASELINE — ORIGINAL 458
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "BASELINE — ORIGINAL 458 FEATURES"
)

print(
    "=" * 80
)


baseline_clf = create_bagging()

baseline_clf.fit(
    X_train,
    y_train
)


baseline_val_prob = (
    baseline_clf
    .predict_proba(X_val)[:, 1]
)

baseline_test_prob = (
    baseline_clf
    .predict_proba(X_test)[:, 1]
)


baseline_threshold, baseline_table = (
    learn_threshold(
        baseline_val_prob,
        y_val
    )
)


print(
    f"Best validation threshold: "
    f"{baseline_threshold:.2f}"
)

print(
    f"Validation F1: "
    f"{baseline_table['f1'].max() * 100:.2f}%"
)


baseline_result = evaluate(
    "Original 458",
    baseline_test_prob,
    y_test,
    baseline_threshold
)

baseline_result["Dimension"] = 458

results.append(
    baseline_result
)


# ============================================================
# SAE LATENT-DIMENSION EXPERIMENT
# ============================================================

for latent_dim in LATENT_DIMS:

    set_seed(SEED)

    # --------------------------------------------------------
    # TRAIN SAE
    # --------------------------------------------------------

    sae = train_sae(
        X_train,
        latent_dim
    )


    # --------------------------------------------------------
    # EXTRACT FEATURES
    # --------------------------------------------------------

    print(
        "\n========== EXTRACTING LATENT FEATURES =========="
    )

    Z_train = extract_latent(
        sae,
        X_train
    )

    Z_val = extract_latent(
        sae,
        X_val
    )

    Z_test = extract_latent(
        sae,
        X_test
    )


    print(
        "Training:",
        Z_train.shape
    )

    print(
        "Validation:",
        Z_val.shape
    )

    print(
        "Test:",
        Z_test.shape
    )


    # --------------------------------------------------------
    # LATENT COLLAPSE CHECK
    # --------------------------------------------------------

    latent_std = np.std(
        Z_train,
        axis=0
    )

    collapsed = np.sum(
        latent_std < 1e-3
    )


    print(
        "\n========== LATENT COLLAPSE CHECK =========="
    )

    print(
        "Latent std:"
    )

    print(
        np.round(
            latent_std,
            6
        )
    )

    print(
        f"Collapsed neurons: "
        f"{collapsed} / {latent_dim}"
    )


    # ========================================================
    # SAE ONLY
    # ========================================================

    print(
        "\n"
        + "-" * 80
    )

    print(
        f"SAE ONLY — LATENT {latent_dim}"
    )

    print(
        "-" * 80
    )


    clf = create_bagging()

    clf.fit(
        Z_train,
        y_train
    )


    val_prob = clf.predict_proba(
        Z_val
    )[:, 1]

    test_prob = clf.predict_proba(
        Z_test
    )[:, 1]


    threshold, table = learn_threshold(
        val_prob,
        y_val
    )


    print(
        f"Best threshold: {threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{table['f1'].max() * 100:.2f}%"
    )


    result = evaluate(
        f"SAE-{latent_dim}",
        test_prob,
        y_test,
        threshold
    )

    result["Dimension"] = latent_dim

    results.append(
        result
    )


    # ========================================================
    # HYBRID
    # ========================================================

    print(
        "\n"
        + "-" * 80
    )

    print(
        f"HYBRID — 458 + {latent_dim}"
    )

    print(
        "-" * 80
    )


    H_train = np.hstack([
        X_train,
        Z_train
    ])

    H_val = np.hstack([
        X_val,
        Z_val
    ])

    H_test = np.hstack([
        X_test,
        Z_test
    ])


    print(
        "Hybrid dimension:",
        H_train.shape[1]
    )


    clf = create_bagging()

    clf.fit(
        H_train,
        y_train
    )


    val_prob = clf.predict_proba(
        H_val
    )[:, 1]

    test_prob = clf.predict_proba(
        H_test
    )[:, 1]


    threshold, table = learn_threshold(
        val_prob,
        y_val
    )


    print(
        f"Best threshold: {threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{table['f1'].max() * 100:.2f}%"
    )


    result = evaluate(
        f"Hybrid-{latent_dim}",
        test_prob,
        y_test,
        threshold
    )

    result["Dimension"] = (
        458 + latent_dim
    )

    results.append(
        result
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


print(
    "\n"
    + "=" * 110
)

print(
    "EXPERIMENT 2E — LATENT DIMENSION ABLATION"
)

print(
    "=" * 110
)

print(
    f"{'Method':<18}"
    f"{'Dim':>7}"
    f"{'Threshold':>12}"
    f"{'Accuracy':>12}"
    f"{'Precision':>13}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
)

print(
    "-" * 110
)


for _, row in results_df.iterrows():

    print(
        f"{row['Method']:<18}"
        f"{int(row['Dimension']):>7}"
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
        f"{row['Method']:<18}"
        f"Predicted = "
        f"{int(row['Predicted Failures']):3d}"
        f" | Detected = "
        f"{int(row['Detected Failures']):3d}"
    )


# ============================================================
# BEST METHOD
# ============================================================

best_idx = results_df["F1"].idxmax()

best = results_df.loc[
    best_idx
]


print(
    "\n========== BEST RESULT =========="
)

print(
    f"Method    : {best['Method']}"
)

print(
    f"Dimension : {int(best['Dimension'])}"
)

print(
    f"F1        : {best['F1'] * 100:.2f}%"
)

print(
    f"Recall    : {best['Recall'] * 100:.2f}%"
)

print(
    f"Precision : {best['Precision'] * 100:.2f}%"
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    "secom_experiment_2e_latent_ablation.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


print(
    "\nResults saved to:"
)

print(
    output_file
)


print(
    "\n========== EXPERIMENT 2E COMPLETE =========="
)