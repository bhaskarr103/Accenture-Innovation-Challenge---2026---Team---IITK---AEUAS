import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "secom.data"
LABEL_FILE = "secom_labels.data"

RANDOM_STATE = 42

# Sparse Autoencoder
RHO = 0.05

# CHANGED:
# Previous = 0.1
# New      = 0.01
BETA = 0.01

L2_WEIGHT = 1e-5

# Architecture
HIDDEN_1 = 128
HIDDEN_2 = 32
LATENT_DIM = 8

# L-BFGS
MAX_ITER = 500


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)


# ============================================================
# LOAD DATA
# ============================================================

print("\n========== LOADING SECOM ==========")

X = pd.read_csv(
    DATA_FILE,
    sep=r"\s+",
    header=None
)

labels = pd.read_csv(
    LABEL_FILE,
    sep=r"\s+",
    header=None
)

y = labels.iloc[:, 0].values

print(
    "Raw feature shape:",
    X.shape
)


# ============================================================
# LABEL CONVERSION
# ============================================================

# Original SECOM labels:
#
# -1 = normal
# +1 = failure
#
# Convert to:
#
# 0 = normal
# 1 = failure

y = np.where(
    y == 1,
    1,
    0
)

print(
    "\n========== TARGET =========="
)

print(
    "Normal :",
    np.sum(y == 0)
)

print(
    "Failure:",
    np.sum(y == 1)
)

print(
    "Failure rate:",
    round(
        np.mean(y) * 100,
        2
    ),
    "%"
)


# ============================================================
# REMOVE DUPLICATE FEATURES
# ============================================================

duplicate_mask = X.T.duplicated()

duplicate_count = duplicate_mask.sum()

X = X.loc[
    :,
    ~duplicate_mask
]

print(
    "\n========== DUPLICATE REMOVAL =========="
)

print(
    "Duplicate columns removed:",
    duplicate_count
)

print(
    "Remaining features:",
    X.shape[1]
)


# ============================================================
# REMOVE FEATURES WITH >50% MISSING
# ============================================================

missing_fraction = X.isna().mean()

missing_mask = (
    missing_fraction > 0.50
)

removed_missing = missing_mask.sum()

X = X.loc[
    :,
    ~missing_mask
]

print(
    "\n========== MISSING FEATURE REMOVAL =========="
)

print(
    "Features with >50% missing removed:",
    removed_missing
)

print(
    "Remaining features:",
    X.shape[1]
)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_train, X_temp, y_train, y_temp = train_test_split(

    X,
    y,

    test_size=0.30,

    stratify=y,

    random_state=RANDOM_STATE

)

X_val, X_test, y_val, y_test = train_test_split(

    X_temp,
    y_temp,

    test_size=0.50,

    stratify=y_temp,

    random_state=RANDOM_STATE

)


print(
    "\n========== DATA SPLIT =========="
)

print(
    "Training   :",
    X_train.shape
)

print(
    "Validation :",
    X_val.shape
)

print(
    "Test       :",
    X_test.shape
)


# ============================================================
# MEDIAN IMPUTATION
# ============================================================

print(
    "\n========== MEDIAN IMPUTATION =========="
)

imputer = SimpleImputer(
    strategy="median"
)

X_train = imputer.fit_transform(
    X_train
)

X_val = imputer.transform(
    X_val
)

X_test = imputer.transform(
    X_test
)

print(
    "Remaining NaNs:",
    np.isnan(X_train).sum()
)


# ============================================================
# STANDARDIZATION
# ============================================================

print(
    "\n========== STANDARDIZATION =========="
)

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_val = scaler.transform(
    X_val
)

X_test = scaler.transform(
    X_test
)

print(
    "Training mean:",
    np.round(
        X_train.mean(axis=0)[:10],
        4
    )
)

print(
    "Training std:",
    np.round(
        X_train.std(axis=0)[:10],
        4
    )
)


# ============================================================
# TORCH TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)


# ============================================================
# SPARSE AUTOENCODER
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


        # ====================================================
        # ENCODER
        # ====================================================
        #
        # 458
        #   ↓
        # 128 + ReLU
        #   ↓
        # 32 + ReLU
        #   ↓
        # 8 + Sigmoid
        #
        # Sigmoid is retained at the latent layer because
        # the KL sparsity formulation operates on activations
        # in the [0,1] range.
        #

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


        # ====================================================
        # DECODER
        # ====================================================
        #
        # 8
        #   ↓
        # 32 + ReLU
        #   ↓
        # 128 + ReLU
        #   ↓
        # 458 + Linear
        #
        # Linear output is important because the input data
        # has been standardized and therefore contains both
        # positive and negative values.
        #

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

        latent = self.encoder(x)

        reconstruction = self.decoder(
            latent
        )

        return reconstruction, latent


# ============================================================
# CREATE SAE
# ============================================================

input_dim = X_train.shape[1]

model = SparseAutoencoder(

    input_dim=input_dim,

    hidden_1=HIDDEN_1,

    hidden_2=HIDDEN_2,

    latent_dim=LATENT_DIM

)


print(
    "\n========== SAE ARCHITECTURE =========="
)

print(
    "Input dimension   :",
    input_dim
)

print(
    "Hidden dimension 1:",
    HIDDEN_1
)

print(
    "Hidden dimension 2:",
    HIDDEN_2
)

print(
    "Latent dimension  :",
    LATENT_DIM
)

print(
    "Target rho        :",
    RHO
)

print(
    "Beta sparsity     :",
    BETA
)

print(
    "L2 weight         :",
    L2_WEIGHT
)


# ============================================================
# KL SPARSITY PENALTY
# ============================================================

def kl_divergence(
    rho,
    rho_hat
):

    # Avoid log(0)

    rho_hat = torch.clamp(
        rho_hat,
        1e-6,
        1.0 - 1e-6
    )

    rho_tensor = torch.tensor(
        rho,
        dtype=rho_hat.dtype,
        device=rho_hat.device
    )

    kl = (

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

    return torch.sum(kl)


# ============================================================
# LOSS
# ============================================================

mse_loss = nn.MSELoss()


def calculate_loss(x):

    reconstruction, z = model(x)


    # --------------------------------------------------------
    # Reconstruction loss
    # --------------------------------------------------------

    reconstruction_loss = mse_loss(
        reconstruction,
        x
    )


    # --------------------------------------------------------
    # Mean latent activation
    # --------------------------------------------------------

    rho_hat = torch.mean(
        z,
        dim=0
    )


    # --------------------------------------------------------
    # KL sparsity loss
    # --------------------------------------------------------

    sparsity_loss = kl_divergence(
        RHO,
        rho_hat
    )


    # --------------------------------------------------------
    # L2 regularization
    # --------------------------------------------------------

    l2_loss = torch.tensor(
        0.0,
        dtype=torch.float32
    )

    for parameter in model.parameters():

        l2_loss += torch.sum(
            parameter ** 2
        )


    # --------------------------------------------------------
    # TOTAL LOSS
    # --------------------------------------------------------

    total_loss = (

        reconstruction_loss

        +

        BETA
        *
        sparsity_loss

        +

        L2_WEIGHT
        *
        l2_loss

    )


    return (

        total_loss,
        reconstruction_loss,
        sparsity_loss,
        l2_loss,
        rho_hat

    )


# ============================================================
# L-BFGS TRAINING
# ============================================================

print(
    "\n========== TRAINING SAE =========="
)

optimizer = optim.LBFGS(

    model.parameters(),

    lr=1.0,

    max_iter=MAX_ITER,

    tolerance_grad=1e-7,

    tolerance_change=1e-9,

    history_size=100,

    line_search_fn="strong_wolfe"

)


closure_calls = 0


def closure():

    global closure_calls

    optimizer.zero_grad()


    (
        total_loss,
        reconstruction_loss,
        sparsity_loss,
        l2_loss,
        rho_hat

    ) = calculate_loss(
        X_train_tensor
    )


    total_loss.backward()

    closure_calls += 1

    return total_loss


optimizer.step(
    closure
)


# ============================================================
# FINAL SAE RESULT
# ============================================================

with torch.no_grad():

    (
        total_loss,
        reconstruction_loss,
        sparsity_loss,
        l2_loss,
        rho_hat

    ) = calculate_loss(
        X_train_tensor
    )


print(
    "\n========== SAE RESULT =========="
)

print(
    "Input dimension       :",
    input_dim
)

print(
    "Hidden dimension 1    :",
    HIDDEN_1
)

print(
    "Hidden dimension 2    :",
    HIDDEN_2
)

print(
    "Latent dimension      :",
    LATENT_DIM
)

print(
    "Rho target            :",
    RHO
)

print(
    "Beta sparsity         :",
    BETA
)

print(
    "Final total loss      :",
    f"{total_loss.item():.6f}"
)

print(
    "Reconstruction MSE    :",
    f"{reconstruction_loss.item():.6f}"
)

print(
    "L2 loss               :",
    f"{l2_loss.item():.6f}"
)

print(
    "KL sparsity loss      :",
    f"{sparsity_loss.item():.6f}"
)

print(
    "Mean latent activation:"
)

print(
    np.round(
        rho_hat.cpu().numpy(),
        4
    )
)

print(
    "L-BFGS closure calls  :",
    closure_calls
)


# ============================================================
# EXTRACT LATENT FEATURES
# ============================================================

print(
    "\n========== EXTRACTING LATENT FEATURES =========="
)

model.eval()

with torch.no_grad():

    Z_train = model.encode(
        X_train_tensor
    ).cpu().numpy()

    Z_val = model.encode(
        X_val_tensor
    ).cpu().numpy()

    Z_test = model.encode(
        X_test_tensor
    ).cpu().numpy()


print(
    "Original dimension:",
    X_train.shape[1]
)

print(
    "Latent dimension:",
    Z_train.shape[1]
)

print(
    "Training latent shape:",
    Z_train.shape
)

print(
    "Validation latent shape:",
    Z_val.shape
)

print(
    "Test latent shape:",
    Z_test.shape
)


# ============================================================
# LATENT FEATURE STATISTICS
# ============================================================

print(
    "\n========== LATENT FEATURE STATISTICS =========="
)

print(
    "Training latent mean:"
)

print(
    np.round(
        Z_train.mean(axis=0),
        4
    )
)

print(
    "Training latent std:"
)

print(
    np.round(
        Z_train.std(axis=0),
        4
    )
)


# ============================================================
# CHECK FOR LATENT COLLAPSE
# ============================================================

print(
    "\n========== LATENT COLLAPSE CHECK =========="
)

latent_std = Z_train.std(
    axis=0
)

collapsed = (
    latent_std < 0.01
)

print(
    "Latent standard deviations:"
)

print(
    np.round(
        latent_std,
        6
    )
)

print(
    "Collapsed latent neurons:",
    int(collapsed.sum()),
    "/",
    LATENT_DIM
)

if collapsed.sum() == 0:

    print(
        "STATUS: No obvious latent collapse."
    )

else:

    print(
        "STATUS: Some latent neurons may be collapsed."
    )


# ============================================================
# SAVE
# ============================================================

np.savez(

    "secom_sae_features.npz",

    X_train=X_train,

    X_val=X_val,

    X_test=X_test,

    Z_train=Z_train,

    Z_val=Z_val,

    Z_test=Z_test,

    y_train=y_train,

    y_val=y_val,

    y_test=y_test

)


print(
    "\nSaved:"
)

print(
    "secom_sae_features.npz"
)


# ============================================================
# COMPLETE
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "SECOM SAE EXPERIMENT COMPLETE"
)

print(
    "=" * 70
)