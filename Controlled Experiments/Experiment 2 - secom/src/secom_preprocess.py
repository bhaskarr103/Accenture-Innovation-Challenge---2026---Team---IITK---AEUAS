import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ============================================================
# FILES
# ============================================================

DATA_FILE = "secom.data"
LABEL_FILE = "secom_labels.data"


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

print(
    "Label shape:",
    y.shape
)


# ============================================================
# LABEL CONVERSION
# ============================================================

# SECOM:
# -1 = normal
# +1 = failure
#
# Convert to:
#  0 = normal
#  1 = failure

y = np.where(
    y == 1,
    1,
    0
)

print(
    "\n========== TARGET =========="
)

print(
    "Normal:",
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
# CHECK SAMPLE ALIGNMENT
# ============================================================

assert len(X) == len(y), (
    "Feature and label sample counts do not match."
)

print(
    "\nSample alignment: OK"
)


# ============================================================
# REMOVE DUPLICATE FEATURES
# ============================================================

print(
    "\n========== DUPLICATE FEATURES =========="
)

duplicate_mask = X.T.duplicated()

duplicate_count = duplicate_mask.sum()

print(
    "Duplicate columns found:",
    duplicate_count
)

X = X.loc[
    :,
    ~duplicate_mask
]

print(
    "Shape after duplicate removal:",
    X.shape
)


# ============================================================
# MISSING VALUE ANALYSIS
# ============================================================

print(
    "\n========== MISSING VALUE FILTER =========="
)

missing_fraction = X.isna().mean()

MISSING_THRESHOLD = 0.50

remove_missing = (
    missing_fraction
    >
    MISSING_THRESHOLD
)

print(
    "Features with >50% missing:",
    remove_missing.sum()
)

X = X.loc[
    :,
    ~remove_missing
]

print(
    "Shape after missing-value filtering:",
    X.shape
)


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print(
    "\n========== DATA SPLIT =========="
)

X_train, X_temp, y_train, y_temp = train_test_split(

    X,
    y,

    test_size=0.30,

    stratify=y,

    random_state=42

)

X_val, X_test, y_val, y_test = train_test_split(

    X_temp,
    y_temp,

    test_size=0.50,

    stratify=y_temp,

    random_state=42

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
# CLASS DISTRIBUTION
# ============================================================

print(
    "\n========== SPLIT DISTRIBUTION =========="
)

for name, labels_split in [

    ("Training", y_train),
    ("Validation", y_val),
    ("Test", y_test)

]:

    normal = np.sum(
        labels_split == 0
    )

    failure = np.sum(
        labels_split == 1
    )

    print(
        f"{name:<12}"
        f"Normal={normal:<5}"
        f"Failure={failure:<5}"
        f"Failure rate={failure / len(labels_split) * 100:.2f}%"
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

X_train_imp = imputer.fit_transform(
    X_train
)

X_val_imp = imputer.transform(
    X_val
)

X_test_imp = imputer.transform(
    X_test
)

print(
    "Remaining NaNs:"
)

print(
    "Training  :",
    np.isnan(X_train_imp).sum()
)

print(
    "Validation:",
    np.isnan(X_val_imp).sum()
)

print(
    "Test      :",
    np.isnan(X_test_imp).sum()
)


# ============================================================
# STANDARDIZATION
# ============================================================

print(
    "\n========== STANDARDIZATION =========="
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train_imp
)

X_val_scaled = scaler.transform(
    X_val_imp
)

X_test_scaled = scaler.transform(
    X_test_imp
)

print(
    "Training mean:",
    np.round(
        X_train_scaled.mean(axis=0)[:10],
        4
    )
)

print(
    "Training std:",
    np.round(
        X_train_scaled.std(axis=0)[:10],
        4
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "SECOM PREPROCESSING COMPLETE"
)

print(
    "=" * 70
)

print(
    "Raw features:",
    590
)

print(
    "After duplicate removal:",
    590 - duplicate_count
)

print(
    "Final features:",
    X_train_scaled.shape[1]
)

print(
    "Training samples:",
    X_train_scaled.shape[0]
)

print(
    "Validation samples:",
    X_val_scaled.shape[0]
)

print(
    "Test samples:",
    X_test_scaled.shape[0]
)

print(
    "Final input dimension:",
    X_train_scaled.shape[1]
)

print(
    "=" * 70
)