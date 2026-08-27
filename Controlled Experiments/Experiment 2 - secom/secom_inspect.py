import pandas as pd
import numpy as np

# ============================================================
# LOAD
# ============================================================

DATA_FILE = "secom.data"

df = pd.read_csv(
    DATA_FILE,
    sep=r"\s+",
    header=None
)

print("\n========== SECOM DATASET ==========")

print("Shape:", df.shape)


# ============================================================
# FIRST ROWS
# ============================================================

print("\n========== FIRST 5 ROWS ==========")

print(df.head())


# ============================================================
# DATA TYPES
# ============================================================

print("\n========== DATA TYPES ==========")

print(
    df.dtypes.value_counts()
)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n========== MISSING VALUES ==========")

missing = df.isna().sum()

print(
    "Total missing values:",
    missing.sum()
)

print(
    "Columns containing missing values:",
    (missing > 0).sum()
)


# ============================================================
# MISSING VALUE PERCENTAGE
# ============================================================

missing_percentage = (

    df.isna().mean()
    * 100

)

print(
    "\n========== MISSING VALUE RANGE =========="
)

print(
    "Minimum:",
    round(
        missing_percentage.min(),
        2
    ),
    "%"
)

print(
    "Maximum:",
    round(
        missing_percentage.max(),
        2
    ),
    "%"
)

print(
    "Mean:",
    round(
        missing_percentage.mean(),
        2
    ),
    "%"
)


# ============================================================
# CONSTANT FEATURES
# ============================================================

nunique = df.nunique(
    dropna=False
)

constant_columns = (

    nunique <= 1

)

print(
    "\n========== CONSTANT FEATURES =========="
)

print(
    "Number of constant columns:",
    constant_columns.sum()
)


# ============================================================
# DUPLICATE COLUMNS
# ============================================================

print(
    "\n========== DUPLICATE COLUMNS =========="
)

duplicate_count = (

    df.T
    .duplicated()
    .sum()

)

print(
    "Duplicate columns:",
    duplicate_count
)


# ============================================================
# STATISTICS
# ============================================================

print(
    "\n========== NUMERICAL STATISTICS =========="
)

print(
    df.describe()
)


# ============================================================
# IMPORTANT:
# SECOM LABEL FILE
# ============================================================

print(
    "\n========== EXPECTED SECOM FILES =========="
)

print(
    "secom.data  -> measurements"
)

print(
    "secom_labels.data -> labels"
)


print(
    "\n========== INSPECTION COMPLETE =========="
)