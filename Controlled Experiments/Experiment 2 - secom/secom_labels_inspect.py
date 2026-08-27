import pandas as pd

LABEL_FILE = "secom_labels.data"

labels = pd.read_csv(
    LABEL_FILE,
    sep=r"\s+",
    header=None
)

print("\n========== SECOM LABEL FILE ==========")

print("Shape:", labels.shape)

print("\n========== FIRST 10 ROWS ==========")

print(labels.head(10))

print("\n========== DATA TYPES ==========")

print(labels.dtypes)

print("\n========== UNIQUE VALUES ==========")

for column in labels.columns:

    print(
        f"\nColumn {column}:"
    )

    print(
        labels[column].value_counts(
            dropna=False
        )
    )

print(
    "\n========== LABEL INSPECTION COMPLETE =========="
)