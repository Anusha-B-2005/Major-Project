# ================================
# DATA AUGMENTATION WITH CTGAN
# ================================

import sys

if len(sys.argv) < 2:
    raise ValueError("Dataset path not provided. Please pass CSV file path.")

# STEP 0: IMPORT LIBRARIES
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# STEP 1: LOAD DATA
data_path = sys.argv[1]
df = pd.read_csv(data_path)

print(df.head())
print(df.info())

# STEP 2: BASIC CLEANING
df = df.drop_duplicates()

categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].astype(str)

# STEP 3: SIMPLE IMPUTATION
def simple_gain_impute(data):
    data = data.copy()
    for col in data.columns:
        if data[col].isnull().sum() > 0:
            if data[col].dtype == 'object':
                data[col] = data[col].fillna(data[col].mode()[0])
            else:
                data[col] = data[col].fillna(data[col].median())
    return data

df_imputed = simple_gain_impute(df)

# STEP 4: PARTIAL DATA
partial_df = df_imputed.sample(frac=0.3, random_state=42)

# STEP 5: METADATA
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(df_imputed)

# STEP 6: CTGAN
ctgan = CTGANSynthesizer(
    metadata,
    epochs=1000,
    batch_size=200,
    verbose=True
)
#ctgan.fit(partial_df)
ctgan.fit(df_imputed)

# STEP 7: SYNTHETIC DATA
synthetic_df = ctgan.sample(num_rows=500)

# STEP 7.1: CLIP NUMERIC COLUMNS
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    synthetic_df[col] = synthetic_df[col].clip(
        lower=df[col].min(),
        upper=df[col].max()
    )
    if df[col].dtype == 'int64':
        synthetic_df[col] = synthetic_df[col].round().astype(int)

# STEP 7.5: COMBINE DATA
augmented_df = pd.concat([df_imputed, synthetic_df], ignore_index=True)

# --------------------------------------------------
# 🔹 FIX: HANDLE REPEATING PRIMARY-KEY-LIKE COLUMNS
# --------------------------------------------------
#for col in df.columns:
    # if column was fully unique in original dataset
    #if df[col].nunique() == len(df):
        #augmented_df[col] = range(1, len(augmented_df) + 1)


# --------------------------------------------------
# 🔹 FIX: HANDLE ONLY NUMERIC PRIMARY KEYS
# --------------------------------------------------
for col in df.columns:
    # primary key must be unique AND numeric
    if (
        df[col].nunique() == len(df)
        and pd.api.types.is_numeric_dtype(df[col])
    ):
        start = df[col].max() + 1
        augmented_df.loc[df_imputed.shape[0]:, col] = range(
            start, start + len(synthetic_df)
        )




# STEP 8: QUALITY CHECK
if len(numeric_cols) > 0:
    numeric_col = numeric_cols[0]
    p_value, ks_stat = ks_2samp(
        df[numeric_col].dropna(),
        synthetic_df[numeric_col]
    )
    print("KS Test p-value:", p_value)
else:
    print("KS Test skipped (no numeric columns)")

# STEP 9: OPTIONAL ML VALIDATION
TARGET = "target"

if TARGET in augmented_df.columns:
    X = augmented_df.drop(TARGET, axis=1)
    y = augmented_df[TARGET]

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    print("Model Accuracy:", accuracy_score(y_test, model.predict(X_test)))
else:
    print("ML validation skipped (no target column)")

# STEP 10: PRIVACY CHECK
duplicates = synthetic_df.merge(df, how="inner")
print("Exact duplicate rows found:", len(duplicates))

# STEP 11: SAVE OUTPUT
augmented_df.to_csv("augmented_dataset.csv", index=False)
print("Augmented dataset saved successfully.")
