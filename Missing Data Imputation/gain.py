import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate

# =========================
# PARAMETERS
# =========================
EVAL_HOLDOUT_FRACTION = 0.2
GAIN_EPOCHS = 200
BATCH_SIZE = 64
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# =========================
# LOAD + ONE HOT
# =========================
def load_and_ohe(path):
    df = pd.read_csv(path)
    df_true = df.copy()
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df_ohe = pd.get_dummies(df, columns=categorical_cols)
    return df_true, df_ohe, categorical_cols

# =========================
# PREPARE DATA
# =========================
def prepare_data_for_gain(df_ohe, eval_fraction=0.2):
    arr = df_ohe.values.astype(float)
    mask = ~np.isnan(arr)

    total_available = mask.sum()
    n_holdout = int(total_available * eval_fraction)

    eval_mask = np.zeros_like(mask, dtype=bool)
    if n_holdout > 0:
        idx = np.argwhere(mask)
        hold = idx[np.random.choice(len(idx), n_holdout, replace=False)]
        eval_mask[hold[:, 0], hold[:, 1]] = True

    train_mask = mask.copy()
    train_mask[eval_mask] = False

    imputer = SimpleImputer(strategy="mean")
    arr_imputed = imputer.fit_transform(arr)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(arr_imputed).astype(np.float32)

    X_train = scaled.copy()
    X_train[~train_mask] = 0.0

    return (
        X_train,
        train_mask.astype(np.float32),
        scaler,
        scaled,
        eval_mask,
        df_ohe.columns.tolist()
    )

# =========================
# BUILD GAIN
# =========================
def build_gain(input_dim):
    X = Input(shape=(input_dim,))
    M = Input(shape=(input_dim,))

    G_in = Concatenate()([X, M])
    g = Dense(128, activation="relu")(G_in)
    g = Dense(128, activation="relu")(g)
    G_out = Dense(input_dim, activation="sigmoid")(g)

    D_in = Concatenate()([X, M])
    d = Dense(128, activation="relu")(D_in)
    d = Dense(128, activation="relu")(d)
    D_out = Dense(input_dim, activation="sigmoid")(d)

    generator = Model([X, M], G_out)
    discriminator = Model([X, M], D_out)

    generator.compile(optimizer="adam", loss="mse")
    discriminator.compile(optimizer="adam", loss="binary_crossentropy")

    return generator, discriminator

# =========================
# TRAIN
# =========================
def train_gain(generator, discriminator, X, M):
    n = X.shape[0]
    for _ in range(GAIN_EPOCHS):
        idx = np.random.permutation(n)
        for i in range(0, n, BATCH_SIZE):
            b = idx[i:i+BATCH_SIZE]
            xb, mb = X[b], M[b]
            generator.train_on_batch([xb, mb], mb)
            discriminator.train_on_batch([xb, mb], mb)

# =========================
# IMPUTE ONLY MISSING
# =========================
def impute(generator, X, M, scaler, columns, categorical_cols=None):
    g_pred = generator.predict([X, M], verbose=0)
    imputed_scaled = M * X + (1 - M) * g_pred
    imputed_original = scaler.inverse_transform(imputed_scaled)
    df = pd.DataFrame(imputed_original, columns=columns)

    # =========================
    # FIX CATEGORICAL COLUMNS TO SINGLE VALUE
    # =========================
    if categorical_cols is not None:
        for col in categorical_cols:
            one_hot_cols = [c for c in columns if c.startswith(col + "_")]
            if not one_hot_cols:
                continue

            # Pick the max value in one-hot for each row
            max_idx = df[one_hot_cols].values.argmax(axis=1)
            # Replace one-hot columns with single label
            df[col] = [one_hot_cols[i].replace(col + "_", "") for i in max_idx]
            # Drop the old one-hot columns
            df.drop(columns=one_hot_cols, inplace=True)

    return df
