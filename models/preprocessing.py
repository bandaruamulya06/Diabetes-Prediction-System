# ==========================================
# IMPORT LIBRARIES
# ==========================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)

from sklearn.model_selection import (
    train_test_split
)

import joblib

# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv(
    "diabetes_prediction_dataset.csv"
)

# ==========================================
# DISPLAY BASIC INFO
# ==========================================

print("\nDATASET HEAD\n")
print(data.head())

print("\nDATASET INFO\n")
print(data.info())

print("\nMISSING VALUES\n")
print(data.isnull().sum())

# ==========================================
# REMOVE DUPLICATES
# ==========================================

data = data.drop_duplicates()

print("\nDUPLICATES REMOVED\n")

# ==========================================
# LABEL ENCODING
# ==========================================

label_encoders = {}

categorical_columns = [
    'gender',
    'smoking_history'
]

for column in categorical_columns:

    encoder = LabelEncoder()

    data[column] = encoder.fit_transform(
        data[column]
    )

    label_encoders[column] = encoder

# ==========================================
# FEATURES AND TARGET
# ==========================================

X = data.drop(
    'diabetes',
    axis=1
)

y = data['diabetes']

# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(

    X_scaled,
    y,

    test_size=0.2,

    random_state=42,

    stratify=y
)

# ==========================================
# SAVE SCALER
# ==========================================

joblib.dump(
    scaler,
    "scaler.pkl"
)

# ==========================================
# SAVE LABEL ENCODERS
# ==========================================

joblib.dump(
    label_encoders,
    "label_encoders.pkl"
)

# ==========================================
# SAVE PREPROCESSED DATA
# ==========================================

np.save(
    "X_train.npy",
    X_train
)

np.save(
    "X_test.npy",
    X_test
)

np.save(
    "y_train.npy",
    y_train
)

np.save(
    "y_test.npy",
    y_test
)

# ==========================================
# FINAL OUTPUT
# ==========================================

print("\nPREPROCESSING COMPLETED SUCCESSFULLY")

print("\nTRAIN SHAPE :", X_train.shape)

print("TEST SHAPE :", X_test.shape)