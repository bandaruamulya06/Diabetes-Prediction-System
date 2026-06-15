# ==========================================
# IMPORT LIBRARIES
# ==========================================

import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib

from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    BatchNormalization
)

# ==========================================
# LOAD DATA
# ==========================================

X_train = np.load("X_train.npy")

X_test = np.load("X_test.npy")

# ==========================================
# REBUILD MODEL
# ==========================================

model = Sequential([

    Dense(64, activation='relu', input_shape=(8,)),
    BatchNormalization(),
    Dropout(0.45),

    Dense(32, activation='relu'),
    BatchNormalization(),
    Dropout(0.45),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(32, activation='relu'),

    Dense(1, activation='sigmoid')

])

# ==========================================
# LOAD TRAINED WEIGHTS
# ==========================================

model.load_weights(
    "weights.weights.h5"
)

# ==========================================
# CREATE SHAP EXPLAINER
# ==========================================

explainer = shap.DeepExplainer(

    model,

    X_train[:100]

)

# ==========================================
# GENERATE SHAP VALUES
# ==========================================

shap_values = explainer.shap_values(

    X_test[:50]
)

# ==========================================
# FEATURE NAMES
# ==========================================

feature_names = [

    'Gender',
    'Age',
    'Hypertension',
    'Heart Disease',
    'Smoking',
    'BMI',
    'HbA1c',
    'Glucose'

]

# ==========================================
# SHAP SUMMARY PLOT
# ==========================================

plt.figure(figsize=(10,6))

shap.summary_plot(

    shap_values[:, :, 0],

    X_test[:50],

    feature_names=feature_names,

    show=False

)

plt.tight_layout()

plt.savefig(
    "shap_summary.png"
)

print(

    "\nSHAP ANALYSIS COMPLETED"

)

print(

    "\nSHAP GRAPH SAVED AS shap_summary.png"

)