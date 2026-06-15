# ==========================================
# IMPORT LIBRARIES
# ==========================================

import numpy as np

from keras.models import Sequential

from keras.layers import (
    Dense,
    Dropout,
    BatchNormalization
)

from keras.callbacks import (
    EarlyStopping
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# LOAD PREPROCESSED DATA
# ==========================================

X_train = np.load("X_train.npy")

X_test = np.load("X_test.npy")

y_train = np.load("y_train.npy")

y_test = np.load("y_test.npy")

# ==========================================
# BUILD NEURAL NETWORK
# ==========================================

model = Sequential([

    Dense(
        64,
        activation='relu',
        input_shape=(8,)
    ),

    BatchNormalization(),

    Dropout(0.25),

    Dense(
        32,
        activation='relu'
    ),

    BatchNormalization(),

    Dropout(0.25),

    Dense(
        64,
        activation='relu'
    ),

    BatchNormalization(),

    Dropout(0.30),

    Dense(
        32,
        activation='relu'
    ),

    Dense(
        1,
        activation='sigmoid'
    )

])

# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=['accuracy']

)

# ==========================================
# EARLY STOPPING
# ==========================================

early_stop = EarlyStopping(

    monitor='val_loss',

    patience=5,

    restore_best_weights=True

)

# ==========================================
# TRAIN MODEL
# ==========================================

history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    epochs=40,

    batch_size=64,

    callbacks=[early_stop],
    class_weight={
    0:1,
    1:2
    }

    verbose=1

)

# ==========================================
# MODEL EVALUATION
# ==========================================

predictions = model.predict(X_test)

predictions = (
    predictions > 0.35
).astype(int)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nMODEL ACCURACY\n")

print(accuracy)

print("\nCLASSIFICATION REPORT\n")

print(

    classification_report(
        y_test,
        predictions
    )

)

print("\nCONFUSION MATRIX\n")

print(

    confusion_matrix(
        y_test,
        predictions
    )

)

# ==========================================
# SAVE MODEL WEIGHTS
# ==========================================

model.save_weights(
    "weights.weights.h5"
)

print("\nMODEL TRAINING COMPLETED")