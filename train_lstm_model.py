import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ============================
# DATASET PATH
# ============================
DATASET_PATH = r"C:\Users\VIAMRSH P S\Desktop\Projects-20240722T093004Z-001\Projects\fraud_detection\fraud_detection\dataset\data"

# ============================
# LOAD DATA
# ============================
dfs = []
for file in os.listdir(DATASET_PATH):
    if file.endswith(".pkl"):
        dfs.append(pd.read_pickle(os.path.join(DATASET_PATH, file)))

df = pd.concat(dfs, ignore_index=True)

df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
df = df.sort_values(['CUSTOMER_ID', 'TX_DATETIME'])

print("Total transactions:", len(df))

# ============================
# FEATURE ENGINEERING
# ============================
df['HIGH_AMOUNT_FLAG'] = (df['TX_AMOUNT'] > 220).astype(int)

features = ['TX_AMOUNT', 'HIGH_AMOUNT_FLAG']
target = 'TX_FRAUD'

# ============================
# NORMALIZATION
# ============================
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# ============================
# SEQUENCE CREATION
# ============================
SEQUENCE_LENGTH = 10
X, y = [], []

for _, group in df.groupby('CUSTOMER_ID'):
    values = group[features].values
    labels = group[target].values

    if len(values) < SEQUENCE_LENGTH + 1:
        continue

    for i in range(len(values) - SEQUENCE_LENGTH):
        X.append(values[i:i + SEQUENCE_LENGTH])
        y.append(labels[i + SEQUENCE_LENGTH])

X = np.array(X)
y = np.array(y)

print("Sequence shape:", X.shape)

# ============================
# TRAIN TEST SPLIT
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================
# LSTM MODEL
# ============================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, X.shape[2])),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# ============================
# TRAIN MODEL
# ============================
model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=512,
    callbacks=[early_stop],
    verbose=1
)

# ============================
# EVALUATION
# ============================
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob > 0.5).astype(int)

print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# ============================
# SAVE MODEL (TF 2.16 FORMAT)
# ============================
model.save("lstm_fraud_model.keras")
print("✅ Model saved as lstm_fraud_model.keras")
