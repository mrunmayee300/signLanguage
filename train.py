import pandas as pd
import numpy as np
import tensorflow as tf
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load dataset
df = pd.read_csv("signLang\sign_language_data.csv")

# Convert labels to numbers
labels = df["Label"].astype("category").cat.codes
features = df.drop("Label", axis=1).values

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define model
model = Sequential([
    Dense(128, activation='relu', input_shape=(42,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(len(df["Label"].unique()), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save model
model.save("sign_language_model.h5")
print("Model saved!")
