"""
Test Accuracy: 0.5885
[[ 987    0]
 [   0 1013]]
 """

from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

dataset = pd.read_csv('water_potability.csv', on_bad_lines="skip")
d2 = dataset[[ 'ph', 'Hardness', 'Conductivity', 'Turbidity', 'Chloramines', 'Potability']]
from sklearn.model_selection import train_test_split

#cleaning absent values
d2 = d2.dropna()
# print(d2.head(9))
# print(d2.describe())

#classifying the dataset
X = d2[['ph', 'Hardness', 'Conductivity', 'Turbidity', 'Chloramines']]
Y = d2[['Potability']]

scaler = StandardScaler()
X = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 50, batch_size = 32, validation_data=(x_test, y_test), verbose=1)

y_pred_prob = model.predict(x_test)
ypred = (y_pred_prob > 0.5).astype(int)
x_test = scaler.transform(x_test)

print(ypred[:5])
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
# print(ypred)

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, roc_auc_score
print(confusion_matrix(y_test, ypred))