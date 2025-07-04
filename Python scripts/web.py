
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, roc_auc_score
from google import genai
# import google.generativeai as genai

"""API KEY"""
apikey = "AIzaSyC0BM4NLMLldcPvBlIqwe49xnM2qzYjfJs"
client = genai.Client(api_key=apikey)

"""
Logistic regression prerequisites: 
"""
dataset = pd.read_csv('water_potability.csv', on_bad_lines="skip")

d2 = dataset[[ 'ph', 'Hardness', 'Conductivity', 'Turbidity', 'Chloramines', 'Potability']]
d2 = d2.dropna()
#     print(d2.head(9))
# print(d2.describe())
X = d2[['ph', 'Hardness', 'Conductivity', 'Turbidity', 'Chloramines']]
Y = d2[['Potability']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)
#training
clf = linear_model.LogisticRegression(C=10)#Regularisation factor, higher = more complex boundaries will be considered by the model
clf.fit(x_train, y_train)
#predicting the entire x_test
y_pred = clf.predict(x_test)
print("Coeffs: ", clf.coef_)
print("Intercept: ", clf.intercept_)


""" NN model prereq: """
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
# dataset = pd.read_csv('water_potability.csv', on_bad_lines="skip")
# d2 = dataset[[ 'ph', 'Hardness', 'Conductivity', 'Turbidity', 'Chloramines', 'Potability']]

# d2 = d2.dropna()
# #     print(d2.head(9))
# # print(d2.describe())

# X = d2[['ph', 'Hardness', 'Conductivity', 'Turbidity', 'Chloramines']]
# Y = d2[['Potability']]
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
# x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# model = keras.Sequential([
#     layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
#     layers.Dense(128, activation='relu'),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(1, activation='sigmoid')
# ])

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs = 50, batch_size = 32, validation_data=(x_test, y_test), verbose=1)

# y_pred_prob = model.predict(x_test)
# ypred = (y_pred_prob > 0.5).astype(int)
# x_test = scaler.transform(x_test)

"""dec tree prerequisites"""
X = d2[['ph', 'Hardness', 'Conductivity', 'Turbidity', 'Chloramines']]
Y = d2[['Potability']].values.ravel() #converts dataframe to 1D array
#.values converts to 2D array
#.ravel converts to 1D array

scaler = StandardScaler()
X = scaler.fit_transform(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

def trainusinggini(x_train, x_test, y_train):
    clf_gini = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=3, min_samples_leaf=5)
    clf_gini.fit(x_train, y_train)
    return clf_gini

def trainusingentropy(x_train, x_test, y_train):
    clf_entropy = DecisionTreeClassifier(criterion='entropy', random_state=42, max_depth=3, min_samples_leaf=5)
    clf_entropy.fit(x_train, y_train)
    return clf_entropy

def prediction(x_test, clf_object):
    ypred = clf_object.predict(x_test)
    print("Predicted 1: ", ypred)
    return ypred


clf_gini = trainusinggini(x_train, x_test, y_train)
ypredgini = prediction(x_test, clf_gini)

clf_ent = trainusingentropy(x_train, x_test, y_train)
ypredent = prediction(x_test, clf_ent)

# App Declaration
app = Flask(__name__)
CORS(app)

#Logistic regression
@app.route('/logreg', methods=['POST'])
def log_reg():
    data = request.json
    ph = data.get("ph")
    hardness = data.get("hardness")
    chloramines = data.get("chloramines")
    conductivity = data.get("conductivity")
    turbidity = data.get("turbidity")

    sample = np.array([[float(ph), float(hardness), float(conductivity), float(turbidity), float(chloramines)]])

    response = f'Potability= {clf.predict(sample)}'
    return jsonify({"message": response})

#Neural Networks
@app.route('/nn', methods=['POST'])
def nn():
    data = request.json
    ph = data.get("ph")
    hardness = data.get("hardness")
    chloramines = data.get("chloramines")
    conductivity = data.get("conductivity")
    turbidity = data.get("turbidity")
    """
    Test Accuracy: 0.5889
    [[252  81]
    [153  71]]
    """
    sample = np.array([[float(ph), float(hardness), float(conductivity), float(turbidity), float(chloramines)]])
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, roc_auc_score
    response = f'Potability= {model.predict(sample)}, \nNot reliable enough with confusion matrix: [[252  81][153  71]] and test accuracy: 0.5889'
    return jsonify({"message": response})

#Decision Tree
@app.route('/dectree', methods=['POST'])
def dectree():
    data = request.json
    ph = data.get("ph")
    hardness = data.get("hardness")
    chloramines = data.get("chloramines")
    conductivity = data.get("conductivity")
    turbidity = data.get("turbidity")
    sample = np.array([[float(ph), float(hardness), float(conductivity), float(turbidity), float(chloramines)]])
    response = clf_gini.predict(sample)
    
    return jsonify({"message": response})

#GenAI
@app.route('/genai', methods=["POST"])
def ai():
    data = request.json
    ph = data.get("ph")
    hardness = data.get("hardness")
    chloramines = data.get("chloramines")
    conductivity = data.get("conductivity")
    turbidity = data.get("turbidity")

    prompt = (
        "From the given data, calculate whether the water is potable or not. Also check whether the conditions are suitable for contamination by a pathogen, and specify which one."
        "Respond with a [0] if it is not, and [1] if it is. "
        "Give a two-sentence reason for your answer, that judges each parameter.\n"
        f"Data:\n- pH: {ph}\n- Chloramines (ppm): {chloramines}\n"
        f"- Turbidity (NTU): {turbidity}\n- Conductivity (µS/cm): {conductivity}\n"
        f"- Hardness (mg/L): {hardness}"
    )

    response = client.models.generate_content(model='gemini-2.0-flash', contents = prompt)

    return jsonify({"message": response.text})

#Location based GenAI
@app.route('/location', methods=['POST'])
def locationai():
    data = request.json
    features = {key: (float(value) if value != "N/A" else None) for key, value in data.items()}
    if None in features.values():
        return jsonify({"error": "Missing or invalid values in input data"}), 400
    text = (
        f"From the given data, calculate whether the water is potable or not. "
    "Respond with a [0] if it is not, and [1] if it is. "
    "Give a two to three sentence reason for your answer, that judges each parameter and comment on potential pathogens that can/might be present in the water, giving preference to the most serious one.\n"
    f"Ensure that the answer doesn't go beyond three lines, and ensure accuracy in your testing, leaning slightly towards potable than non-potable. Ensure that your answer includes possibility of potential pathogens existing.\n"
    f"If you think that fecal or total coliform could render the water unhealthy, know that it will be treated in a water treatment plant later."
        f"- pH: {features['pH']}\n"
        f"- BOD: {features['BOD']} mg/L\n"
        f"- DO: {features['DO']} mg/L\n"
        f"- Nitrate: {features['Nitrate']} mg/L\n"
        f"- Mean temperature: {features['Temperature_Mean']}\n"
        f"- Mean Conductivity: {features['Conductivity_Mean']}\n"
        f"- Fecal Coliform: {features['Fecal_Coliform']}\n"
        f"- Total Coliform: {features['Total_Coliform']}\n"
        f"If a value is -1, data is not present and do not consider it."
    )
    response = client.models.generate_content(model='gemini-2.0-flash', contents = text)
    return jsonify({"potability": response.text})

#Port Assignment
if __name__ == '__main__':
    app.run(port=5000, debug=True)
