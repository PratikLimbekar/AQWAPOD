# Mean Squared Error:  0.23385199695192568
# R2 Score:  -0.0010637131973569858
#MSE: Ideally close to 0
#R2 score: if = 1, then perfect model, if = 0, then model performs as well as predicting the mean
#if R2 score less than 0, then model is worse than predicting the mean

#Hence, Linear Regression cannot be used for such dataset.

#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#importing dataset
dataset = pd.read_csv('water_potability.csv', on_bad_lines="skip")
#print(dataset.head())

#choosing columns
d2 = dataset[[ 'Hardness', 'Conductivity', 'Turbidity', 'Chloramines', 'Potability']]

#cleaning absent values
d2 = d2.dropna()
print(d2.head(9))
print(d2.describe())

#classifying the dataset
X = d2[['Hardness', 'Conductivity', 'Turbidity', 'Chloramines']]
Y = d2[['Potability']]

#splitting into training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

#training
clf = linear_model.LinearRegression()
clf.fit(x_train, y_train)

#predicting the entire x_test
y_pred = clf.predict(x_test)
print("Coeffs: ", clf.coef_)
print("Intercept: ", clf.intercept_)

#comparing y_pred and actual y_test
acc = accuracy_score(y_test, y_pred) #used for classification, not regression
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("R2 Score: ", r2_score(y_test, y_pred))
# print("Accuracy Score: ", acc), only for classification

#Giving sample input
sample_input = np.array([[200,400,3,2]])
potability_pred = clf.predict(sample_input)
print("Predicted Potability: ", potability_pred)

