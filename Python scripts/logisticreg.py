#MSE: Ideally close to 0
#R2 score: if = 1, then perfect model, if = 0, then model performs as well as predicting the mean
#if R2 score less than 0, then model is worse than predicting the mean

"""
Implements Logistic Regression

Classification metrics: important wale
Coeffs:  [[-0.00194069 -0.00418881 -0.00232937 -0.2091081  -0.27075915]]
Intercept:  [1.17809973]
Mean Squared Error:  0.0935
R2 Score:  -0.1359529095884755
Accuracy Score:  0.9065
ROC AUC Score 0.5033258514331534
Classification Score:                precision    recall  f1-score   support

           0       0.91      1.00      0.95      1819
           1       0.20      0.01      0.02       181

    accuracy                           0.91      2000
   macro avg       0.56      0.50      0.49      2000
weighted avg       0.85      0.91      0.87      2000

Confusion Matrix:  [[1811    8]
 [ 179    2]]
"""

"""
Assumptions of Logistic Regression
We will explore the assumptions of logistic regression as understanding these assumptions is important to ensure that we are using appropriate application of the model. The assumption include:

Independent observations: Each observation is independent of the other. meaning there is no correlation between any input variables.
Binary dependent variables: It takes the assumption that the dependent variable must be binary or dichotomous, meaning it can take only two values. For more than two categories SoftMax functions are used.
***Linearity relationship between independent variables and log odds: The relationship between the independent variables and the log odds of the dependent variable should be linear.
No outliers: There should be no outliers in the dataset.
Large sample size: The sample size is sufficiently large

"""
#Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, roc_auc_score

#importing dataset
dataset = pd.read_csv('water_potability.csv', on_bad_lines="skip")

#choosing columns
d2 = dataset[[ 'ph', 'Hardness', 'Conductivity', 'Turbidity', 'Chloramines', 'Potability']]

#cleaning absent values
d2 = d2.dropna()
print(d2.head(9))
print(d2.describe())

#classifying the dataset
X = d2[['ph', 'Hardness', 'Conductivity', 'Turbidity', 'Chloramines']]
Y = d2[['Potability']]

#splitting into training and testing
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=42)

print(y_train.value_counts())

#training
clf = linear_model.LogisticRegression(C=10)#Regularisation factor, higher = more complex boundaries will be considered by the model
clf.fit(x_train, y_train)

#predicting the entire x_test
y_pred = clf.predict(x_test)
print("Coeffs: ", clf.coef_)
print("Intercept: ", clf.intercept_)

#comparing y_pred and actual y_test
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred))
print("R2 Score: ", r2_score(y_test, y_pred))
acc = accuracy_score(y_test, y_pred)
print("Accuracy Score: ", acc)
print("ROC AUC Score", roc_auc_score(y_test, y_pred))
print("Classification Score: ", classification_report(y_test, y_pred, zero_division=1))
print("Confusion Matrix: ", confusion_matrix(y_test, y_pred))

# #Giving sample input
# sample_input = np.array([[200,400,3,2]])
# potability_pred = clf.predict(sample_input)
# print("Predicted Potability: ", potability_pred)
