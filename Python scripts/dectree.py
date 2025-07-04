from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

"""
GINI:
Confusion Matrix:  [[1819    0]
 [ 181    0]]
Accuracy:  0.9095
(True positive, False negative
False Positive, True negative)

Report:                precision    recall  f1-score   support

           0       0.91      1.00      0.95      1819
           1       0.00      0.00      0.00       181

    accuracy                           0.91      2000
   macro avg       0.45      0.50      0.48      2000
weighted avg       0.83      0.91      0.87      2000

Entropy:
Confusion Matrix:  [[1819    0]
 [ 181    0]]
Accuracy:  0.9095

Report:                precision    recall  f1-score   support

           0       0.91      1.00      0.95      1819
           1       0.00      0.00      0.00       181

    accuracy                           0.91      2000
   macro avg       0.45      0.50      0.48      2000
weighted avg       0.83      0.91      0.87      2000

"""

dataset = pd.read_csv('water_potability.csv', on_bad_lines="skip")
d2 = dataset[[ 'ph', 'Hardness', 'Conductivity', 'Turbidity', 'Chloramines', 'Potability']]
from sklearn.model_selection import train_test_split

#cleaning absent values
d2 = d2.dropna()
# print(d2.head(9))
# print(d2.describe())

#classifying the dataset
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

def cal_accuracy(y_test, ypred):
    print("Confusion Matrix: ", confusion_matrix(y_test, ypred))
    print('Accuracy: ',accuracy_score(y_test, ypred))
    print('Report: ', classification_report(y_test, ypred))

from sklearn import tree
def plotdectree(clf_object, featurename, classname):
    plt.figure(figsize=(15,10))
    tree.plot_tree(clf_object, filled=True, feature_names= featurename, class_names = classname, rounded=True)
    plt.show()

clf_gini = trainusinggini(x_train, x_test, y_train)
ypredgini = prediction(x_test, clf_gini)
print("GINI: ")
cal_accuracy(y_test, ypredgini)
print("Y using GINI: ", ypredgini)

clf_ent = trainusingentropy(x_train, x_test, y_train)
ypredent = prediction(x_test, clf_ent)
print("Entropy: ")
cal_accuracy(y_test, ypredent)
print("Y using Entropy: ", ypredent)

plotdectree(clf_gini, ['ph', 'Hardness', 'Conductivity', 'Turbidity', 'Chloramines'], ['0','1'])
plotdectree(clf_ent, ['ph', 'Hardness', 'Conductivity', 'Turbidity', 'Chloramines'], ['0','1'])