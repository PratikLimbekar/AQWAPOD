"""PLOTS
Heatmap:
              Hardness  Conductivity  Turbidity  Chloramines  Potability
Hardness      1.000000     -0.023915  -0.014449    -0.030054   -0.013837
Conductivity -0.023915      1.000000   0.005798    -0.020486   -0.008128
Turbidity    -0.014449      0.005798   1.000000     0.002363    0.001581
Chloramines  -0.030054     -0.020486   0.002363     1.000000    0.023779
Potability   -0.013837     -0.008128   0.001581     0.023779    1.000000

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv('water_potability.csv', on_bad_lines="skip")
d2 = dataset[[ 'ph', 'Hardness', 'Conductivity', 'Turbidity', 'Chloramines', 'Potability']]

#cleaning absent values
d2 = d2.dropna()
# print(d2.head(9))
# print(d2.describe())

#classifying the dataset
X = d2[['ph', 'Hardness', 'Conductivity', 'Turbidity', 'Chloramines']]
Y = d2[['Potability']]

# plt.scatter(d2['Hardness'], d2[['Conductivity']])
# plt.scatter(d2['Hardness'], d2[['Turbidity']])
# plt.scatter(d2['Hardness'], d2[['Chloramines']])
# plt.scatter(d2['Hardness'], d2[['Potability']])
# plt.scatter(d2['Turbidity'], d2[['Conductivity']])
# plt.scatter(d2['Chloramines'], d2[['Conductivity']])
# plt.show()

correlation = d2.corr()
print(correlation)

sns.heatmap(correlation, annot=True, cmap='YlGnBu')
plt.show()


"""Django (backend language) + MongoDB(database language)
React (Frontend)
"""