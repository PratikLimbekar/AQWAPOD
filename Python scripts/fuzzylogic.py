#Entropy Calculation is useful when the values are grouped
#Pearson Correlation Coefficient is useful to find LINEAR correlation
#PCC close to 1 -> strong positive corr, 0 -> no relation and close to -1 indicates strong negative correlation
#Fuzzy Logic laga ke rakhte apan

"""Pearson Result:
PearsonRResult(statistic=array([-0.01383656]), pvalue=array([0.42854159]))
PearsonRResult(statistic=array([nan]), pvalue=array([nan]))
PearsonRResult(statistic=array([0.03374331]), pvalue=array([0.05346364]))     
PearsonRResult(statistic=array([0.02377897]), pvalue=array([0.17361022]))     
PearsonRResult(statistic=array([nan]), pvalue=array([nan]))
PearsonRResult(statistic=array([-0.00812832]), pvalue=array([0.64188455]))    
PearsonRResult(statistic=array([nan]), pvalue=array([nan]))
PearsonRResult(statistic=array([0.00158068]), pvalue=array([0.92793916])) 
"""



import pandas as pd

df = pd.read_csv('water_potability.csv')

df2 = df[['Hardness', 'ph','Solids','Chloramines','Sulfate','Conductivity','Trihalomethanes','Turbidity']]
def entropycalc(df):
    dfmin = df.min()
    dfmax = df.max()

    value_counts = df.value_counts(normalize=True)

    P = (df - dfmin)/(dfmax  - dfmin)
    Hx = - np.sum(value_counts * np.log2(value_counts))
    return Hx

#print(df2['Hardness'])

#Pearson Correlation:
from scipy.stats import pearsonr

hardness = df[['Hardness']]
ph = df[['ph']]
solids = df[['Solids']]
chloramines = df[['Chloramines']]
sulfates = df[['Sulfate']]
conductivity = df[['Conductivity']]
trihalo = df[['Trihalomethanes']]
turbidity = df[['Turbidity']]
pota = df[['Potability']]

def correlation(col1, col2):
    return pearsonr(col1,col2)

print(correlation(hardness, pota))
print(correlation(ph, pota))
print(correlation(solids, pota))
print(correlation(chloramines, pota))
print(correlation(sulfates, pota))
print(correlation(conductivity, pota))
print(correlation(trihalo, pota))
print(correlation(turbidity, pota))
