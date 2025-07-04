# Append Pandas DataFrame to Existing CSV File
# importing pandas module
import pandas as pd

# data of Player and their performance
data = {
    'ph': [7.8],
    'hardness': [200],
    'chloramines': [400]

}

# Make data frame of above data
df = pd.DataFrame(data)

# append data frame to CSV file
df.to_csv('water_potability.csv', mode='a', index=False, header=False)

# print message
print("Data appended successfully.")

