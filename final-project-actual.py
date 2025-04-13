import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Load the dataset
# df = pd.read_csv('f1_pitstops_2018_2024.csv')

# # Grab relevant columns
# x_cols = ['Air_Temp_C', 'Track_Temp_C', 'Humidity_%', 'Wind_Speed_KMH']
# x_variables = df[x_cols]
# label = df['AvgPitStopTime']


# # Clean data
# df = df.dropna(subset=x_cols)
# df = df.dropna(subset=['AvgPitStopTime'])

# df.to_csv('f1_pitstops_2018_2024_cleaned.csv', index=False)

# Load the cleaned dataset
df = pd.read_csv('f1_pitstops_2018_2024_cleaned.csv')

# Grab relevant columns
x_cols = ['Track_Temp_C', 'Humidity_%', 'Wind_Speed_KMH']
x_variables = df[x_cols]
label = df['AvgPitStopTime']

# Describe the data
# print(label.describe())
# print(x_variables.describe())

# # Check for normality of the data
# for col in x_cols:
#     plt.figure(figsize=(10, 6))
#     plt.hist(df[col], bins=30, edgecolor='black')
#     plt.title(f'Distribution of {col}')
#     plt.xlabel(col)
#     plt.ylabel('Frequency')
#     plt.grid(axis='y', alpha=0.75)
#     plt.show()

# # Check for normality of the label
# plt.figure(figsize=(10, 6))
# plt.hist(label, bins=30, edgecolor='black')
# plt.title('Average Pit Stop Time')
# plt.xlabel('Average Pit Stop Time (seconds)')
# plt.ylabel('Frequency')
# plt.grid(axis='y', alpha=0.75)
# plt.show()


# print("Correlation Results:")
# print(df[['AvgPitStopTime'] + x_cols].corr())

# Normalize the data
def min_max_normalize(df):
    return (df - df.min()) / (df.max() - df.min())

df_normalized = min_max_normalize(df[x_cols])
df_normalized['AvgPitStopTime'] = label

print("Normalized Data:")
print(df_normalized.describe())

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df_normalized[x_cols], df_normalized['AvgPitStopTime'], test_size=0.2, random_state=42)

print("Training Set Size:")
print(x_train.shape)
print("Testing Set Size:")
print(x_test.shape)

