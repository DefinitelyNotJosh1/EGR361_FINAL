import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.formula.api import ols



# Load the dataset
df = pd.read_csv('f1_pitstops_2018_2024.csv')

# Grab relevant columns
x_cols = ['Air_Temp_C', 'Track_Temp_C', 'Humidity_%', 'Wind_Speed_KMH']
x_variables = df[x_cols]
label = df['AvgPitStopTime']


# Clean data
df = df.dropna(subset=x_cols)
df = df.dropna(subset=['AvgPitStopTime'])


# Describe the data
print(label.describe())
print(x_variables.describe())

# Check for normality of the data
for col in x_cols:
    plt.figure(figsize=(10, 6))
    plt.hist(df[col], bins=30, edgecolor='black')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

# Check for normality of the label
plt.figure(figsize=(10, 6))
plt.hist(label, bins=30, edgecolor='black')
plt.title('Average Pit Stop Time')
plt.xlabel('Average Pit Stop Time (seconds)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()


print("Correlation Results:")
print(df[['AvgPitStopTime'] + x_cols].corr())