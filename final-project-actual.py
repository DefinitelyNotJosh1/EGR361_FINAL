import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# Load the dataset
df = pd.read_csv('f1_pitstops_2018_2024.csv')
# df = df[df['Constructor'] == 'Red Bull']

x_cols = ['Air_Temp_C', 'Track_Temp_C', 'Humidity_%', 'Wind_Speed_KMH']

df = df.dropna(subset=x_cols)
df = df.dropna(subset=['AvgPitStopTime'])

label = df['AvgPitStopTime']
print(label.describe())
x_variables = df[x_cols]

print(x_variables.describe())

plt.figure(figsize=(10, 6))
plt.hist(label, bins=30, edgecolor='black')
plt.title('Average Pit Stop Time')
plt.xlabel('Average Pit Stop Time (seconds)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()