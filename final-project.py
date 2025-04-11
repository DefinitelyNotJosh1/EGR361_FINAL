import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# Load the dataset
df  = pd.read_csv('insurance.csv')
print(df.shape)
df = df.drop(columns=['sex', 'smoker', 'region'])
print(df.shape)

label = df['charges']
# print(label.describe())
print(df.describe())


# graph distribution of variables and describe them
plt.figure(figsize=(10, 6))
plt.hist(label, bins=30, edgecolor='black')
plt.title('Distribution of Charges')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()

