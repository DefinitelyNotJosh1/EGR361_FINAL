import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.model_selection import train_test_split


from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


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

x_variables = df_normalized[x_cols]
label = df_normalized['AvgPitStopTime']

print("Normalized Data:")
print(df_normalized.describe())

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(df_normalized[x_cols], df_normalized['AvgPitStopTime'], test_size=0.2, random_state=42)

print("Training Set Size:")
print(x_train.shape)
print("Testing Set Size:")
print(x_test.shape)

################################################################
######## BE VERY CAREFUL ABOUT CHANGING ANY CODE BELOW THIS LINE
################################################################

X = x_variables
y = label

# remove nan (empty) values in the labels 
ynan = np.isnan(y)

not_ynan = [not y for y in ynan] # flip truth values for masking
X = X[not_ynan]
y = y[not_ynan]
X.reset_index(drop=True,inplace=True)

# split into 5 folds 
kf = KFold(n_splits=5)

beta = []
RMSE_train = []
RMSE_test = []
R2_train = []
R2_test = [] 


for i, (train_index, test_index) in enumerate(kf.split(X)):

    # define the training and testing sets 
    xTrain = X.loc[train_index,:]
    xTest = X.loc[test_index,:]
    yTrain = y[train_index]
    yTest = y[test_index]

    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(xTrain, yTrain)

    y_pred_train = regr.predict(xTrain) # find predicted y-values for the training set 
    y_pred_test = regr.predict(xTest) # find predicted y-values for the testing set 

    # this is going to create a colorful figure where every color is the data pairs from one fold
    plt.scatter(yTest,y_pred_test)
    plt.xlabel('True Y Values')
    plt.ylabel('Predicted Y Values')
    plt.savefig('y_pred_vs_y_true.png')
   

    # save the regression coefficients (beta)
    beta.append(regr.coef_)
    # print("Coefficients (Beta): \n", regr.coef_)

    # save the root mean squared error (RMSE): 0 is a perfect prediction
    this_rmse_train = mean_squared_error(yTrain, y_pred_train)**0.5 
    this_rmse_test = mean_squared_error(yTest, y_pred_test)**0.5 
    RMSE_train.append(this_rmse_train)
    RMSE_test.append(this_rmse_test)
    # print("Mean squared error: %.2f" % mean_squared_error(yTest, y_pred_test))

    # The coefficient of determination (R2): 1 is perfect prediction
    R2_train.append(r2_score(yTrain,y_pred_train))
    R2_test.append(r2_score(yTest,y_pred_test))
    # print("Coefficient of determination: %.2f" % r2_score(yTest, y_pred_test))

print("\nthis is the beta: ")
print(beta)

print("\nthis is the RMSE for the training set: ")
print(RMSE_train)

print("\nthis is the RMSE for the testing set: ")
print(RMSE_test)

print("\nthis is the R2 for the training set: ")
print(R2_train)

print("\nthis is the R2 for the testing set: ")
print(R2_test)

# Plot outputs
plt.clf
plt.subplot(121)
plt.scatter(range(1,6), RMSE_train, color="black")
plt.scatter(range(1,6), RMSE_test, color="blue")

plt.xticks((1,2,3,4,5))

plt.xlabel('fold number')
plt.ylabel('RMSE')

plt.subplot(122)
plt.scatter(range(1,6), R2_train, color="black")
plt.scatter(range(1,6), R2_test, color="blue")

plt.xticks((1,2,3,4,5))

plt.xlabel('fold number')
plt.ylabel('R2')

# plt.show() 
plt.savefig('ml_results.png')

# plot model over track temperature data
plt.clf()
plt.scatter(x_variables['Track_Temp_C'], label, color="black")
plt.plot(xTest['Track_Temp_C'], y_pred_test, color='red', label='Regression line')
plt.xlabel('Track Temperature (C)')
plt.legend()
plt.ylabel('Average Pit Stop Time (seconds)')
plt.title('Average Pit Stop Time vs Track Temperature')
plt.show()
plt.savefig('model_over_track_temp.png')

# print data points in test set that went over the max in the training set
for col in x_cols:
    max_val = xTrain[col].max()
    print(f"\nMax {col} in training set: {max_val}")
    print(f"Test set values greater than {col} max:")
    print(xTest[xTest[col] > max_val])
    print(xTest[xTest[col] > max_val][col])
    print("\n")

# print data points in test set that went under the min in the training set
for col in x_cols:
    min_val = xTrain[col].min()
    print(f"\nMin {col} in training set: {min_val}")
    print(f"Test set values less than {col} min:")
    print(xTest[xTest[col] < min_val])
    print(xTest[xTest[col] < min_val][col])
    print("\n")

