
## importing the necessary libraries

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


## loading the data in a dataframe   
df = pd.read_csv('data.csv')

## aligning the column names (all names in lowercase replacing the space with underscore)
df.columns = df.columns.str.lower().str.replace(' ', '_')

## getting the list of values for each column
strings = list(df.dtypes[df.dtypes == 'str'].index)

## aligning the values of each column 
for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')

# ## printing each column and its firs 10 unique values
for col in df.columns :
    print()
    print(col)
    print(df[col].unique()[:10])
    print(df[col].nunique())


## transforming car price (target variable to visualize) into logarithm distribution to remove the long tail problem
price_logs = np.log1p(df.msrp)

##  plotting the graph
sns.histplot(price_logs, bins=50)

## getting the total number of null values 
print(df.isnull().sum())

# ## displaying the graph 
# plt.show()

## preparing the validation framework 

total = len(df)

## calculating the size of data required for validation , test and training  
n_valid = int(total * 0.2)
n_test = int(total * 0.2)
n_train = total - n_valid - n_test

'''''
## getting the dataframes for each step (we get an ordered data frame for each step which is a problem)

df_valid = df.iloc[:n_valid]
df_test = df.iloc[n_valid: n_valid + n_test]
df_train = df.iloc[n_valid+n_test:]
'''


## shuffling the data by generating a sequence of numbers and shuffle them 

np.random.seed(2)

index = np.arange(total)
np.random.shuffle(index)

## indexing the dataframe of each step with the shuffled numbers 
df_train = df.iloc[index[:n_train]]
df_valid = df.iloc[index[n_train:n_train+n_valid]]
df_test = df.iloc[index[n_train+n_valid:]]

## resetting the indexes for each dataframe
df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


## preparing the target y values
y_train = np.log1p(df_train.msrp.values)
y_valid = np.log1p(df_valid.msrp.values)
y_test = np.log1p(df_test.msrp.values)

## deleting the target y from the dataframes 
del df_train['msrp']
del df_valid['msrp']
del df_test['msrp']

## implementing the linear regression function to predict the price of the car

X = [
    [148, 24, 1385],
    [132, 25, 2031],
    [453, 11, 86],
    [158, 24, 185],
    [172, 25, 201],
    [413, 11, 86],
    [38,  54, 185],
    [142, 25, 431],
    [453, 31, 86],
]

X = np.array(X)

y = [100, 200, 150, 250, 100, 200, 150, 250, 120]


## calculating the wights w of the features 
def train_linear_regression(X,y):

    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    ## calculating the wights using the gram matrix with the formula : w = (XᵀX)⁻¹Xᵀy
    XtX = X.T.dot(X)
    XtX_inverse = np.linalg.inv(XtX)

    w_full = XtX_inverse.dot(X.T).dot(y)

    return w_full[0], w_full[1:]



## getting the int columns and there values  to get the base wight
base = ['engine_hp', 'engine_cylinders', 'highway_mpg',
        'city_mpg', 'popularity']

X_train = df_train[base].values

## filling the missing values with zeros
X_train = df_train[base].fillna(0).values

## calculating the base wight based on the int features 
w0 , w = train_linear_regression(X_train, y_train)

## predicting the values of the target (the price)
y_predicted = w0 + X_train.dot(w)

sns.histplot(y_predicted, color='red', alpha=0.5, bins=50)
sns.histplot(y_train, color='blue', alpha=0.5, bins=50)

# plt.show()

## implementing the RMSE function 
def rmse(y, y_pred):
    se = (y - y_pred)**2
    mse = se.mean()
    return np.sqrt(mse)

print(rmse(y_train, y_predicted))