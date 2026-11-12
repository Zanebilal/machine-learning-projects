
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

## printing each column and its firs 10 unique values
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

## displaying the graph 
plt.show()