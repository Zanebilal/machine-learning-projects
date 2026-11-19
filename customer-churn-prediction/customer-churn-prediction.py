import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from IPython.display import display



df = pd.read_csv('Telco-Customer-Churn.csv')

## preparing the data
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'str'].index)

for c in categorical_columns :
    df[c] = df[c].str.lower().str.replace(' ', '_')


df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)


# splitting the dataframe

df_full_train , df_test = train_test_split(df, test_size= 0.2 , random_state=1)
df_train , df_valid = train_test_split(df_full_train , test_size=0.25, random_state=1)

## resetting the indexes of the dataframes
df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

## getting the target values 
y_train = df_train.churn.values
y_valid = df_valid.churn.values
y_test = df_test.churn.values

## removing the target values from the dataframes
del df_train['churn']
del df_valid['churn']
del df_test['churn']


df_full_train = df_full_train.reset_index(drop=True)

## churn analysis using categorical features 
global_churn = df_full_train.churn.mean() 

for c in categorical_columns:
    print(c)
    df_group = df_full_train.groupby(c).churn.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn
    df_group['risk'] = df_group['mean'] / global_churn
    display(df_group)
    print()
    print()