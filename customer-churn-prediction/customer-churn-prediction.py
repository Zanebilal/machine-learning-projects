import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from IPython.display import display
from sklearn.metrics import mutual_info_score



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
# mutual information using sklearn library 
def mutual_info_churn_score(series):
    return mutual_info_score(df_full_train.churn, series)

mutual_info = df_full_train[categorical_columns].apply(mutual_info_churn_score) 

# determining the important numerical features with correlation

numerical_columns = ['tenure', 'monthlycharges', 'totalcharges']

print(df_full_train[numerical_columns].corrwith(df_full_train.churn).abs())

# analyzing the churn of numerical columns in categories
print("the max number of tenure : ", df_full_train.tenure.max())

## splitting the tenure into categories and calculate there churn for each category (less then 2 months, between 2 month and 1 year, more then 1 year)
print("for less then 2 months of subscription :" , df_full_train[df_full_train.tenure <= 2 ].churn.mean())   ## for less then 2 months of subscription
print("etween 2 months and one year : " , df_full_train[(df_full_train.tenure > 2) & (df_full_train.tenure <= 12)].churn.mean())     ## between 2 months and one year
print("more then one year :", df_full_train[df_full_train.tenure <= 2 ].churn.mean()  )    ## more then one year

## splitting the  monthlycharges  into categories and calculate there churn for each category
print("paying less then 20 dollars : ", df_full_train[df_full_train.monthlycharges <= 20].churn.mean())
print("paying between 20 and 50 dollars : ", df_full_train[(df_full_train.monthlycharges > 20) & (df_full_train.monthlycharges <= 50)].churn.mean())
print("paying more then 50 dollars : ", df_full_train[df_full_train.monthlycharges > 50].churn.mean())