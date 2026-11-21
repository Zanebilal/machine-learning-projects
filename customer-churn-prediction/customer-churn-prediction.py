import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from IPython.display import display
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression



df = pd.read_csv('Telco-Customer-Churn.csv')

## preparing the data
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.select_dtypes(include=['object', 'string']).columns)

categorical_columns.remove('churn')
categorical_columns.remove('customerid')

for c in categorical_columns :
    df[c] = df[c].str.lower().str.replace(' ', '_')


df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df['churn'] = (df['churn'].str.lower() == 'yes').astype(int)

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

## getting the numerical columns
numerical_columns = ['tenure', 'monthlycharges', 'totalcharges']

## performing Feature Vectorization
### Initializing DictVectorizer
dv = DictVectorizer(sparse=False)

### Converting the training DataFrame into a list of dictionaries 
train_dict = df_train[categorical_columns + numerical_columns].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

## encoding the validation df 
valid_dict = df_valid[categorical_columns + numerical_columns].to_dict(orient='records')
X_valid = dv.transform(valid_dict)


#  the logistic regression function 
model = LogisticRegression(solver='lbfgs')
model.fit(X_train, y_train)

print(" the bias wights is : " , model.intercept_[0])
print("the other wights are :" , model.coef_[0].round(3))

## making predictions  
y_pred = model.predict_proba(X_valid)[:, 1]

churn_decision = (y_pred >= 0.5)
print(churn_decision)

## comparing the prediction with the actual value 
print((y_valid == churn_decision).mean())    ## we have got 80% guessed right
