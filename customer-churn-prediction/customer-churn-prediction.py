import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle
from sklearn.model_selection import train_test_split
from IPython.display import display
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from tqdm.auto import tqdm


# parameters

C = 1.0
n_splits = 5
output_file = f'model_C={C}.bin'


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

## making predictions  
y_pred = model.predict_proba(X_valid)[:, 1]

## assigning each weight with its appropriate feature 
dict(zip(dv.get_feature_names_out(), model.coef_[0].round(3)))


## getting the full training data to train the model
dicts_full_train = df_full_train[categorical_columns + numerical_columns].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

y_full_train = df_full_train.churn.values

# training the model
model = LogisticRegression(solver='lbfgs')
model.fit(X_full_train, y_full_train)

## setting the test data 
dicts_test = df_test[categorical_columns + numerical_columns].to_dict(orient='records')
X_test = dv.transform(dicts_test)

## checking the model accuracy on the test data
y_pred = model.predict_proba(X_test)[:, 1]
churn_decision = (y_pred >= 0.5)
print((churn_decision == y_pred).mean())  ## 80% predicted right (same as the validation part)

## checking for a single customer
customer = dicts_test[-1]
X_small = dv.transform([customer])
print("the prediction of this customer to churns is : " , model.predict_proba(X_small)[0, 1]) ## we get 67% (above the 50% threshold ==> the customer will churn)

print("the real value of this customer to churn is :" , y_test[-1])   ## we get 1 ==> the customer will churn


## checking the accuracy of the model
accuracy_score(y_valid, y_pred >= 0.5)

## checking for different thresholds  (best one is 0.5 with 80% right prediction)
thresholds = np.linspace(0, 1, 21)

scores = []

for t in thresholds:
    score = accuracy_score(y_valid, y_pred >= t)
    print('%.2f %.3f' % (t, score))
    scores.append(score)


## comparing it with a dummy model (never seen the data before)
print(1-y_valid.mean())  ## we get 73%

## computing the count of each case in the confusion table

actual_positive = (y_valid == 1)
actual_negative = (y_valid == 0)

t = 0.5
predict_positive = (y_pred >= t)
predict_negative = (y_pred < t)

tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()

fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()

## creating the confusion table 
confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])

print((confusion_matrix / confusion_matrix.sum()).round(2))

## computes the ROC curve and the area for our model 
print(roc_auc_score(y_valid, y_pred))


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical_columns + numerical_columns].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model


def predict(df, dv, model):
    dicts = df[categorical_columns + numerical_columns].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


## cross validation using k-fold and selecting the best value for param c 

n_splits = 5

for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


## training the model with the full data
dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
print("the new auc after evaluating the model is : ", auc)


## saving the model 
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')


