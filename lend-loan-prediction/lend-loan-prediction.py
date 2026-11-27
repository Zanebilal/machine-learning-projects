import pandas as pd
import numpy as np

import seaborn as sns

import xgboost as xgb

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier



# data preparation 

df = pd.read_csv('CreditScoring.csv')
df.columns = df.columns.str.lower()

## re-encoding the categorical variables
 
status_values = {
    1: 'ok',
    2: 'default',
    0: 'unknown'
}

df.status = df.status.map(status_values)

home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk'
}

df.home = df.home.map(home_values)

marital_values = {
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'separated',
    5: 'divorced',
    0: 'unk'
}

df.marital = df.marital.map(marital_values)

records_values = {
    1: 'no',
    2: 'yes',
    0: 'unk'
}

df.records = df.records.map(records_values)

job_values = {
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others',
    0: 'unk'
}

df.job = df.job.map(job_values)

## handling messing values
### missing values in the dataframe are encoded as 99999999
for c in ['income', 'assets', 'debt']:
    df[c] = df[c].replace(to_replace=99999999.0, value = np.nan)


## removing the unknown status
df = df[df.status != 'unknown'].reset_index(drop=True)

## splitting the data into training, validation , test data
df_full_train , df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train , df_valid = train_test_split(df_full_train, test_size=0.25, random_state=11)

### resetting the indexes of each df
df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

## preparing the target variable "status = Default "
y_train = (df_train.status == 'default').astype('int').values
y_valid = (df_valid.status == 'default').astype('int').values
y_test = (df_test.status == 'default').astype('int').values

## removing the target column from the dataframes
del df_train['status']
del df_valid['status']
del df_test['status']

# training the decision tree

## filling the missing values with 0 
train_dicts = df_train.fillna(0).to_dict(orient='records')

dv = DictVectorizer(sparse=False)

## getting the feature matrix for the training data 
X_train = dv.fit_transform(train_dicts)

## training the tree
dt= DecisionTreeClassifier(max_depth=2)
dt.fit(X_train, y_train)

## testing the model in the validation data
valid_dicts = df_valid.fillna(0).to_dict(orient='records')

X_valid = dv.transform(valid_dicts)

### the prediction of the model in the validation data
y_predict = dt.predict_proba(X_valid)[:,1] 

### getting the auc score to see the precision of the model
auc = roc_auc_score(y_valid, y_predict)

### getting the auc score for the training data
y_pred = dt.predict_proba(X_train)[:, 1]
auc = roc_auc_score(y_train, y_pred)

# implementing the boosting forest 
## wrapping the data into DMatrix 
features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_valid, label=y_valid, feature_names=features)

## training the model 
xgb_params = {
    'eta': 0.3,
    'max_depth': 6,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'nthread': 8,

    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=10)

## getting the predictions
y_pred = model.predict(dval)
print(roc_auc_score(y_valid, y_pred))

# choosing the right model between the three models using the tunned params

## decision tree model
dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15)
dt.fit(X_train, y_train)

y_pred = dt.predict_proba(X_valid)[:, 1]
print("thw auc value for decision tree model :" , roc_auc_score(y_valid, y_pred))

## random forest model

rf = RandomForestClassifier(n_estimators=200,
                            max_depth=10,
                            min_samples_leaf=3,
                            random_state=1)
rf.fit(X_train, y_train)

y_pred = rf.predict_proba(X_valid)[:, 1]
print("thw auc value for random forest model :" ,roc_auc_score(y_valid, y_pred))


## XGBoost model

xgb_params = {
    'eta': 0.1,
    'max_depth': 3,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=175)

y_pred = model.predict(dval)
print("thw auc value for random forest model :" ,roc_auc_score(y_valid, y_pred))



