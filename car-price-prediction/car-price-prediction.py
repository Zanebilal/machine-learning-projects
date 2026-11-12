import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


   
df = pd.read_csv('data.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

strings = list(df.dtypes[df.dtypes == 'str'].index)

for col in strings:
    df[col] = df[col].str.lower().str.replace(' ', '_')

for col in df.columns :
    print()
    print(col)
    print(df[col].unique()[:10])
    print(df[col].nunique())


price_logs = np.log1p(df.msrp)

sns.histplot(price_logs, bins=50)

print(df.isnull().sum())


# plt.show()