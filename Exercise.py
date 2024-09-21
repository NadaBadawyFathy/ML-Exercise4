import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df = pd.read_csv("carprices.csv")

le = LabelEncoder()
df['Car Model'] = le.fit_transform(df['Car Model'])
X = df.drop('Sell Price($)', axis='columns')

ct = ColumnTransformer([('Car Model', OneHotEncoder(), [0])], remainder = 'passthrough')

X = ct.fit_transform(X)
X = X[:,1:]
y = df['Sell Price($)']

model = linear_model.LinearRegression()
model.fit(X,y)

print(model.predict([[0,1,45000,4]]))
print(model.predict([[1,0,86000,7]]))
print(model.score(X,y))