import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/breastcancer.csv")
df.drop(['id'], 1, inplace=True)
df = df.dropna(axis=1, how='all')
null_counts = df.isnull().sum()
print(df)
print(null_counts)
x = np.array(df.drop(["diagnosis"],1))
y = np.array(df["diagnosis"])
print(x,y)

xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.2,random_state=42)

model = svm.SVC()
model.fit(xtrain,ytrain)

score = model.score(xtest,ytest)
print(score)

test = np.array([[18.99,10,152.8,1050,0.2084,0.4776,0.2001,0.1471,0.2419,0.07871,1.095,0.9053,7.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,30.38,17.33,185.6,2024,0.1622,0.6656,0.8000,0.2654,0.4601,0.1189]])
pred = model.predict(test)
print(pred)