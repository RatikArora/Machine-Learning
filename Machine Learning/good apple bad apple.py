import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv("data/apple_quality.csv")
print(data.shape)
x = data.drop('Quality',axis=1)
y = data['Quality']
print(x.shape,y.shape)
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=42,shuffle=True)
print(xtrain.shape,ytrain.shape,xtest.shape,ytest.shape)


model = RandomForestClassifier(n_estimators=300)

model.fit(xtrain,ytrain)
ypred = model.predict(xtest)
acc = accuracy_score(ytest,ypred)
matrix = confusion_matrix(ytest,ypred)
print(matrix)
print(model.score(xtest,ytest))
print(acc)


