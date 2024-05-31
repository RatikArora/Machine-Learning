import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("data/heart-disease.csv")
print(data.head())

x = data.drop('target',axis=1)
# print(x.head())
y = data["target"]
# print(y.head())

from sklearn.model_selection import train_test_split,cross_val_score

xtrain, xtest, ytrain, ytest = train_test_split(x,y,random_state=1,test_size=0.2)

print(xtrain.shape,ytest.shape)

# from sklearn import svm
# model2 = svm.SVC()
# model2.fit(xtrain,ytrain)
# print(model2.score(xtest,ytest))

# from sklearn.neighbors import KNeighborsClassifier
# model3 = KNeighborsClassifier()
# model3.fit(xtrain,ytrain)
# print(model3.score(xtest,ytest))


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
model = RandomForestClassifier(n_estimators=100)
model.fit(xtrain,ytrain)
simplescore = model.score(xtest,ytest)
score = cross_val_score(model,x,y,cv=10)
print(score)
print(simplescore)
# cross val score is always better than the simple score \
print(np.mean(score))
ypred = model.predict(xtest)
print(ypred)
cm = confusion_matrix(ytest,ypred)
print(cm)

