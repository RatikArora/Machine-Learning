from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import numpy as np

housing = fetch_california_housing()
print(housing)

import pandas as pd

data = pd.DataFrame(housing["data"], columns = housing['feature_names'])
print(data)
# as target is equal to median house value in $100,000 
data["target"] = housing['target']
print(data)
x = data.drop('target',axis=1)
y = data['target']

xtrain , xtest, ytrain, ytest =train_test_split(x,y,random_state=42,test_size=0.2,shuffle=True)


from sklearn import linear_model

model1 = linear_model.Ridge()
model1.fit(xtrain,ytrain)
ypred = model1.predict(xtest)
print(ypred)
print(x.shape,y.shape)

print(model1.score(xtest,ytest))

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=500)
model.fit(xtrain,ytrain)
print(model.score(xtest,ytest))
