import numpy as np
import pandas as pd

data = pd.read_csv("data/icecreamdataset.csv")
print(data.head())
x = data["Temperature"]
y = data["Ice Cream Profits"]       
print(x.head(),y.head())

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x , y = np.array([x]).reshape(-1,1) , np.array([y]).reshape(-1,1)

xtrain, xtest, ytrain, ytest = train_test_split(x,y,random_state=1,test_size=0.2)

reg = LinearRegression()
reg.fit(X=xtrain,y=ytrain)
ypred = reg.predict(xtest)
print(ypred,ytest)

from sklearn.metrics import mean_squared_error, r2_score
r2 = r2_score(ytest, ypred)

print(f'R-squared: {r2}')

import matplotlib.pyplot as plt 

plt.scatter(xtest, ytest, color='blue', label='Actual Data')
plt.plot(xtest, ypred, color='red', linewidth=2, label='Regression Line')

plt.xlabel('Temperature')
plt.ylabel('Ice Cream Profits')
plt.title('Linear Regression on Ice Cream Profits')
plt.legend()
plt.show()

