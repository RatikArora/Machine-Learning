import pandas as pd
import numpy as np

data = pd.read_csv("data/car-sales-extended.csv")
# print(data.head())

x = data.drop("Price",axis=1)
# print(x.head())
y = data["Price"]
# print(y.head())

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

categorical_features = ['Make','Colour','Doors']
onehot = OneHotEncoder()
transformer = ColumnTransformer([('onehot',onehot,categorical_features)],remainder='passthrough')

Xtrans = transformer.fit_transform(x)
# print(Xtrans)

x = pd.DataFrame(Xtrans)
print(x,"\n",x.dtypes)

from sklearn.model_selection import train_test_split
np.random.seed(42)
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2)


from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(xtrain,ytrain)
score = model.score(xtest,ytest)
print(score)



