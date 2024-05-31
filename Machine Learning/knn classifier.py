import pandas as pd
import numpy as np 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/IRIS.csv')
print(data.head())
x = data.drop(['species'], axis=1)
print(x.head())
y = data['species']
print(y.head())

xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=42,shuffle=True)
print(xtrain.shape,xtest.shape,ytest.shape)

knn=KNeighborsClassifier(n_neighbors=5,metric='euclidean')

knn.fit(xtrain,ytrain)
ypred = knn.predict(xtest)
acc = accuracy_score(ytest,ypred)
print(acc)

conf_matrix = confusion_matrix(ytest, ypred)
print("Confusion Matrix:")
print(conf_matrix)


from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(ytest, ypred, average='macro')
recall = recall_score(ytest, ypred, average='macro')
f1 = f1_score(ytest, ypred, average='macro')

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

inp = pd.DataFrame({
    'sepal_length': [5.1],
    'sepal_width': [3.5],
    'petal_length': [1.4],
    'petal_width': [0.2]
})

custom_pred = knn.predict(inp)
print("Predicted Species:", custom_pred)