import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data = pd.read_csv("data/diabetes_dataset.csv")

x = data.drop("Outcome",axis=1)
y = data["Outcome"]

xtrain, xtest, ytrain, ytest = train_test_split(x, y, shuffle=True, random_state=42, test_size=0.2)
print(xtrain)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(xtrain, ytrain)

y_pred = clf.predict(xtest)

report = classification_report(ytest, y_pred)
print(report)

export_graphviz(clf, out_file='dot_data.dot', feature_names=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI" ,"DiabetesPedigreeFunction","Outcome"])

