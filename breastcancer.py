#! /usr/bin/python3.5

import numpy as np
from sklearn import model_selection, neighbors
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Load the iris dataset

dataset = pd.read_csv('pima-indians-diabetes.data',
                      names=('Pregnancies', 'Glucose', 'BP', 'SkinThickness',
                             'Insulin', 'BMI', 'DiabetesPedigreeFunction',
                             'Age', 'Outcome'))

print(dataset.head(20))
dataset.replace('?', -99999, inplace=True)

X = np.array(dataset.drop(['Outcome'], 1))
Y = np.array(dataset['Outcome'])

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=200)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)

accuracy = clf.score(X_test, Y_test)
print('Accuracy = ', accuracy)

random = np.array([[6,80,72,35,0,32.2,0.647,35],[6,148,72,35,0,33.6,0.627,50]])
random = random.reshape(len(random), -1)

kfold = model_selection.KFold(n_splits=10, random_state=200)
cv_results = model_selection.cross_val_score(clf, X_train, Y_train, cv=kfold, scoring='accuracy')

prediction = clf.predict(random)

print('My prediction is: ', prediction)

knn = neighbors.KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_test)
print(accuracy_score(Y_test, predictions))
#print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))
