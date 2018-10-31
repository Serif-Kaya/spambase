# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 10:28:11 2018

@author: serif
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC 

# Importing the dataset
dataset = pd.read_csv('spambase.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 57].values

dataset.head()

dataset.describe()
list(dataset.columns.values)


dataset.y.value_counts().plot(kind="bar",
                                       title="spam distro",
                                       figsize=(8,8),rot=25,
                                       colormap='Paired')


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



#fitting


model= SVC(kernel='linear', C=1)
model.fit(X_train, y_train)
model.score(y_train,y_train)

#predicting
y_pred=model.predict(X_test)



#fitting -2


model2= SVC(kernel='rbf', C=1.2, gamma=1)
model2.fit(X_train, y_train)
model2.score(y_train,y_train)

#predicting
y_pred2=model2.predict(X_test)


from sklearn.model_selection import cross_val_score
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

cross_val_score(model, X, y, scoring='recall_macro',cv=5)
dd=classification_report(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)

print(dd)
print(cm)

accuracy_score(y_test,y_pred)

#------model2

cross_val_score(model2, X, y, scoring='recall_macro',cv=5)
dd2=classification_report(y_test,y_pred2)
cm2=confusion_matrix(y_test,y_pred2)

print(dd2)
print(cm2)

accuracy_score(y_test,y_pred2)




#logistic regression
from sklearn.linear_model import LogisticRegression 

model_log= LogisticRegression(random_state=0, solver='lbfgs')
model_log.fit(X_train, y_train)
model2.score(X_train,y_train)

y_pred_log=model_log.predict(X_test)

cr_log=classification_report(y_test,y_pred_log)
cm_log=confusion_matrix(y_test,y_pred_log)

print(cr_log)
print(cm_log)

accuracy_score(y_test,y_pred2)
