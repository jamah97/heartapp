import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


heart_data = pd.read_csv('heart-3.csv')
heart_data.head()

# feature selection
x = heart_data.drop(['output'], axis=1)
y = heart_data['output'].values

# train and test split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.20,random_state = 5)

#model Logistic Regression
reg = LogisticRegression()
reg.fit(X_train, y_train)
y_pred=reg.predict(X_test)
print('Logistic Regression accuracy score:', accuracy_score(y_test,y_pred)*100)


import pickle

file = open('model.pkl', 'wb')

pickle.dump(reg, file)
