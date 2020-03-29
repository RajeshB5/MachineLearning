#import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('linear_regression_Salary_Data.csv')
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, 1]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size = 1/3, random_state=0)
     
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train,Y_train)

Y_predict = linreg.predict(X_test)

plt.plot(X_train,Y_train,'ro')
plt.plot(X_test,Y_test,'rx')
plt.plot(X_train,linreg.predict(X_train))
plt.show()
