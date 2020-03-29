import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("multiple_linear_regression_50_Startups.csv")
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, 4]

if dataset.isnull().sum().sum():
    print("missing value is present")
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labEnc_X = LabelEncoder()
X.iloc[:,3] = labEnc_X.fit_transform(X.iloc[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#Avoid dummy variable trap
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X_train,Y_train)

Y_perdict = linreg.predict(X_test)

#plt.plot(Y_perdict, Y_test, 'ro')
#plt.show()

import statsmodels.formula.api as st
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]]
regrasor_ols = st.ols(endog = Y, exog = X_opt).fit()


