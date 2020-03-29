import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data_preprocesing-Data.csv')
X = dataset.iloc[:, 0:3]
Y = dataset.iloc[:, 3]

#print(X.isnull().sum())
#X.loc[2,'Age'] = 55
#medi = X['Age'].median()
#X["Age"].fillna(medi, inplace=True)
#X["Salary"].fillna(X["Salary"].median(), inplace=True)
#print(X)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
imputer = SimpleImputer(missing_values = np.nan)
imputer = imputer.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3] = imputer.transform(X.iloc[:, 1:3])

labEnc_X = LabelEncoder()
X.iloc[:,0] = labEnc_X.fit_transform(X.iloc[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labEnc_Y = LabelEncoder()
Y = labEnc_Y.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size = 0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
X_train = scX.fit_transform(X_train)  
X_test = scX.transform(X_test)

