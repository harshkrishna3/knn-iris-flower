#%%codecell
#importing dependencies
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score

#%%codecell
#load data
iris = load_iris()
# print(iris)
X = pd.DataFrame(iris.data, columns=iris.feature_names)
print(X.head())
y = pd.DataFrame(iris.target, columns=['flower'])
# print(y)

#%%codecell
#preprocessing
ohe = OneHotEncoder(categories = 'auto', drop='first', sparse=False)
y = pd.DataFrame(ohe.fit_transform(y), columns=iris.target_names[:-1])
print(y.head())
scalar = StandardScaler()
X = pd.DataFrame(scalar.fit_transform(X), columns=X.columns.values)
print(X.head())

#%%codecell
#spilting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4)
X_test, X_cv, y_test, y_cv = train_test_split(X_test, y_test, test_size=0.5)

#%%codecell
#training model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

#%%codecell
#predicting data and calculating accuracy
y_pred_test = knn.predict(X_test)
print('accuracy on test set =', knn.score(X_test, y_test))
print('accuracy on cross validation set =', knn.score(X_cv, y_cv))
y_pred_cv = knn.predict(X_cv)
print('r2 score on test set =', r2_score(y_test, y_pred_test))
print('r2 score on cross validation set =', r2_score(y_cv, y_pred_cv))
