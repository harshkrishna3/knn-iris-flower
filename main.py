#%%codecell
#importing dependencies
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import pandas as pd

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#%%codecell
#finding optimal value of k
score = []
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    score.append(cross_val_score(knn, X_train, y_train, scoring = 'accuracy', cv = 10).mean())
print(score)
plt.plot(range(1, 11), score)


#%%codecell
#training model
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)

#%%codecell
#predicting data and calculating accuracy
y_pred = knn.predict(X_test)
print('accuracy:', knn.score(X_test, y_test))
print('r2 score:', r2_score(y_test, y_pred))
