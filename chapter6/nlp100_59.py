# svmを試してみる
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import GridSearchCV

x_train = pd.read_table("train.feature.txt")
y_train = pd.read_table("train.txt")['CATEGORY']
x_test = pd.read_table("test.feature.txt")
y_test = pd.read_table("test.txt")['CATEGORY']

# svcを用いて分類し，その正解率を求める．パラメータは正解率の高いものを選ぶ
search_params = {"kernel": ["rbf", "linear", "sigmoid"], "C": [10**i for i in range(-10, 10)]}
gs = GridSearchCV(SVC(), search_params, scoring="accuracy", cv=3, refit=True, n_jobs=-1)  # グリッドサーチでパラメータ探索
gs.fit(x_train, y_train)
print(gs.best_estimator_)
train_accuracy = accuracy_score(y_train, gs.predict(x_train))
test_accuracy = accuracy_score(y_test, gs.predict(x_test))
print(train_accuracy, test_accuracy)
