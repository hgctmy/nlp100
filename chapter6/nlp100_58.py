from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa

x_train = pd.read_table("train.feature.txt")
y_train = pd.read_table("train.txt")['CATEGORY']
x_valid = pd.read_table("valid.feature.txt")
y_valid = pd.read_table("valid.txt")['CATEGORY']
x_test = pd.read_table("test.feature.txt")
y_test = pd.read_table("test.txt")['CATEGORY']

train_accuracy = []
valid_accuracy = []
test_accuracy = []
prm = [0.01, 0.05, 0.1, 0.5, 1.0]  # 正則化パラメータ
for i in prm:  # それぞれの正則化パラメータについて，学習と正解率の計算を行う
    lr = LogisticRegression(max_iter=3000, C=i)
    lr.fit(x_train, y_train)
    train_accuracy.append(accuracy_score(y_train, lr.predict(x_train)))
    valid_accuracy.append(accuracy_score(y_valid, lr.predict(x_valid)))
    test_accuracy.append(accuracy_score(y_test, lr.predict(x_test)))

# 正則化パラメータを横軸，正解率を縦軸としたグラフ
plt.plot(prm, train_accuracy, label="train")
plt.plot(prm, valid_accuracy, label='valid')
plt.plot(prm, test_accuracy, label='test')
plt.ylim(0, 1.1)
plt.ylabel('正解率')
plt.xlabel('正則化パラメータ')
plt.legend()
plt.show()

'''
ans58.png
過学習傾向はあるものの1以下の正則化パラメータではパラメータが大きくなるほどテスト，検証データについての正解率が高くなった．
大きすぎても過学習になるため1くらいが良いのでは？
'''
