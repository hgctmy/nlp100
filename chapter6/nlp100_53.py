import pandas as pd
import pickle
import numpy as np

with open('model.pickle', mode='rb') as f:
    lr = pickle.load(f)  # 学習したモデルを読み込む

X_train = pd.read_table("train.feature.txt")
predicted_train = pd.DataFrame([np.max(lr.predict_proba(X_train), axis=1), lr.predict(X_train)])  # 予測した結果とその確率
X_test = pd.read_table("test.feature.txt")
predicted_test = pd.DataFrame([np.max(lr.predict_proba(X_test), axis=1), lr.predict(X_test)])  # 予測した結果とその確率
predicted_train.to_csv("predict_train.txt", sep="\t", index=False)
predicted_test.to_csv("predict_test.txt", sep="\t", index=False)

'''
train
0.84208212  0.87286569  0.76032397 ...
b           e            b         ...

test
0.88350657	0.6699923	0.87644455  ...
b	        t	        e           ...
'''
