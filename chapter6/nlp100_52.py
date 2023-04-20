from sklearn.linear_model import LogisticRegression
import pandas as pd
import pickle

X = pd.read_table("train.feature.txt")
Y = pd.read_table("train.txt")['CATEGORY']

lr = LogisticRegression(max_iter=3000)  # デフォルトだと収束しなかったためmax_iter=3000とした
lr.fit(X, Y)

with open('model.pickle', mode='wb') as f:
    pickle.dump(lr, f)
