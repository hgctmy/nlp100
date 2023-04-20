import pandas as pd
import pickle

with open('model.pickle', mode='rb') as f:
    lr = pickle.load(f)

X = pd.read_table("train.feature.txt")
print(lr.predict(X))
