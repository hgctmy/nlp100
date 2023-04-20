import pandas as pd
import pickle
import numpy as np

with open('model.pickle', mode='rb') as f:
    lr = pickle.load(f)

X = pd.read_table("train.feature.txt")
print(np.max(lr.predict_proba(X), axis=1), lr.predict(X), sep="\n", file=f)

'''
[0.84208212 0.87286569 0.76032397 ... 0.34452535 0.82772827 0.7959898 ]
['b' 'e' 'b' ... 'e' 'e' 'e']
'''
