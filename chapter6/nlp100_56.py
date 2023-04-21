import pandas as pd
import sklearn.metrics

train_predicted = pd.read_table("predict_train.txt").iloc[1, :]
train_answer = pd.read_table("train.txt")['CATEGORY']
test_predicted = pd.read_table("predict_test.txt").iloc[1, :]
test_answer = pd.read_table("test.txt")['CATEGORY']

sklearn.metrics.precision_score()
sklearn.metrics.recall_score()
sklearn.metrics.f1_score()
# 途中
