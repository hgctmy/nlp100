import pandas as pd
from sklearn.metrics import confusion_matrix

train_predicted = pd.read_table("predict_train.txt").iloc[1, :]
train_answer = pd.read_table("train.txt")['CATEGORY']
test_predicted = pd.read_table("predict_test.txt").iloc[1, :]
test_answer = pd.read_table("test.txt")['CATEGORY']
# trainの混同行列
print("train:\n", confusion_matrix(train_predicted, train_answer))
# testの混同行列
print("test:\n", confusion_matrix(test_predicted, test_answer))

'''
train:
 [[4464   10  109  150]
 [  34 4187  149  114]
 [   0    0  483    0]
 [   4    1    2  965]]
test:
 [[558   8  24  37]
 [ 21 522  29  23]
 [  0   0  34   0]
 [  1   1   2  74]]

混同行列の内容
 [[TP   FN  FN  FN]
 [ FP   TN  TN  TN]
 [ FP   TN  TN  TN]
 [ FP   TN  TN  TN]]
'''
