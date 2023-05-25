import pandas as pd

train_predicted = pd.read_table("predict_train.txt").iloc[1, :]
train_answer = pd.read_table("train.txt")['CATEGORY']
test_predicted = pd.read_table("predict_test.txt").iloc[1, :]
test_answer = pd.read_table("test.txt")['CATEGORY']

# trainの正解率
n = sum(1 for x, y in zip(train_predicted, train_answer) if x == y)  # 正解数
accuracy_train = n / train_predicted.shape[0]
print(accuracy_train)
# testの正解率
n = sum(1 for x, y in zip(test_predicted, test_answer) if x == y)  # 正解数
accuracy_test = n / test_predicted.shape[0]
print(accuracy_test)

# sklearnのaccuracy_score()でもできる
'''
% python nlp100_54.py
0.946308095952024
0.8905547226386806
'''
