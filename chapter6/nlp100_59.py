# lightGBMを試してみる
from sklearn.metrics import accuracy_score
import pandas as pd
import optuna.integration.lightgbm as lgb_o

x_train = pd.read_table("train.feature.txt")
y_train = [0 if x == 'b' else 1 if x == 'e' else 2 if x == 'm' else 3 for x in pd.read_table("train.txt")['CATEGORY']]
x_valid = pd.read_table("valid.feature.txt")
y_valid = [0 if x == 'b' else 1 if x == 'e' else 2 if x == 'm' else 3 for x in pd.read_table("valid.txt")['CATEGORY']]
x_test = pd.read_table("test.feature.txt")
y_test = [0 if x == 'b' else 1 if x == 'e' else 2 if x == 'm' else 3 for x in pd.read_table("test.txt")['CATEGORY']]

# lightGBMを用いて分類し，その正解率を求める．パラメータは正解率の高いものを選ぶ．
train_data = lgb_o.Dataset(x_train, label=y_train)
eval_data = lgb_o.Dataset(x_valid, label=y_valid, reference=train_data)
params = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'multiclass', 'num_class': 4, 'metric': 'multi_logloss'}
gbm = lgb_o.train(params, train_data, valid_sets=[train_data, eval_data], num_boost_round=100, early_stopping_rounds=10)
print(gbm.params)
train_accuracy = accuracy_score(y_train, gbm.predict(x_train))
test_accuracy = accuracy_score(y_test, gbm.predict(x_test))
print(train_accuracy, test_accuracy)
