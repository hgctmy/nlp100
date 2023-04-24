# lightGBMを試してみる
from sklearn.metrics import accuracy_score
import pandas as pd
import optuna.integration.lightgbm as lgb_o

# データを読み込む．目的変数となるカテゴリはアルファベット['b','e','m','t']を[0,1,2,3]に変換
x_train = pd.read_table("train.feature.txt")
y_train = [0 if x == 'b' else 1 if x == 'e' else 2 if x == 'm' else 3 for x in pd.read_table("train.txt")['CATEGORY']]
x_valid = pd.read_table("valid.feature.txt")
y_valid = [0 if x == 'b' else 1 if x == 'e' else 2 if x == 'm' else 3 for x in pd.read_table("valid.txt")['CATEGORY']]
x_test = pd.read_table("test.feature.txt")
y_test = [0 if x == 'b' else 1 if x == 'e' else 2 if x == 'm' else 3 for x in pd.read_table("test.txt")['CATEGORY']]

# lightGBMを用いて分類し，その正解率を求める．パラメータはoptunaを利用して選ぶ．
train_data = lgb_o.Dataset(x_train, label=y_train)
eval_data = lgb_o.Dataset(x_valid, label=y_valid, reference=train_data)
params = {'task': 'train', 'boosting_type': 'gbdt', 'objective': 'multiclass', 'num_class': 4, 'metric': 'multi_logloss'}
gbm = lgb_o.train(params, train_data, valid_sets=[train_data, eval_data], num_boost_round=100, early_stopping_rounds=10)
print(gbm.params)
train_accuracy = accuracy_score(y_train, gbm.predict(x_train).argmax(axis=1))
test_accuracy = accuracy_score(y_test, gbm.predict(x_test).argmax(axis=1))
print("訓練データの正解率", train_accuracy, "\nテストデータの正解率", test_accuracy)


'''
{'task': 'train', 'boosting_type': 'gbdt', 'objective': 'multiclass', 'num_class': 4, 'metric': 'multi_logloss', 'feature_pre_filter': False, 'lambda_l1': 0.0013097965192696546, 'lambda_l2': 6.66467018592995e-08, 'num_leaves': 57, 'feature_fraction': 0.5, 'bagging_fraction': 1.0, 'bagging_freq': 0, 'min_child_samples': 5, 'num_iterations': 100, 'early_stopping_round': 10}
訓練データの正解率 0.9994377811094453
テストデータの正解率 0.8920539730134932
'''
