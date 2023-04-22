import pandas as pd
from sklearn.metrics import classification_report

test_predicted = pd.read_table("predict_test.txt").iloc[1, :]
test_answer = pd.read_table("test.txt")['CATEGORY']

# 適合率，再現率，f1スコアおよびそれらのマクロ平均，マイクロ平均，加重平均を算出
print(classification_report(test_answer, test_predicted))
# 正解データと推定データに含まれるラベル種類が一致した時，マイクロ平均は正解率に一致するため，出力のaccuracyの部分がマイクロ平均である．

'''
% python nlp100_56.py
              precision    recall  f1-score   support

           b       0.89      0.96      0.92       580
           e       0.88      0.98      0.93       531
           m       1.00      0.38      0.55        89
           t       0.95      0.55      0.70       134

    accuracy                           0.89      1334
   macro avg       0.93      0.72      0.78      1334
weighted avg       0.90      0.89      0.88      1334
'''
