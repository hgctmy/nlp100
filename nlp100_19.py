import pandas as pd

df = pd.read_table("popular-names.txt", header=None)
print(df[0].value_counts().head(10))  # 一列目の要素の頻度を多い順に10個

# 13.4身近なライブラリに親しむ,p.172

'''
% python nlp100_19.py
James        118
William      111
Robert       108
John         108
Mary          92
Charles       75
Michael       74
Elizabeth     73
Joseph        70
Margaret      60
Name: 0, dtype: int64
'''
