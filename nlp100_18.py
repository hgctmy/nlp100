import pandas as pd

df = pd.read_table("popular-names.txt", header=None)
df_sorted = df.sort_values(2, ascending=False)
print(df_sorted.head(10))

# 8.1説明変数,p.100

'''
% python nlp100_18.py
            0  1      2     3
1340    Linda  F  99689  1947
1360    Linda  F  96211  1948
1350    James  M  94757  1947
1550  Michael  M  92704  1957
1351   Robert  M  91640  1947
1380    Linda  F  91016  1949
1530  Michael  M  90656  1956
1570  Michael  M  90517  1958
1370    James  M  88584  1948
1490  Michael  M  88528  1954
'''
