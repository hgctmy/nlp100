import pandas as pd

df = pd.read_table("popular-names.txt", header=None)
print(set(df.iloc[:, 0]))

# 9.1変数を削除する,p.112
