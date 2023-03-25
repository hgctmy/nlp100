import pandas as pd

df = pd.read_table("popular-names.txt", header=None)
print(df[0].value_counts().head(10)) #一列目の要素の頻度を多い順に10個

#13.4身近なライブラリに親しむ,p.172