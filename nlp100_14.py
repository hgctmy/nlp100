import pandas as pd

df = pd.read_table("popular-names.txt", header=None)
print(df.head(10))

#13.4身近なライブラリに親しむ,p.172