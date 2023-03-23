import pandas as pd

df = pd.read_table("popular-names.txt", header=None)
print(set(df.iloc[:, 0]))