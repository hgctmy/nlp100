import pandas as pd

df = pd.read_table("popular-names.txt", header=None)
df_sorted = df.sort_values(2, ascending=False)
print(df_sorted.head(10))