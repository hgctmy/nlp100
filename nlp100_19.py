import pandas as pd

df = pd.read_table("popular-names.txt", header=None)
print(df[0].value_counts().head(10))