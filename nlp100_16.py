import sys
import pandas as pd

n = int(sys.argv[1])
df = pd.read_table("popular-names.txt", header=None)
line = df.shape[0]//n
for i in range(n):
    df.split = df.iloc[i*line:(i+1)*line]
    df.split.to_csv(f"popular-names{i}.txt", sep="\t", header=False, index=False)