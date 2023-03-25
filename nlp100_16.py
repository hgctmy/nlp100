import sys
import pandas as pd

n = int(sys.argv[1]) #標準入力からnを受け取る
df = pd.read_table("popular-names.txt", header=None)
line = df.shape[0]//n #n分割したときの１つのファイルの行数
for i in range(n):
    df.split = df.iloc[i*line:(i+1)*line]
    df.split.to_csv(f"popular-names{i}.txt", sep="\t", header=False, index=False)

#13.4身近なライブラリに親しむ,p.172