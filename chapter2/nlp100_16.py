import sys
import pandas as pd

n = int(sys.argv[1])  # 標準入力からnを受け取る
df = pd.read_table("popular-names.txt", header=None)
line = df.shape[0]//n  # n分割したときの１つのファイルの行数
for i in range(n):
    df.split = df.iloc[i*line:(i+1)*line]
    df.split.to_csv(f"popular-names{i}.txt", sep="\t", header=False, index=False)

# 13.4身近なライブラリに親しむ,p.172

'''
% python nlp100_16.py 2
popular-names0.txt
Mary	F	7065	1880
Anna	F	2604	1880
Emma	F	2003	1880
Elizabeth	F	1939	1880
Minnie	F	1746	1880
Margaret	F	1578	1880
Ida	F	1472	1880
Alice	F	1414	1880
Bertha	F	1320	1880
Sarah	F	1288	1880

popular-names1.txt
James	M	86857	1949
Robert	M	83872	1949
John	M	81161	1949
William	M	61501	1949
Michael	M	60046	1949
David	M	59601	1949
Richard	M	50939	1949
Thomas	M	45202	1949
Charles	M	40042	1949
Larry	M	31809	1949
'''
