import pandas as pd

col1 = pd.read_csv("col1.txt", header=None)
col2 = pd.read_csv("col2.txt", header=None)

merge = pd.concat([col1, col2], axis=1)  # 列方向に連結
merge.to_csv("merge.txt", sep="\t", header=False, index=False)

# 6.6コードの意図を書く,p.76

'''
Mary	F
Anna	F
Emma	F
Elizabeth	F
Minnie	F
Margaret	F
Ida	F
Alice	F
Bertha	F
Sarah	F

paste col1.txt col2.txt > col12.txt
'''
