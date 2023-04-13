import pandas as pd

df = pd.read_table("popular-names.txt", header=None)

col1 = df.iloc[:, 0]
col2 = df.iloc[:, 1]

col1.to_csv("col1.txt", sep=",", header=False, index=False)
col2.to_csv("col2.txt", sep=",", header=False, index=False)

# 5.1コメントするべきではないこと,p.57

'''
col1.txt
Mary
Anna
Emma
Elizabeth
Minnie
Margaret
Ida
Alice
Bertha
Sarah

col2.txt
F
F
F
F
F
F
F
F
F
F
'''
