import pandas as pd

df = pd.read_table("popular-names.txt", header=None)
print(df.head(10))

# 13.4身近なライブラリに親しむ,p.172

'''
% python nlp100_14.py
           0  1     2     3
0       Mary  F  7065  1880
1       Anna  F  2604  1880
2       Emma  F  2003  1880
3  Elizabeth  F  1939  1880
4     Minnie  F  1746  1880
5   Margaret  F  1578  1880
6        Ida  F  1472  1880
7      Alice  F  1414  1880
8     Bertha  F  1320  1880
9      Sarah  F  1288  1880


head -n 10 popular-names.txt
'''
