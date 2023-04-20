import pandas as pd

df = pd.read_table("popular-names.txt", header=None)
print(df.tail(10))

# 13.4身近なライブラリに親しむ,p.172

'''
% python nlp100_15.py
             0  1      2     3
2770      Liam  M  19837  2018
2771      Noah  M  18267  2018
2772   William  M  14516  2018
2773     James  M  13525  2018
2774    Oliver  M  13389  2018
2775  Benjamin  M  13381  2018
2776    Elijah  M  12886  2018
2777     Lucas  M  12585  2018
2778     Mason  M  12435  2018
2779     Logan  M  12352  2018

tail -n 10 popular-names.txt
'''
